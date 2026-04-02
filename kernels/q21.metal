#include "common.h"

// ===================================================================
// TPC-H Q21 KERNELS — Suppliers Who Kept Orders Waiting
// ===================================================================
// For each failed order (o_orderstatus = 'F'):
// - Find lineitems where l_receiptdate > l_commitdate (late)
// - The supplier was the ONLY late supplier on that order
// - But there must be at least one OTHER supplier on the order (EXISTS l2)
//
// Strategy: Two-pass approach using per-order tracking.
//
// Pass 1: For each lineitem on a failed order, track:
//   - multi_supp_bitmap[orderkey]: set if order has multiple distinct suppkeys
//   - first_supp[orderkey]: first suppkey seen (via CAS)
//   - For late items: late_supp[orderkey] via CAS, multi_late_bitmap if multiple late supps
//
// Pass 2: For each late lineitem on a failed order:
//   - Check if order has multi_supp (EXISTS condition)
//   - Check if this supplier is the only late one (NOT EXISTS other late)
//   - If supplier is in SAUDI ARABIA, count it

// Pass 1: Build per-order supplier tracking
kernel void q21_build_order_tracking_kernel(
    const device int* l_orderkey         [[buffer(0)]],
    const device int* l_suppkey          [[buffer(1)]],
    const device int* l_receiptdate     [[buffer(2)]],
    const device int* l_commitdate      [[buffer(3)]],
    const device int* orders_status_map  [[buffer(4)]],  // orderkey → 1 if 'F', -1 otherwise
    device atomic_int* first_supp        [[buffer(5)]],  // orderkey → first suppkey seen
    device atomic_uint* multi_supp_bm    [[buffer(6)]],  // bitmap: order has multiple supps
    device atomic_int* late_supp         [[buffer(7)]],  // orderkey → first late suppkey
    device atomic_uint* multi_late_bm    [[buffer(8)]],  // bitmap: order has multiple late supps
    constant uint& lineitem_size         [[buffer(9)]],
    constant uint& map_size              [[buffer(10)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    uint global_id = (group_id * threads_per_group) + thread_id_in_group;
    for (uint i = global_id; i < lineitem_size; i += grid_size) {
        int okey = l_orderkey[i];
        if ((uint)okey >= map_size) continue;
        int status = orders_status_map[okey];
        if (status != 1) continue;  // only failed orders

        int sk = l_suppkey[i];

        // Track distinct suppliers
        int expected = -1;
        if (!atomic_compare_exchange_weak_explicit(&first_supp[okey], &expected, sk,
                memory_order_relaxed, memory_order_relaxed)) {
            // Slot taken — if different suppkey, mark multi_supp
            if (expected != sk) {
                bitmap_set(multi_supp_bm, okey);
            }
        }

        // Track late suppliers
        bool is_late = (l_receiptdate[i] > l_commitdate[i]);
        if (is_late) {
            int exp_late = -1;
            if (!atomic_compare_exchange_weak_explicit(&late_supp[okey], &exp_late, sk,
                    memory_order_relaxed, memory_order_relaxed)) {
                if (exp_late != sk) {
                    bitmap_set(multi_late_bm, okey);
                }
            }
        }
    }
}

// Pass 2: Count qualifying lineitems per SAUDI ARABIA supplier
kernel void q21_count_qualifying_kernel(
    const device int* l_orderkey         [[buffer(0)]],
    const device int* l_suppkey          [[buffer(1)]],
    const device int* l_receiptdate     [[buffer(2)]],
    const device int* l_commitdate      [[buffer(3)]],
    const device int* orders_status_map  [[buffer(4)]],
    const device int* first_supp         [[buffer(5)]],
    const device uint* multi_supp_bm     [[buffer(6)]],
    const device int* late_supp          [[buffer(7)]],
    const device uint* multi_late_bm     [[buffer(8)]],
    const device uint* sa_supp_bitmap    [[buffer(9)]],   // SAUDI ARABIA supplier bitmap
    device atomic_uint* supp_count       [[buffer(10)]],  // suppkey → count
    constant uint& lineitem_size         [[buffer(11)]],
    constant uint& map_size              [[buffer(12)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    uint global_id = (group_id * threads_per_group) + thread_id_in_group;
    for (uint i = global_id; i < lineitem_size; i += grid_size) {
        int okey = l_orderkey[i];
        if ((uint)okey >= map_size) continue;
        int status = orders_status_map[okey];
        if (status != 1) continue;

        int sk = l_suppkey[i];
        // Must be late
        if (l_receiptdate[i] <= l_commitdate[i]) continue;
        // Must be SAUDI ARABIA supplier
        if (!bitmap_test(sa_supp_bitmap, sk)) continue;
        // EXISTS l2: order has multiple suppliers
        if (!bitmap_test(multi_supp_bm, okey)) {
            // Check if first_supp != sk (means at least 2 suppliers, but we track via CAS)
            // If first_supp[okey] == sk and no multi_supp bit, only one supplier → skip
            continue;
        }
        // NOT EXISTS l3: no other late supplier
        if (bitmap_test(multi_late_bm, okey)) continue;  // multiple late → skip
        // late_supp[okey] should be sk (we are the only late one)
        if (late_supp[okey] != sk) continue;

        atomic_fetch_add_explicit(&supp_count[sk], 1u, memory_order_relaxed);
    }
}
