#include "common.h"

// ===================================================================
// TPC-H Q20 KERNELS — Potential Part Promotion
// ===================================================================
// GPU aggregates SUM(l_quantity) into a hash table keyed by (partkey, suppkey).
// GPU then scans partsupp, probes HT, applies threshold, sets qualifying bitmap.

struct Q20HTEntry {
    atomic_int key_hi;   // partkey or -1 (empty)
    atomic_int key_lo;   // suppkey
    atomic_float value;  // sum(l_quantity)
};

kernel void q20_aggregate_lineitem_kernel(
    const device int* l_partkey         [[buffer(0)]],
    const device int* l_suppkey         [[buffer(1)]],
    const device float* l_quantity      [[buffer(2)]],
    const device int* l_shipdate        [[buffer(3)]],
    const device uint* part_bitmap      [[buffer(4)]],  // forest% parts
    device Q20HTEntry* ht               [[buffer(5)]],
    constant uint& lineitem_size        [[buffer(6)]],
    constant uint& ht_mask              [[buffer(7)]],
    constant int& date_start            [[buffer(8)]],
    constant int& date_end              [[buffer(9)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    uint global_id = (group_id * threads_per_group) + thread_id_in_group;
    for (uint i = global_id; i < lineitem_size; i += grid_size) {
        int date = l_shipdate[i];
        if (date < date_start || date >= date_end) continue;
        int pk = l_partkey[i];
        if (!bitmap_test(part_bitmap, pk)) continue;
        int sk = l_suppkey[i];
        float qty = l_quantity[i];

        // Hash on combined key
        uint h = sf100_hash_key(pk * 100001 + sk);
        for (uint j = 0; j <= ht_mask; j++) {
            uint slot = (h + j) & ht_mask;
            int expected = -1;
            // Try to claim slot
            if (atomic_compare_exchange_weak_explicit(&ht[slot].key_hi, &expected, pk,
                    memory_order_relaxed, memory_order_relaxed)) {
                // Won the slot
                atomic_store_explicit(&ht[slot].key_lo, sk, memory_order_relaxed);
                atomic_fetch_add_explicit(&ht[slot].value, qty, memory_order_relaxed);
                break;
            }
            // Check if this slot has our key
            int cur_pk = atomic_load_explicit(&ht[slot].key_hi, memory_order_relaxed);
            int cur_sk = atomic_load_explicit(&ht[slot].key_lo, memory_order_relaxed);
            if (cur_pk == pk && cur_sk == sk) {
                atomic_fetch_add_explicit(&ht[slot].value, qty, memory_order_relaxed);
                break;
            }
        }
    }
}

// GPU kernel to probe partsupp against Q20 HT and set qualifying suppkey bitmap
kernel void q20_probe_partsupp_kernel(
    const device int* ps_partkey          [[buffer(0)]],
    const device int* ps_suppkey          [[buffer(1)]],
    const device int* ps_availqty         [[buffer(2)]],
    const device uint* part_bitmap        [[buffer(3)]],  // forest% parts
    const device uint* canada_bitmap      [[buffer(4)]],  // CANADA suppliers
    const device Q20HTEntry* ht           [[buffer(5)]],
    device atomic_uint* qual_bitmap       [[buffer(6)]],  // output: qualifying suppkeys
    constant uint& partsupp_size          [[buffer(7)]],
    constant uint& ht_mask                [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= partsupp_size) return;

    int pk = ps_partkey[tid];
    if (!bitmap_test(part_bitmap, pk)) return;

    int sk = ps_suppkey[tid];
    if (!bitmap_test(canada_bitmap, sk)) return;

    // Probe HT for (pk, sk)
    uint h = sf100_hash_key(pk * 100001 + sk);
    for (uint j = 0; j <= ht_mask; j++) {
        uint slot = (h + j) & ht_mask;
        int cur_pk = atomic_load_explicit(&ht[slot].key_hi, memory_order_relaxed);
        if (cur_pk == -1) break;  // empty
        if (cur_pk == pk) {
            int cur_sk = atomic_load_explicit(&ht[slot].key_lo, memory_order_relaxed);
            if (cur_sk == sk) {
                float sum_qty = atomic_load_explicit(&ht[slot].value, memory_order_relaxed);
                if ((float)ps_availqty[tid] > 0.5f * sum_qty) {
                    // Set bit in qualifying bitmap
                    atomic_fetch_or_explicit(&qual_bitmap[(uint)sk >> 5],
                                            1u << ((uint)sk & 31u), memory_order_relaxed);
                }
                return;
            }
        }
    }
}
