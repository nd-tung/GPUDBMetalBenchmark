#include "common.h"

// ===================================================================
// TPC-H Q17 KERNELS — Small-Quantity-Order Revenue
// ===================================================================
// Two GPU passes:
// Pass 1: Aggregate sum_qty and count per partkey (for qualifying parts: Brand#23, MED BOX)
// Pass 2: Sum l_extendedprice where l_quantity < 0.2 * avg(l_quantity) for that partkey

// Pass 1: Aggregate quantity stats per qualifying partkey
kernel void q17_aggregate_qty_stats_kernel(
    const device int* l_partkey          [[buffer(0)]],
    const device float* l_quantity       [[buffer(1)]],
    const device uint* part_bitmap       [[buffer(2)]],   // qualifying parts bitmap
    device atomic_uint* sum_qty_cents_map [[buffer(3)]],  // partkey → sum(l_quantity*100)
    device atomic_uint* count_map        [[buffer(4)]],   // partkey → count
    constant uint& lineitem_size         [[buffer(5)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    uint global_id = (group_id * threads_per_group) + thread_id_in_group;
    for (uint i = global_id; i < lineitem_size; i += grid_size) {
        int pk = l_partkey[i];
        if (!bitmap_test(part_bitmap, pk)) continue;
        uint qty_cents = (uint)floor(l_quantity[i] * 100.0f + 0.5f);
        atomic_fetch_add_explicit(&sum_qty_cents_map[pk], qty_cents, memory_order_relaxed);
        atomic_fetch_add_explicit(&count_map[pk], 1u, memory_order_relaxed);
    }
}

// Pass 2: Sum extendedprice where qty < threshold
kernel void q17_sum_revenue_kernel(
    const device int* l_partkey          [[buffer(0)]],
    const device float* l_quantity       [[buffer(1)]],
    const device float* l_extendedprice  [[buffer(2)]],
    const device uint* part_bitmap       [[buffer(3)]],
    const device float* threshold_map    [[buffer(4)]],   // partkey → 0.2 * avg(qty)
    device atomic_float& total_revenue   [[buffer(5)]],
    constant uint& lineitem_size         [[buffer(6)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    float local_revenue = 0.0f;
    uint global_id = (group_id * threads_per_group) + thread_id_in_group;
    for (uint i = global_id; i < lineitem_size; i += grid_size) {
        int pk = l_partkey[i];
        if (!bitmap_test(part_bitmap, pk)) continue;
        float thresh = threshold_map[pk];
        if (thresh <= 0.0f) continue;
        if (l_quantity[i] < thresh) {
            local_revenue += l_extendedprice[i];
        }
    }

    // Collapse per-row atomics into one atomic add per threadgroup.
    threadgroup float shared[32];
    float tg_total = tg_reduce_float(local_revenue, thread_id_in_group, threads_per_group, shared);
    if (thread_id_in_group == 0) {
        atomic_fetch_add_explicit(&total_revenue, tg_total, memory_order_relaxed);
    }
}
