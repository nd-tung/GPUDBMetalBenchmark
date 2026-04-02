#include "common.h"

// ===================================================================
// TPC-H Q18 KERNELS — Large Volume Customer
// ===================================================================
// GPU aggregates SUM(l_quantity) per orderkey into direct map,
// then filters orders with sum > 300 and compacts results.

kernel void q18_aggregate_quantity_kernel(
    const device int* l_orderkey        [[buffer(0)]],
    const device float* l_quantity      [[buffer(1)]],
    device atomic_float* qty_map        [[buffer(2)]],  // orderkey → sum(quantity)
    constant uint& lineitem_size        [[buffer(3)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    uint global_id = (group_id * threads_per_group) + thread_id_in_group;
    for (uint i = global_id; i < lineitem_size; i += grid_size) {
        int okey = l_orderkey[i];
        float qty = l_quantity[i];
        atomic_fetch_add_explicit(&qty_map[okey], qty, memory_order_relaxed);
    }
}

// Filter orders with sum(qty) > threshold, compact qualifying rows
struct Q18OutputRow {
    int o_orderkey;
    int o_custkey;
    int o_orderdate;
    float o_totalprice;
    float sum_qty;
};

kernel void q18_filter_orders_kernel(
    const device int* o_orderkey        [[buffer(0)]],
    const device int* o_custkey         [[buffer(1)]],
    const device int* o_orderdate       [[buffer(2)]],
    const device float* o_totalprice    [[buffer(3)]],
    const device float* qty_map         [[buffer(4)]],
    device Q18OutputRow* output         [[buffer(5)]],
    device atomic_uint& output_count    [[buffer(6)]],
    constant uint& orders_size          [[buffer(7)]],
    constant uint& qty_map_size         [[buffer(8)]],
    constant float& threshold           [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= orders_size) return;
    int okey = o_orderkey[tid];
    if ((uint)okey >= qty_map_size) return;
    float sq = qty_map[okey];
    if (sq <= threshold) return;

    uint pos = atomic_fetch_add_explicit(&output_count, 1u, memory_order_relaxed);
    output[pos].o_orderkey = okey;
    output[pos].o_custkey = o_custkey[tid];
    output[pos].o_orderdate = o_orderdate[tid];
    output[pos].o_totalprice = o_totalprice[tid];
    output[pos].sum_qty = sq;
}
