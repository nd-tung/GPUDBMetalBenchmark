#include "common.h"

// ===================================================================
// TPC-H Q10 KERNELS — Returned Item Reporting
// ===================================================================
// Top 20 customers by lost revenue on returned items (3-month window)

// Build orders direct map: orderkey → custkey (filtered by date)
kernel void q10_build_orders_map_kernel(
    const device int* o_orderkey     [[buffer(0)]],
    const device int* o_custkey      [[buffer(1)]],
    const device int* o_orderdate    [[buffer(2)]],
    device int* orders_map           [[buffer(3)]],
    constant uint& orders_size       [[buffer(4)]],
    constant int& date_start         [[buffer(5)]],
    constant int& date_end           [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= orders_size) return;
    int date = o_orderdate[tid];
    if (date < date_start || date >= date_end) return;
    int okey = o_orderkey[tid];
    orders_map[okey] = o_custkey[tid];
}

// Probe lineitem: filter returnflag='R', aggregate revenue per custkey
kernel void q10_probe_and_aggregate_kernel(
    const device int* l_orderkey        [[buffer(0)]],
    const device char* l_returnflag     [[buffer(1)]],
    const device float* l_extendedprice [[buffer(2)]],
    const device float* l_discount      [[buffer(3)]],
    const device int* orders_map        [[buffer(4)]],
    device atomic_float* cust_revenue   [[buffer(5)]],
    constant uint& lineitem_size        [[buffer(6)]],
    constant uint& map_size             [[buffer(7)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    uint global_id = (group_id * threads_per_group) + thread_id_in_group;

    for (uint i = global_id; i < lineitem_size; i += grid_size) {
        if (l_returnflag[i] != 'R') continue;

        int okey = l_orderkey[i];
        if ((uint)okey >= map_size) continue;
        int ck = orders_map[okey];
        if (ck == -1) continue;

        float revenue = l_extendedprice[i] * (1.0f - l_discount[i]);
        atomic_fetch_add_explicit(&cust_revenue[ck], revenue, memory_order_relaxed);
    }
}
