#include "common.h"

// ===================================================================
// TPC-H Q7 KERNELS — Volume Shipping
// ===================================================================
// Revenue between FRANCE and GERMANY for years 1995-1996
// Result bins: 4 = 2 nation pairs × 2 years
// bin = pair_idx * 2 + year_idx
// pair 0: (FRANCE→GERMANY), pair 1: (GERMANY→FRANCE)
// year 0: 1995, year 1: 1996

// Build supplier map: suppkey → nationkey (only FRANCE/GERMANY)
kernel void q7_build_supplier_map_kernel(
    const device int* s_suppkey     [[buffer(0)]],
    const device int* s_nationkey   [[buffer(1)]],
    device int* supp_nation_map     [[buffer(2)]],
    constant int& france_nk         [[buffer(3)]],
    constant int& germany_nk        [[buffer(4)]],
    constant uint& supplier_size    [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= supplier_size) return;
    int nk = s_nationkey[tid];
    if (nk == france_nk || nk == germany_nk)
        supp_nation_map[s_suppkey[tid]] = nk;
}

// Build customer map: custkey → nationkey (only FRANCE/GERMANY)
kernel void q7_build_customer_map_kernel(
    const device int* c_custkey     [[buffer(0)]],
    const device int* c_nationkey   [[buffer(1)]],
    device int* cust_nation_map     [[buffer(2)]],
    constant int& france_nk         [[buffer(3)]],
    constant int& germany_nk        [[buffer(4)]],
    constant uint& customer_size    [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= customer_size) return;
    int nk = c_nationkey[tid];
    if (nk == france_nk || nk == germany_nk)
        cust_nation_map[c_custkey[tid]] = nk;
}

// Build orders map: orderkey → custkey
kernel void q7_build_orders_map_kernel(
    const device int* o_orderkey    [[buffer(0)]],
    const device int* o_custkey     [[buffer(1)]],
    device int* orders_map          [[buffer(2)]],
    constant uint& orders_size      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= orders_size) return;
    orders_map[o_orderkey[tid]] = o_custkey[tid];
}

// Probe lineitem: check date, lookup chains, aggregate into 4 bins
kernel void q7_probe_and_aggregate_kernel(
    const device int* l_orderkey        [[buffer(0)]],
    const device int* l_suppkey         [[buffer(1)]],
    const device int* l_shipdate        [[buffer(2)]],
    const device float* l_extendedprice [[buffer(3)]],
    const device float* l_discount      [[buffer(4)]],
    const device int* orders_map        [[buffer(5)]],
    const device int* cust_nation_map   [[buffer(6)]],
    const device int* supp_nation_map   [[buffer(7)]],
    device atomic_float* revenue_bins   [[buffer(8)]],  // 4 bins
    constant uint& lineitem_size        [[buffer(9)]],
    constant uint& orders_map_size      [[buffer(10)]],
    constant int& france_nk             [[buffer(11)]],
    constant int& germany_nk            [[buffer(12)]],
    constant int& date_start            [[buffer(13)]],
    constant int& date_end              [[buffer(14)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    uint global_id = (group_id * threads_per_group) + thread_id_in_group;

    for (uint i = global_id; i < lineitem_size; i += grid_size) {
        int shipdate = l_shipdate[i];
        if (shipdate < date_start || shipdate > date_end) continue;

        int okey = l_orderkey[i];
        if ((uint)okey >= orders_map_size) continue;
        int ck = orders_map[okey];
        if (ck == -1) continue;

        int supp_nk = supp_nation_map[l_suppkey[i]];
        if (supp_nk == -1) continue;
        int cust_nk = cust_nation_map[ck];
        if (cust_nk == -1) continue;

        // Check if valid pair: (FRANCE,GERMANY) or (GERMANY,FRANCE)
        int pair_idx;
        if (supp_nk == france_nk && cust_nk == germany_nk) pair_idx = 0;
        else if (supp_nk == germany_nk && cust_nk == france_nk) pair_idx = 1;
        else continue;

        int year = shipdate / 10000;
        int year_idx;
        if (year == 1995) year_idx = 0;
        else if (year == 1996) year_idx = 1;
        else continue;

        float revenue = l_extendedprice[i] * (1.0f - l_discount[i]);
        atomic_fetch_add_explicit(&revenue_bins[pair_idx * 2 + year_idx], revenue, memory_order_relaxed);
    }
}
