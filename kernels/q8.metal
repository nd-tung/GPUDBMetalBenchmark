#include "common.h"

// ===================================================================
// TPC-H Q8 KERNELS — National Market Share
// ===================================================================
// Market share of BRAZIL in AMERICA region for ECONOMY ANODIZED STEEL (1995-1996)
// Result: 4 atomic floats: [brazil_95, brazil_96, total_95, total_96]

// Build orders map: orderkey → custkey + year (only for orders in date range + AMERICA customer)
kernel void q8_build_orders_map_kernel(
    const device int* o_orderkey        [[buffer(0)]],
    const device int* o_custkey         [[buffer(1)]],
    const device int* o_orderdate       [[buffer(2)]],
    device int* orders_custkey_map      [[buffer(3)]],
    device int* orders_year_map         [[buffer(4)]],
    const device int* cust_nation_map   [[buffer(5)]],
    constant uint& orders_size          [[buffer(6)]],
    constant int& date_start            [[buffer(7)]],
    constant int& date_end              [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= orders_size) return;
    int date = o_orderdate[tid];
    if (date < date_start || date > date_end) return;
    int ck = o_custkey[tid];
    int cust_nk = cust_nation_map[ck];
    if (cust_nk == -1) return;  // customer not in AMERICA
    int okey = o_orderkey[tid];
    orders_custkey_map[okey] = ck;
    orders_year_map[okey] = date / 10000;  // 1995 or 1996
}

// Probe lineitem: check part bitmap, lookup chains, aggregate
kernel void q8_probe_and_aggregate_kernel(
    const device int* l_orderkey        [[buffer(0)]],
    const device int* l_partkey         [[buffer(1)]],
    const device int* l_suppkey         [[buffer(2)]],
    const device float* l_extendedprice [[buffer(3)]],
    const device float* l_discount      [[buffer(4)]],
    const device uint* part_bitmap      [[buffer(5)]],
    const device int* orders_custkey_map [[buffer(6)]],
    const device int* orders_year_map   [[buffer(7)]],
    const device int* supp_nation_map   [[buffer(8)]],
    device atomic_float* result_bins    [[buffer(9)]],
    constant uint& lineitem_size        [[buffer(10)]],
    constant uint& orders_map_size      [[buffer(11)]],
    constant int& brazil_nk             [[buffer(12)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    uint global_id = (group_id * threads_per_group) + thread_id_in_group;

    for (uint i = global_id; i < lineitem_size; i += grid_size) {
        int pk = l_partkey[i];
        if (!bitmap_test(part_bitmap, pk)) continue;

        int okey = l_orderkey[i];
        if ((uint)okey >= orders_map_size) continue;
        int ck = orders_custkey_map[okey];
        if (ck == -1) continue;

        int year = orders_year_map[okey];
        int year_idx = year - 1995;
        if (year_idx < 0 || year_idx > 1) continue;

        float revenue = l_extendedprice[i] * (1.0f - l_discount[i]);
        atomic_fetch_add_explicit(&result_bins[2 + year_idx], revenue, memory_order_relaxed);

        int supp_nk = supp_nation_map[l_suppkey[i]];
        if (supp_nk == brazil_nk) {
            atomic_fetch_add_explicit(&result_bins[year_idx], revenue, memory_order_relaxed);
        }
    }
}
