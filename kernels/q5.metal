#include "common.h"

// ===================================================================
// TPC-H Q5 KERNELS — Local Supplier Volume
// ===================================================================
/*
SELECT n_name, SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM customer, orders, lineitem, supplier, nation, region
WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey
  AND l_suppkey = s_suppkey AND c_nationkey = s_nationkey
  AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey
  AND r_name = 'ASIA'
  AND o_orderdate >= '1994-01-01' AND o_orderdate < '1995-01-01'
ORDER BY revenue DESC;
*/

// KERNEL 1: Build customer -> nationkey direct map
// Only inserts customers whose nationkey is in the ASIA region (via nation_bitmap).
kernel void q5_build_customer_nation_map_kernel(
    const device int* c_custkey     [[buffer(0)]],
    const device int* c_nationkey   [[buffer(1)]],
    device int* customer_nation_map [[buffer(2)]],
    const device uint* nation_bitmap [[buffer(3)]],
    constant uint& customer_size    [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= customer_size) return;

    int nk = c_nationkey[index];
    if (!bitmap_test(nation_bitmap, nk)) return;

    int key = c_custkey[index];
    customer_nation_map[key] = nk;
}

// KERNEL 2: Build supplier -> nationkey direct map
// Only inserts suppliers whose nationkey is in the ASIA region.
kernel void q5_build_supplier_nation_map_kernel(
    const device int* s_suppkey      [[buffer(0)]],
    const device int* s_nationkey    [[buffer(1)]],
    device int* supplier_nation_map  [[buffer(2)]],
    const device uint* nation_bitmap [[buffer(3)]],
    constant uint& supplier_size     [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= supplier_size) return;

    int nk = s_nationkey[index];
    if (!bitmap_test(nation_bitmap, nk)) return;

    int key = s_suppkey[index];
    supplier_nation_map[key] = nk;
}

// KERNEL 3: Build orders -> nationkey direct map filtered by date range and customer-in-ASIA
// Direct indexed write: orders_nation_map[orderkey] = nationkey. No atomics, no collisions.
kernel void q5_build_orders_map_kernel(
    const device int* o_orderkey           [[buffer(0)]],
    const device int* o_custkey            [[buffer(1)]],
    const device int* o_orderdate          [[buffer(2)]],
    device int* orders_nation_map          [[buffer(3)]],
    constant uint& orders_size             [[buffer(4)]],
    constant int& date_start               [[buffer(5)]],
    constant int& date_end                 [[buffer(6)]],
    constant uint& map_size                [[buffer(7)]],
    const device int* customer_nation_map  [[buffer(8)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= orders_size) return;

    int date = o_orderdate[index];
    if (date < date_start || date >= date_end) return;

    int ck = o_custkey[index];
    int cust_nk = customer_nation_map[ck];
    if (cust_nk == -1) return;

    int key = o_orderkey[index];
    if ((uint)key < map_size)
        orders_nation_map[key] = cust_nk;
}

// KERNEL 4: Probe lineitem -> orders direct map, check same-nation constraint,
// aggregate revenue by nationkey into a 25-element atomic_float array.
kernel void q5_probe_and_aggregate_kernel(
    const device int* l_orderkey        [[buffer(0)]],
    const device int* l_suppkey         [[buffer(1)]],
    const device float* l_extendedprice [[buffer(2)]],
    const device float* l_discount      [[buffer(3)]],
    const device int* orders_nation_map [[buffer(4)]],
    const device int* supplier_nation_map [[buffer(5)]],
    device atomic_float* nation_revenue [[buffer(6)]],
    constant uint& lineitem_size        [[buffer(7)]],
    constant uint& map_size             [[buffer(8)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    uint global_id = (group_id * threads_per_group) + thread_id_in_group;
    const int BATCH_SIZE = 4;

    for (uint i = global_id; i < lineitem_size; i += grid_size * BATCH_SIZE) {
        for (int k = 0; k < BATCH_SIZE; k++) {
            uint idx = i + k * grid_size;
            if (idx >= lineitem_size) break;

            int orderkey = l_orderkey[idx];
            if ((uint)orderkey >= map_size) continue;

            // Direct map lookup — O(1), no hash collisions
            int cust_nationkey = orders_nation_map[orderkey];
            if (cust_nationkey == -1) continue;

            // Check supplier is in ASIA and same nation as customer
            int suppkey = l_suppkey[idx];
            int supp_nationkey = supplier_nation_map[suppkey];
            if (supp_nationkey != cust_nationkey) continue;

            // Aggregate revenue
            float revenue = l_extendedprice[idx] * (1.0f - l_discount[idx]);
            atomic_fetch_add_explicit(&nation_revenue[cust_nationkey], revenue, memory_order_relaxed);
        }
    }
}
