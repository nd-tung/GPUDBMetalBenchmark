#include "common.h"

// ===================================================================
// TPC-H Q11 KERNELS — Important Stock Identification
// ===================================================================
/*
SELECT ps_partkey, SUM(ps_supplycost * ps_availqty) AS value
FROM partsupp, supplier, nation
WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey
  AND n_name = 'GERMANY'
GROUP BY ps_partkey
HAVING SUM(ps_supplycost * ps_availqty) > (
    SELECT SUM(ps_supplycost * ps_availqty) * 0.0001
    FROM partsupp, supplier, nation
    WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey
      AND n_name = 'GERMANY'
)
ORDER BY value DESC;
*/

// Single-kernel approach:
// 1. Scan partsupp, filter by supplier bitmap (GERMANY suppliers)
// 2. Atomic-add ps_supplycost * ps_availqty into value_map[ps_partkey]
// 3. Threadgroup reduction for global sum -> partial_sums[]
// CPU then: threshold = sum(partial_sums) * fraction, scan value_map for > threshold

kernel void q11_aggregate_kernel(
    const device int*   ps_partkey    [[buffer(0)]],
    const device int*   ps_suppkey    [[buffer(1)]],
    const device float* ps_supplycost [[buffer(2)]],
    const device int*   ps_availqty   [[buffer(3)]],
    const device uint*  supp_bitmap   [[buffer(4)]],
    device atomic_float* value_map    [[buffer(5)]],
    device float*       partial_sums  [[buffer(6)]],
    constant uint&      data_size     [[buffer(7)]],
    uint group_id           [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group  [[threads_per_threadgroup]],
    uint grid_size          [[threads_per_grid]])
{
    float local_sum = 0.0f;
    uint gid = group_id * threads_per_group + thread_id_in_group;

    for (uint i = gid; i < data_size; i += grid_size) {
        int sk = ps_suppkey[i];
        if (!bitmap_test(supp_bitmap, sk)) continue;

        float val = ps_supplycost[i] * float(ps_availqty[i]);
        atomic_fetch_add_explicit(&value_map[ps_partkey[i]], val, memory_order_relaxed);
        local_sum += val;
    }

    // Threadgroup reduction for global sum
    threadgroup float shared[32];
    float total = tg_reduce_float(local_sum, thread_id_in_group, threads_per_group, shared);
    if (thread_id_in_group == 0) {
        partial_sums[group_id] = total;
    }
}
