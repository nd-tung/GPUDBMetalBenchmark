#include "common.h"

// ===================================================================
// TPC-H Q6 KERNELS — Forecasting Revenue Change
// ===================================================================
// SELECT SUM(l_extendedprice * l_discount) AS revenue
// FROM lineitem
// WHERE l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01' 
//   AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24;

// Stage 1: Filter and compute partial revenue sums per threadgroup
kernel void q6_filter_and_sum_stage1(
    const device int* l_shipdate,        // Date as YYYYMMDD integer
    const device float* l_discount,      // Discount factor 
    const device float* l_quantity,      // Quantity
    const device float* l_extendedprice, // Extended price
    device float* partial_revenues,      // Output: partial sums per threadgroup
    constant uint& data_size,
    constant int& start_date,            // 19940101 (1994-01-01)
    constant int& end_date,              // 19950101 (1995-01-01)
    constant float& min_discount,        // 0.05
    constant float& max_discount,        // 0.07
    constant float& max_quantity,        // 24.0
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    // 1. Each thread computes a local revenue sum
    float local_revenue = 0.0f;
    
    for (uint index = (group_id * threads_per_group) + thread_id_in_group;
         index < data_size;
         index += grid_size) {
        
        // Apply all filter conditions
        if (l_shipdate[index] >= start_date && 
            l_shipdate[index] < end_date &&
            l_discount[index] >= min_discount && 
            l_discount[index] <= max_discount &&
            l_quantity[index] < max_quantity) {
            
            // Calculate revenue for this qualifying row
            local_revenue += l_extendedprice[index] * l_discount[index];
        }
    }
    
    // 2. SIMD-accelerated threadgroup reduction
    threadgroup float shared_memory[32];
    float total = tg_reduce_float(local_revenue, thread_id_in_group, threads_per_group, shared_memory);

    // 3. The first thread in each group writes the partial revenue sum
    if (thread_id_in_group == 0) {
        partial_revenues[group_id] = total;
    }
}

// Stage 2: Reduce partial revenue sums to final result
kernel void q6_final_sum_stage2(
    const device float* partial_revenues,
    device float* final_revenue,
    constant uint& num_threadgroups,
    uint index [[thread_position_in_grid]])
{
    // A single thread sums all partial revenues to get final result
    if (index == 0) {
        float total_revenue = 0.0f;
        for (uint i = 0; i < num_threadgroups; ++i) {
            total_revenue += partial_revenues[i];
        }
        final_revenue[0] = total_revenue;
    }
}

// ===================================================================
// Q6 SF100 CHUNKED EXECUTION KERNELS
// ===================================================================

kernel void q6_chunked_stage1(
    const device int*   l_shipdate        [[buffer(0)]],
    const device float* l_discount        [[buffer(1)]],
    const device float* l_quantity        [[buffer(2)]],
    const device float* l_extendedprice   [[buffer(3)]],
    device float* partial_revenues        [[buffer(4)]],
    constant uint& chunk_size             [[buffer(5)]],
    constant int&  start_date             [[buffer(6)]],
    constant int&  end_date               [[buffer(7)]],
    constant float& min_discount          [[buffer(8)]],
    constant float& max_discount          [[buffer(9)]],
    constant float& max_quantity          [[buffer(10)]],
    uint group_id           [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group  [[threads_per_threadgroup]],
    uint grid_size          [[threads_per_grid]])
{
    float local_revenue = 0.0f;
    for (uint i = (group_id * threads_per_group) + thread_id_in_group;
         i < chunk_size; i += grid_size) {
        if (l_shipdate[i] >= start_date && l_shipdate[i] < end_date &&
            l_discount[i] >= min_discount && l_discount[i] <= max_discount &&
            l_quantity[i] < max_quantity) {
            local_revenue += l_extendedprice[i] * l_discount[i];
        }
    }
    // SIMD-accelerated threadgroup reduction
    threadgroup float shared[32];
    float result = tg_reduce_float(local_revenue, thread_id_in_group, threads_per_group, shared);
    if (thread_id_in_group == 0) partial_revenues[group_id] = result;
}

// Q6 chunked stage2: reduce partials to a single value
kernel void q6_chunked_stage2(
    const device float* partial_revenues [[buffer(0)]],
    device float* final_revenue          [[buffer(1)]],
    constant uint& num_threadgroups      [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index != 0) return;
    float total = 0.0f;
    for (uint i = 0; i < num_threadgroups; ++i) total += partial_revenues[i];
    final_revenue[0] = total;
}
