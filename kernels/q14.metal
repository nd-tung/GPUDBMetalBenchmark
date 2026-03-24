#include "common.h"

// ===================================================================
// TPC-H Q14 KERNELS — Promotion Effect
// ===================================================================
// SELECT 100.00 * SUM(CASE WHEN p_type LIKE 'PROMO%'
//            THEN l_extendedprice * (1 - l_discount) ELSE 0 END)
//        / SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue
// FROM lineitem, part
// WHERE l_partkey = p_partkey
//   AND l_shipdate >= DATE '1995-09-01'
//   AND l_shipdate < DATE '1995-10-01';
//
// GPU approach: promo_bitmap built on CPU (part.p_type LIKE 'PROMO%'),
// scan lineitem with date filter, accumulate promo_sum and total_sum.

kernel void q14_filter_and_sum_stage1(
    const device int*   l_partkey         [[buffer(0)]],
    const device int*   l_shipdate        [[buffer(1)]],
    const device float* l_extendedprice   [[buffer(2)]],
    const device float* l_discount        [[buffer(3)]],
    const device uint*  promo_bitmap      [[buffer(4)]],
    device float* partial_promo           [[buffer(5)]],
    device float* partial_total           [[buffer(6)]],
    constant uint& data_size              [[buffer(7)]],
    constant int&  start_date             [[buffer(8)]],
    constant int&  end_date               [[buffer(9)]],
    uint group_id           [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group  [[threads_per_threadgroup]],
    uint grid_size          [[threads_per_grid]])
{
    float local_promo = 0.0f;
    float local_total = 0.0f;

    for (uint i = (group_id * threads_per_group) + thread_id_in_group;
         i < data_size; i += grid_size) {
        int sd = l_shipdate[i];
        if (sd >= start_date && sd < end_date) {
            float revenue = l_extendedprice[i] * (1.0f - l_discount[i]);
            local_total += revenue;
            if (bitmap_test(promo_bitmap, l_partkey[i])) {
                local_promo += revenue;
            }
        }
    }

    threadgroup float shared[32];
    float promo_sum = tg_reduce_float(local_promo, thread_id_in_group, threads_per_group, shared);
    float total_sum = tg_reduce_float(local_total, thread_id_in_group, threads_per_group, shared);

    if (thread_id_in_group == 0) {
        partial_promo[group_id] = promo_sum;
        partial_total[group_id] = total_sum;
    }
}

kernel void q14_final_sum_stage2(
    const device float* partial_promo     [[buffer(0)]],
    const device float* partial_total     [[buffer(1)]],
    device float* final_result            [[buffer(2)]],
    constant uint& num_threadgroups       [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index == 0) {
        float promo = 0.0f, total = 0.0f;
        for (uint i = 0; i < num_threadgroups; ++i) {
            promo += partial_promo[i];
            total += partial_total[i];
        }
        final_result[0] = promo;
        final_result[1] = total;
    }
}

// --- Chunked variants ---
kernel void q14_chunked_stage1(
    const device int*   l_partkey         [[buffer(0)]],
    const device int*   l_shipdate        [[buffer(1)]],
    const device float* l_extendedprice   [[buffer(2)]],
    const device float* l_discount        [[buffer(3)]],
    const device uint*  promo_bitmap      [[buffer(4)]],
    device float* partial_promo           [[buffer(5)]],
    device float* partial_total           [[buffer(6)]],
    constant uint& chunk_size             [[buffer(7)]],
    constant int&  start_date             [[buffer(8)]],
    constant int&  end_date               [[buffer(9)]],
    uint group_id           [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group  [[threads_per_threadgroup]],
    uint grid_size          [[threads_per_grid]])
{
    float local_promo = 0.0f;
    float local_total = 0.0f;

    for (uint i = (group_id * threads_per_group) + thread_id_in_group;
         i < chunk_size; i += grid_size) {
        int sd = l_shipdate[i];
        if (sd >= start_date && sd < end_date) {
            float revenue = l_extendedprice[i] * (1.0f - l_discount[i]);
            local_total += revenue;
            if (bitmap_test(promo_bitmap, l_partkey[i])) {
                local_promo += revenue;
            }
        }
    }

    threadgroup float shared[32];
    float promo_sum = tg_reduce_float(local_promo, thread_id_in_group, threads_per_group, shared);
    float total_sum = tg_reduce_float(local_total, thread_id_in_group, threads_per_group, shared);

    if (thread_id_in_group == 0) {
        partial_promo[group_id] = promo_sum;
        partial_total[group_id] = total_sum;
    }
}

kernel void q14_chunked_stage2(
    const device float* partial_promo     [[buffer(0)]],
    const device float* partial_total     [[buffer(1)]],
    device float* final_result            [[buffer(2)]],
    constant uint& num_threadgroups       [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index == 0) {
        float promo = 0.0f, total = 0.0f;
        for (uint i = 0; i < num_threadgroups; ++i) {
            promo += partial_promo[i];
            total += partial_total[i];
        }
        final_result[0] = promo;
        final_result[1] = total;
    }
}
