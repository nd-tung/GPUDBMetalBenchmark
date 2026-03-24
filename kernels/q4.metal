#include "common.h"

// ===================================================================
// TPC-H Q4 KERNELS — Order Priority Checking
// ===================================================================
// SELECT o_orderpriority, COUNT(*) AS order_count
// FROM orders
// WHERE o_orderdate >= DATE '1993-07-01'
//   AND o_orderdate < DATE '1993-10-01'
//   AND EXISTS (SELECT * FROM lineitem
//              WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate)
// GROUP BY o_orderpriority ORDER BY o_orderpriority;
//
// Two-phase GPU approach:
//   Phase 1: Scan lineitem, build bitmap of orderkeys with late delivery
//            (l_commitdate < l_receiptdate) using atomic OR.
//   Phase 2: Scan orders with date filter + bitmap probe, count by priority.
//            Priority bins: '1'→0, '2'→1, '3'→2, '4'→3, '5'→4.

// Phase 1: Build late-delivery bitmap from lineitem
kernel void q4_build_late_bitmap(
    const device int*  l_orderkey         [[buffer(0)]],
    const device int*  l_commitdate       [[buffer(1)]],
    const device int*  l_receiptdate      [[buffer(2)]],
    device atomic_uint* late_bitmap       [[buffer(3)]],
    constant uint& data_size              [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= data_size) return;
    if (l_commitdate[tid] < l_receiptdate[tid]) {
        int key = l_orderkey[tid];
        atomic_fetch_or_explicit(&late_bitmap[(uint)key >> 5],
                                 1u << ((uint)key & 31u),
                                 memory_order_relaxed);
    }
}

// Phase 2: Scan orders, filter by date + bitmap, count by priority
kernel void q4_count_by_priority_stage1(
    const device int*  o_orderkey         [[buffer(0)]],
    const device int*  o_orderdate        [[buffer(1)]],
    const device char* o_orderpriority    [[buffer(2)]],
    const device uint* late_bitmap        [[buffer(3)]],
    device uint* partial_counts           [[buffer(4)]],  // 5 bins per TG
    constant uint& data_size              [[buffer(5)]],
    constant int&  date_start             [[buffer(6)]],
    constant int&  date_end               [[buffer(7)]],
    uint group_id           [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group  [[threads_per_threadgroup]],
    uint grid_size          [[threads_per_grid]])
{
    uint local_counts[5] = {0, 0, 0, 0, 0};

    for (uint i = (group_id * threads_per_group) + thread_id_in_group;
         i < data_size; i += grid_size) {
        int od = o_orderdate[i];
        if (od < date_start || od >= date_end) continue;
        if (!bitmap_test(late_bitmap, o_orderkey[i])) continue;

        int bin = o_orderpriority[i] - '1';  // '1'→0 .. '5'→4
        if (bin >= 0 && bin < 5) local_counts[bin] += 1;
    }

    threadgroup uint shared[32];
    for (int b = 0; b < 5; b++) {
        uint r = tg_reduce_uint(local_counts[b], thread_id_in_group, threads_per_group, shared);
        if (thread_id_in_group == 0) {
            partial_counts[group_id * 5 + b] = r;
        }
    }
}

kernel void q4_final_count_stage2(
    const device uint* partial_counts     [[buffer(0)]],
    device uint* final_counts             [[buffer(1)]],
    constant uint& num_threadgroups       [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index == 0) {
        uint totals[5] = {0, 0, 0, 0, 0};
        for (uint tg = 0; tg < num_threadgroups; tg++) {
            for (int b = 0; b < 5; b++) {
                totals[b] += partial_counts[tg * 5 + b];
            }
        }
        for (int b = 0; b < 5; b++) final_counts[b] = totals[b];
    }
}

// --- Chunked variants ---
kernel void q4_chunked_build_late_bitmap(
    const device int*  l_orderkey         [[buffer(0)]],
    const device int*  l_commitdate       [[buffer(1)]],
    const device int*  l_receiptdate      [[buffer(2)]],
    device atomic_uint* late_bitmap       [[buffer(3)]],
    constant uint& chunk_size             [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= chunk_size) return;
    if (l_commitdate[tid] < l_receiptdate[tid]) {
        int key = l_orderkey[tid];
        atomic_fetch_or_explicit(&late_bitmap[(uint)key >> 5],
                                 1u << ((uint)key & 31u),
                                 memory_order_relaxed);
    }
}

kernel void q4_chunked_count_stage1(
    const device int*  o_orderkey         [[buffer(0)]],
    const device int*  o_orderdate        [[buffer(1)]],
    const device char* o_orderpriority    [[buffer(2)]],
    const device uint* late_bitmap        [[buffer(3)]],
    device uint* partial_counts           [[buffer(4)]],
    constant uint& chunk_size             [[buffer(5)]],
    constant int&  date_start             [[buffer(6)]],
    constant int&  date_end               [[buffer(7)]],
    uint group_id           [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group  [[threads_per_threadgroup]],
    uint grid_size          [[threads_per_grid]])
{
    uint local_counts[5] = {0, 0, 0, 0, 0};

    for (uint i = (group_id * threads_per_group) + thread_id_in_group;
         i < chunk_size; i += grid_size) {
        int od = o_orderdate[i];
        if (od < date_start || od >= date_end) continue;
        if (!bitmap_test(late_bitmap, o_orderkey[i])) continue;

        int bin = o_orderpriority[i] - '1';
        if (bin >= 0 && bin < 5) local_counts[bin] += 1;
    }

    threadgroup uint shared[32];
    for (int b = 0; b < 5; b++) {
        uint r = tg_reduce_uint(local_counts[b], thread_id_in_group, threads_per_group, shared);
        if (thread_id_in_group == 0) {
            partial_counts[group_id * 5 + b] = r;
        }
    }
}

kernel void q4_chunked_final_stage2(
    const device uint* partial_counts     [[buffer(0)]],
    device uint* final_counts             [[buffer(1)]],
    constant uint& num_threadgroups       [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index == 0) {
        uint totals[5] = {0, 0, 0, 0, 0};
        for (uint tg = 0; tg < num_threadgroups; tg++) {
            for (int b = 0; b < 5; b++) {
                totals[b] += partial_counts[tg * 5 + b];
            }
        }
        for (int b = 0; b < 5; b++) final_counts[b] = totals[b];
    }
}
