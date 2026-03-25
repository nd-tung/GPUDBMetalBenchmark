#include "common.h"

// ===================================================================
// TPC-H Q12 KERNELS — Shipping Modes and Order Priority
// ===================================================================
// SELECT l_shipmode,
//   SUM(CASE WHEN o_orderpriority IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS high_line_count,
//   SUM(CASE WHEN o_orderpriority NOT IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS low_line_count
// FROM orders, lineitem
// WHERE o_orderkey = l_orderkey
//   AND l_shipmode IN ('MAIL', 'SHIP')
//   AND l_commitdate < l_receiptdate
//   AND l_shipdate < l_commitdate
//   AND l_receiptdate >= DATE '1994-01-01'
//   AND l_receiptdate < DATE '1995-01-01'
// GROUP BY l_shipmode ORDER BY l_shipmode;
//
// GPU approach: priority_bitmap built on CPU (orderkey where o_orderpriority is HIGH/URGENT),
// scan lineitem with shipmode + date filters, bucket into 4 bins: {MAIL,SHIP} x {high,low}.
// shipmode identified by first char: 'M'=MAIL, 'S'=SHIP.

kernel void q12_filter_and_count_stage1(
    const device int*   l_orderkey        [[buffer(0)]],
    const device char*  l_shipmode        [[buffer(1)]],
    const device int*   l_shipdate        [[buffer(2)]],
    const device int*   l_commitdate      [[buffer(3)]],
    const device int*   l_receiptdate     [[buffer(4)]],
    const device uint*  priority_bitmap   [[buffer(5)]],
    device uint* partial_counts           [[buffer(6)]],  // 4 bins per TG
    constant uint& data_size              [[buffer(7)]],
    constant int&  receipt_start          [[buffer(8)]],
    constant int&  receipt_end            [[buffer(9)]],
    uint group_id           [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group  [[threads_per_threadgroup]],
    uint grid_size          [[threads_per_grid]])
{
    // 4 bins: 0=MAIL-HIGH, 1=MAIL-LOW, 2=SHIP-HIGH, 3=SHIP-LOW
    uint local_counts[4] = {0, 0, 0, 0};

    for (uint i = (group_id * threads_per_group) + thread_id_in_group;
         i < data_size; i += grid_size) {
        char sm = l_shipmode[i];
        int mode_idx = -1;
        if (sm == 'M') mode_idx = 0;       // MAIL
        else if (sm == 'S') mode_idx = 2;   // SHIP
        if (mode_idx < 0) continue;

        int rd = l_receiptdate[i];
        if (rd < receipt_start || rd >= receipt_end) continue;
        if (l_commitdate[i] >= rd) continue;           // l_commitdate < l_receiptdate
        if (l_shipdate[i] >= l_commitdate[i]) continue; // l_shipdate < l_commitdate

        bool is_high = bitmap_test(priority_bitmap, l_orderkey[i]);
        local_counts[mode_idx + (is_high ? 0 : 1)] += 1;
    }

    threadgroup uint shared[32];
    for (int b = 0; b < 4; b++) {
        uint r = tg_reduce_uint(local_counts[b], thread_id_in_group, threads_per_group, shared);
        if (thread_id_in_group == 0) {
            partial_counts[group_id * 4 + b] = r;
        }
    }
}

kernel void q12_final_count_stage2(
    const device uint* partial_counts     [[buffer(0)]],
    device uint* final_counts             [[buffer(1)]],
    constant uint& num_threadgroups       [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index == 0) {
        uint totals[4] = {0, 0, 0, 0};
        for (uint tg = 0; tg < num_threadgroups; tg++) {
            for (int b = 0; b < 4; b++) {
                totals[b] += partial_counts[tg * 4 + b];
            }
        }
        for (int b = 0; b < 4; b++) final_counts[b] = totals[b];
    }
}

// ===================================================================
// Pre-computation: Build priority bitmap on GPU
// Sets bit for each orderkey where o_orderpriority is '1' (1-URGENT) or '2' (2-HIGH)
// ===================================================================
kernel void q12_build_priority_bitmap(
    const device int*   o_orderkey        [[buffer(0)]],
    const device char*  o_orderpriority   [[buffer(1)]],
    device atomic_uint* priority_bitmap   [[buffer(2)]],
    constant uint& data_size              [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= data_size) return;
    char p = o_orderpriority[tid];
    if (p == '1' || p == '2') {
        bitmap_set(priority_bitmap, o_orderkey[tid]);
    }
}

kernel void q12_chunked_build_priority_bitmap(
    const device int*   o_orderkey        [[buffer(0)]],
    const device char*  o_orderpriority   [[buffer(1)]],
    device atomic_uint* priority_bitmap   [[buffer(2)]],
    constant uint& chunk_size             [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= chunk_size) return;
    char p = o_orderpriority[tid];
    if (p == '1' || p == '2') {
        bitmap_set(priority_bitmap, o_orderkey[tid]);
    }
}

// --- Chunked variants ---
kernel void q12_chunked_stage1(
    const device int*   l_orderkey        [[buffer(0)]],
    const device char*  l_shipmode        [[buffer(1)]],
    const device int*   l_shipdate        [[buffer(2)]],
    const device int*   l_commitdate      [[buffer(3)]],
    const device int*   l_receiptdate     [[buffer(4)]],
    const device uint*  priority_bitmap   [[buffer(5)]],
    device uint* partial_counts           [[buffer(6)]],
    constant uint& chunk_size             [[buffer(7)]],
    constant int&  receipt_start          [[buffer(8)]],
    constant int&  receipt_end            [[buffer(9)]],
    uint group_id           [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group  [[threads_per_threadgroup]],
    uint grid_size          [[threads_per_grid]])
{
    uint local_counts[4] = {0, 0, 0, 0};

    for (uint i = (group_id * threads_per_group) + thread_id_in_group;
         i < chunk_size; i += grid_size) {
        char sm = l_shipmode[i];
        int mode_idx = -1;
        if (sm == 'M') mode_idx = 0;
        else if (sm == 'S') mode_idx = 2;
        if (mode_idx < 0) continue;

        int rd = l_receiptdate[i];
        if (rd < receipt_start || rd >= receipt_end) continue;
        if (l_commitdate[i] >= rd) continue;
        if (l_shipdate[i] >= l_commitdate[i]) continue;

        bool is_high = bitmap_test(priority_bitmap, l_orderkey[i]);
        local_counts[mode_idx + (is_high ? 0 : 1)] += 1;
    }

    threadgroup uint shared[32];
    for (int b = 0; b < 4; b++) {
        uint r = tg_reduce_uint(local_counts[b], thread_id_in_group, threads_per_group, shared);
        if (thread_id_in_group == 0) {
            partial_counts[group_id * 4 + b] = r;
        }
    }
}

kernel void q12_chunked_stage2(
    const device uint* partial_counts     [[buffer(0)]],
    device uint* final_counts             [[buffer(1)]],
    constant uint& num_threadgroups       [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index == 0) {
        uint totals[4] = {0, 0, 0, 0};
        for (uint tg = 0; tg < num_threadgroups; tg++) {
            for (int b = 0; b < 4; b++) {
                totals[b] += partial_counts[tg * 4 + b];
            }
        }
        for (int b = 0; b < 4; b++) final_counts[b] = totals[b];
    }
}
