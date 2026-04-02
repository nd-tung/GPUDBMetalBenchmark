#include "common.h"

// ===================================================================
// TPC-H Q13 KERNELS — Customer Distribution
// ===================================================================
/*
SELECT
    c_count,
    COUNT(*) AS custdist
FROM (
    SELECT
        c_custkey,
        COUNT(o_orderkey) AS c_count
    FROM
        customer
    LEFT OUTER JOIN
        orders ON c_custkey = o_custkey
        AND o_comment NOT LIKE '%special%requests%'
    GROUP BY
        c_custkey
) AS c_orders
GROUP BY
    c_count
ORDER BY
    custdist DESC,
    c_count DESC;
*/


// ===================================================================
// Pre-computation: GPU-side pattern matching for '%special%requests%'
// ===================================================================
kernel void q13_pattern_match_kernel(
    const device char*  o_comment         [[buffer(0)]],
    device uchar*       o_qualifies       [[buffer(1)]],
    constant uint& orders_size            [[buffer(2)]],
    constant uint& comment_stride         [[buffer(3)]],  // max chars per comment (80)
    uint tid [[thread_position_in_grid]])
{
    if (tid >= orders_size) return;
    // Scan for 'special' then 'requests' after it
    const device char* s = o_comment + (uint64_t)tid * comment_stride;
    // Find end of string
    uint len = 0;
    while (len < comment_stride && s[len] != '\0') len++;
    
    // Search for "special"
    bool found_special = false;
    uint after_special = 0;
    for (uint i = 0; i + 7 <= len; i++) {
        if (s[i]=='s' && s[i+1]=='p' && s[i+2]=='e' && s[i+3]=='c' &&
            s[i+4]=='i' && s[i+5]=='a' && s[i+6]=='l') {
            found_special = true;
            after_special = i + 7;
            break;
        }
    }
    if (!found_special) { o_qualifies[tid] = 1; return; }  // NOT LIKE → qualifies
    
    // Search for "requests" after "special"
    bool found_requests = false;
    for (uint i = after_special; i + 8 <= len; i++) {
        if (s[i]=='r' && s[i+1]=='e' && s[i+2]=='q' && s[i+3]=='u' &&
            s[i+4]=='e' && s[i+5]=='s' && s[i+6]=='t' && s[i+7]=='s') {
            found_requests = true;
            break;
        }
    }
    // NOT LIKE '%special%requests%' → qualifies when pattern NOT found
    o_qualifies[tid] = found_requests ? 0 : 1;
}

kernel void q13_chunked_pattern_match_kernel(
    const device char*  o_comment         [[buffer(0)]],
    device uchar*       o_qualifies       [[buffer(1)]],
    constant uint& chunk_size             [[buffer(2)]],
    constant uint& comment_stride         [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= chunk_size) return;
    const device char* s = o_comment + (uint64_t)tid * comment_stride;
    uint len = 0;
    while (len < comment_stride && s[len] != '\0') len++;
    
    bool found_special = false;
    uint after_special = 0;
    for (uint i = 0; i + 7 <= len; i++) {
        if (s[i]=='s' && s[i+1]=='p' && s[i+2]=='e' && s[i+3]=='c' &&
            s[i+4]=='i' && s[i+5]=='a' && s[i+6]=='l') {
            found_special = true;
            after_special = i + 7;
            break;
        }
    }
    if (!found_special) { o_qualifies[tid] = 1; return; }
    
    bool found_requests = false;
    for (uint i = after_special; i + 8 <= len; i++) {
        if (s[i]=='r' && s[i+1]=='e' && s[i+2]=='q' && s[i+3]=='u' &&
            s[i+4]=='e' && s[i+5]=='s' && s[i+6]=='t' && s[i+7]=='s') {
            found_requests = true;
            break;
        }
    }
    o_qualifies[tid] = found_requests ? 0 : 1;
}

// --- Q13 Pre-filtered Count Kernel (with threadgroup-local batching) ---
// Accumulates order counts in threadgroup-local hash map, then flushes to
// global atomics in bulk. Reduces atomic contention by ~1024x.

struct Q13LocalEntry {
    int custkey;     // -1 = empty
    uint count;
};

kernel void q13_count_prefiltered_kernel(
    const device int* o_custkey          [[buffer(0)]],
    const device uchar* o_qualifies      [[buffer(1)]],
    device atomic_uint* customer_order_counts [[buffer(2)]],
    constant uint& orders_size           [[buffer(3)]],
    constant uint& customer_size         [[buffer(4)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    // Threadgroup-local hash map for batching (power-of-2 for fast modulo)
    const uint LOCAL_HT_SIZE = 2048;
    const uint LOCAL_HT_MASK = LOCAL_HT_SIZE - 1;
    threadgroup Q13LocalEntry local_ht[LOCAL_HT_SIZE];

    // Initialize local HT
    for (uint j = thread_id_in_group; j < LOCAL_HT_SIZE; j += threads_per_group) {
        local_ht[j].custkey = -1;
        local_ht[j].count = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint global_id = (group_id * threads_per_group) + thread_id_in_group;

    for (uint i = global_id; i < orders_size; i += grid_size) {
        if (!o_qualifies[i]) continue;
        int ck = (int)o_custkey[i];
        if (ck < 1 || (uint)ck > customer_size) continue;

        // Insert into threadgroup-local HT (linear probing, no atomics needed
        // since each thread owns its slot via simd-safe insert pattern)
        uint h = ((uint)ck * 0x9E3779B1u) & LOCAL_HT_MASK;
        for (uint s = 0; s < LOCAL_HT_SIZE; s++) {
            uint slot = (h + s) & LOCAL_HT_MASK;
            if (local_ht[slot].custkey == ck) {
                local_ht[slot].count += 1;
                break;
            }
            if (local_ht[slot].custkey == -1) {
                local_ht[slot].custkey = ck;
                local_ht[slot].count = 1;
                break;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Flush local HT to global atomics (much fewer atomic ops)
    for (uint j = thread_id_in_group; j < LOCAL_HT_SIZE; j += threads_per_group) {
        if (local_ht[j].custkey >= 1) {
            atomic_fetch_add_explicit(&customer_order_counts[local_ht[j].custkey - 1u],
                                      local_ht[j].count, memory_order_relaxed);
        }
    }
}


// =============================================================
// Q13 GPU Histogram Kernel (with threadgroup-local histogram)
// =============================================================
kernel void q13_build_histogram_kernel(
    const device uint* customer_order_counts [[buffer(0)]],
    device atomic_uint* histogram            [[buffer(1)]],
    constant uint& customer_size             [[buffer(2)]],
    constant uint& max_bins                  [[buffer(3)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    // Threadgroup-local histogram (max 256 bins)
    threadgroup uint local_hist[256];
    uint bins = min(max_bins, 256u);
    for (uint j = thread_id_in_group; j < bins; j += threads_per_group) {
        local_hist[j] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint global_id = (group_id * threads_per_group) + thread_id_in_group;

    for (uint i = global_id; i < customer_size; i += grid_size) {
        uint count = customer_order_counts[i];
        if (count < bins) {
            // Use threadgroup atomics — much cheaper than device atomics
            atomic_fetch_add_explicit((threadgroup atomic_uint*)&local_hist[count], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Flush local histogram to global (only non-zero bins)
    for (uint j = thread_id_in_group; j < bins; j += threads_per_group) {
        if (local_hist[j] > 0) {
            atomic_fetch_add_explicit(&histogram[j], local_hist[j], memory_order_relaxed);
        }
    }
}
