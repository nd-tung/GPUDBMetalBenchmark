#include "common.h"

// ===================================================================
// TPC-H Q16 KERNELS — Parts/Supplier Relationship
// ===================================================================
// Count distinct suppliers per (brand, type, size) group,
// excluding complained suppliers. CPU builds group_id map + complaints bitmap.
// GPU scans partsupp and directly sets bits in per-group suppkey bitmaps.

// Fused scan + distinct bitmap set: for each qualifying partsupp row,
// set bit for suppkey in the bitmap of its group.
kernel void q16_scan_and_bitmap_kernel(
    const device int* ps_partkey        [[buffer(0)]],
    const device int* ps_suppkey        [[buffer(1)]],
    const device int* part_group_map    [[buffer(2)]],  // partkey → group_id (-1 = not qualified)
    const device uint* complaint_bitmap [[buffer(3)]],  // suppkey bitmap (1 = complained, skip)
    device atomic_uint* group_bitmaps   [[buffer(4)]],  // flat array: group_count × bv_ints
    constant uint& partsupp_size        [[buffer(5)]],
    constant uint& part_map_size        [[buffer(6)]],
    constant uint& bv_ints              [[buffer(7)]],  // words per group bitmap
    uint tid [[thread_position_in_grid]])
{
    if (tid >= partsupp_size) return;

    int pk = ps_partkey[tid];
    if ((uint)pk >= part_map_size) return;
    int gid = part_group_map[pk];
    if (gid < 0) return;

    int sk = ps_suppkey[tid];
    if (bitmap_test(complaint_bitmap, sk)) return;

    // Set bit in this group's bitmap
    uint word_idx = (uint)gid * bv_ints + ((uint)sk >> 5);
    uint bit = 1u << ((uint)sk & 31u);
    atomic_fetch_or_explicit(&group_bitmaps[word_idx], bit, memory_order_relaxed);
}

// Count set bits (popcount) in each group's bitmap → group_counts
kernel void q16_popcount_kernel(
    const device uint* group_bitmaps    [[buffer(0)]],  // flat: group_count × bv_ints
    device uint* group_counts           [[buffer(1)]],  // one count per group
    constant uint& num_groups           [[buffer(2)]],
    constant uint& bv_ints              [[buffer(3)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid_in_group [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint gid = group_id;
    if (gid >= num_groups) return;

    const device uint* bv = group_bitmaps + (uint64_t)gid * bv_ints;

    // Grid-stride within this threadgroup over bv_ints words
    uint local_count = 0;
    for (uint w = tid_in_group; w < bv_ints; w += tg_size) {
        local_count += popcount(bv[w]);
    }

    // Threadgroup reduction
    threadgroup uint shared[32];
    uint r = tg_reduce_uint(local_count, tid_in_group, tg_size, shared);
    if (tid_in_group == 0) {
        group_counts[gid] = r;
    }
}
