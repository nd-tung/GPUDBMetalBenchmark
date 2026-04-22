#include "common.h"

// ===================================================================
// TPC-H Q19 KERNELS — Discounted Revenue
// ===================================================================
// Fused GPU path:
// 1. Build part group map.
// 2. Scan lineitem with inline shipmode/instruction filtering.
// 3. Accumulate directly into one atomic revenue buffer.

kernel void q19_filter_and_sum_stage1(
    const device int*   l_partkey         [[buffer(0)]],
    const device float* l_quantity        [[buffer(1)]],
    const device float* l_extendedprice   [[buffer(2)]],
    const device float* l_discount        [[buffer(3)]],
    const device char*  l_shipmode        [[buffer(4)]],
    const device char*  l_shipinstruct    [[buffer(5)]],
    const device uchar* part_group_map    [[buffer(6)]],
    device atomic_float& total_revenue    [[buffer(7)]],
    constant uint& data_size              [[buffer(8)]],
    constant uint& map_size               [[buffer(9)]],
    constant uint& shipmode_stride        [[buffer(10)]],
    constant uint& shipinstruct_stride    [[buffer(11)]],
    uint group_id           [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group  [[threads_per_threadgroup]],
    uint grid_size          [[threads_per_grid]])
{
    float local_revenue = 0.0f;

    for (uint i = (group_id * threads_per_group) + thread_id_in_group;
         i < data_size; i += grid_size) {
        const device char* sm = l_shipmode + (uint64_t)i * shipmode_stride;
        const device char* si = l_shipinstruct + (uint64_t)i * shipinstruct_stride;

        bool instruct_ok = (si[0]=='D' && si[1]=='E' && si[2]=='L');
        bool is_air     = (sm[0]=='A' && sm[1]=='I' && sm[2]=='R' && (sm[3]=='\0' || sm[3]==' '));
        bool is_reg_air = (sm[0]=='R' && sm[1]=='E' && sm[2]=='G' && sm[3]==' ' && sm[4]=='A');
        if (!(instruct_ok && (is_air || is_reg_air))) continue;

        int pk = l_partkey[i];
        if (pk < 0 || (uint)pk >= map_size) continue;
        uchar grp = part_group_map[pk];
        if (grp > 2) continue;

        float qty = l_quantity[i];
        bool match = false;
        if (grp == 0)      match = (qty >= 1.0f && qty <= 11.0f);
        else if (grp == 1) match = (qty >= 10.0f && qty <= 20.0f);
        else               match = (qty >= 20.0f && qty <= 30.0f);

        if (match) {
            local_revenue += l_extendedprice[i] * (1.0f - l_discount[i]);
        }
    }

    threadgroup float shared[32];
    float total = tg_reduce_float(local_revenue, thread_id_in_group, threads_per_group, shared);
    if (thread_id_in_group == 0 && total != 0.0f) {
        atomic_fetch_add_explicit(&total_revenue, total, memory_order_relaxed);
    }
}

// ===================================================================
// Pre-computation: Build part group map on GPU
// ===================================================================
kernel void q19_build_part_group_map_kernel(
    const device int*   p_partkey         [[buffer(0)]],
    const device char*  p_brand           [[buffer(1)]],
    const device char*  p_container       [[buffer(2)]],
    const device int*   p_size            [[buffer(3)]],
    device uchar*       part_group_map    [[buffer(4)]],
    constant uint& data_size              [[buffer(5)]],
    constant uint& brand_stride           [[buffer(6)]],
    constant uint& container_stride       [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= data_size) return;
    int pk = p_partkey[tid];
    int sz = p_size[tid];
    const device char* brand = p_brand + (uint64_t)tid * brand_stride;
    const device char* cont  = p_container + (uint64_t)tid * container_stride;

    uchar grp = 0xFF;

    if (brand[6]=='1' && brand[7]=='2' && sz >= 1 && sz <= 5 &&
        cont[0]=='S' && cont[1]=='M' && cont[2]==' ') {
        char c3 = cont[3]; char c4 = cont[4]; char c5 = cont[5];
        if ((c3=='C' && c4=='A' && c5=='S') || (c3=='B' && c4=='O') ||
            (c3=='P' && c4=='A' && c5=='C') || (c3=='P' && c4=='K')) grp = 0;
    }
    else if (brand[6]=='2' && brand[7]=='3' && sz >= 1 && sz <= 10 &&
             cont[0]=='M' && cont[1]=='E' && cont[2]=='D' && cont[3]==' ') {
        char c4 = cont[4]; char c5 = cont[5]; char c6 = cont[6];
        if ((c4=='B' && c5=='A' && c6=='G') || (c4=='B' && c5=='O') ||
            (c4=='P' && c5=='K') || (c4=='P' && c5=='A' && c6=='C')) grp = 1;
    }
    else if (brand[6]=='3' && brand[7]=='4' && sz >= 1 && sz <= 15 &&
             cont[0]=='L' && cont[1]=='G' && cont[2]==' ') {
        char c3 = cont[3]; char c4 = cont[4]; char c5 = cont[5];
        if ((c3=='C' && c4=='A' && c5=='S') || (c3=='B' && c4=='O') ||
            (c3=='P' && c4=='A' && c5=='C') || (c3=='P' && c4=='K')) grp = 2;
    }

    part_group_map[pk] = grp;
}

// ===================================================================
// Chunked variant
// ===================================================================
kernel void q19_chunked_stage1(
    const device int*   l_partkey         [[buffer(0)]],
    const device float* l_quantity        [[buffer(1)]],
    const device float* l_extendedprice   [[buffer(2)]],
    const device float* l_discount        [[buffer(3)]],
    const device char*  l_shipmode        [[buffer(4)]],
    const device char*  l_shipinstruct    [[buffer(5)]],
    const device uchar* part_group_map    [[buffer(6)]],
    device atomic_float& total_revenue    [[buffer(7)]],
    constant uint& chunk_size             [[buffer(8)]],
    constant uint& map_size               [[buffer(9)]],
    constant uint& shipmode_stride        [[buffer(10)]],
    constant uint& shipinstruct_stride    [[buffer(11)]],
    uint group_id           [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group  [[threads_per_threadgroup]],
    uint grid_size          [[threads_per_grid]])
{
    float local_revenue = 0.0f;

    for (uint i = (group_id * threads_per_group) + thread_id_in_group;
         i < chunk_size; i += grid_size) {
        const device char* sm = l_shipmode + (uint64_t)i * shipmode_stride;
        const device char* si = l_shipinstruct + (uint64_t)i * shipinstruct_stride;

        bool instruct_ok = (si[0]=='D' && si[1]=='E' && si[2]=='L');
        bool is_air     = (sm[0]=='A' && sm[1]=='I' && sm[2]=='R' && (sm[3]=='\0' || sm[3]==' '));
        bool is_reg_air = (sm[0]=='R' && sm[1]=='E' && sm[2]=='G' && sm[3]==' ' && sm[4]=='A');
        if (!(instruct_ok && (is_air || is_reg_air))) continue;

        int pk = l_partkey[i];
        if (pk < 0 || (uint)pk >= map_size) continue;
        uchar grp = part_group_map[pk];
        if (grp > 2) continue;

        float qty = l_quantity[i];
        bool match = false;
        if (grp == 0)      match = (qty >= 1.0f && qty <= 11.0f);
        else if (grp == 1) match = (qty >= 10.0f && qty <= 20.0f);
        else               match = (qty >= 20.0f && qty <= 30.0f);

        if (match) {
            local_revenue += l_extendedprice[i] * (1.0f - l_discount[i]);
        }
    }

    threadgroup float shared[32];
    float total = tg_reduce_float(local_revenue, thread_id_in_group, threads_per_group, shared);
    if (thread_id_in_group == 0 && total != 0.0f) {
        atomic_fetch_add_explicit(&total_revenue, total, memory_order_relaxed);
    }
}