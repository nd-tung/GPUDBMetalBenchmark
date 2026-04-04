#include "common.h"

// ===================================================================
// TPC-H Q19 KERNELS — Discounted Revenue
// ===================================================================
// SELECT SUM(l_extendedprice * (1 - l_discount)) AS revenue
// FROM lineitem, part
// WHERE p_partkey = l_partkey
//   AND (   (p_brand='Brand#12' AND p_container IN SM* AND l_quantity>=1 AND l_quantity<=11
//            AND p_size BETWEEN 1 AND 5 AND l_shipmode IN ('AIR','AIR REG') AND l_shipinstruct='DELIVER IN PERSON')
//        OR (p_brand='Brand#23' AND p_container IN MED* AND l_quantity>=10 AND l_quantity<=20
//            AND p_size BETWEEN 1 AND 10 AND l_shipmode IN ('AIR','AIR REG') AND l_shipinstruct='DELIVER IN PERSON')
//        OR (p_brand='Brand#34' AND p_container IN LG* AND l_quantity>=20 AND l_quantity<=30
//            AND p_size BETWEEN 1 AND 15 AND l_shipmode IN ('AIR','AIR REG') AND l_shipinstruct='DELIVER IN PERSON')
//       )
//
// GPU approach: Part map built on CPU — partkey → group_id (0/1/2 or 0xFF=no match).
// Group encodes the brand+container+size qualification.
// Scan lineitem: pre-filter shipmode(first char 'A') + shipinstruct(first char 'D'),
// then probe part map and check quantity range for the matched group.

kernel void q19_filter_and_sum_stage1(
    const device int*   l_partkey         [[buffer(0)]],
    const device float* l_quantity        [[buffer(1)]],
    const device float* l_extendedprice   [[buffer(2)]],
    const device float* l_discount        [[buffer(3)]],
    const device uchar* l_qualifies       [[buffer(4)]],  // pre-computed: AIR/REG AIR + DELIVER IN PERSON
    const device uchar* part_group_map    [[buffer(5)]],
    device float* partial_revenue         [[buffer(6)]],
    constant uint& data_size              [[buffer(7)]],
    constant uint& map_size               [[buffer(8)]],
    uint group_id           [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group  [[threads_per_threadgroup]],
    uint grid_size          [[threads_per_grid]])
{
    float local_revenue = 0.0f;

    for (uint i = (group_id * threads_per_group) + thread_id_in_group;
         i < data_size; i += grid_size) {
        // Pre-filter: CPU-computed flag for shipmode IN (AIR, REG AIR) AND shipinstruct = DELIVER IN PERSON
        if (!l_qualifies[i]) continue;

        int pk = l_partkey[i];
        if (pk < 0 || (uint)pk >= map_size) continue;
        uchar grp = part_group_map[pk];
        if (grp > 2) continue;  // 0xFF = no match

        float qty = l_quantity[i];
        bool match = false;
        if (grp == 0)      match = (qty >= 1.0f && qty <= 11.0f);   // Brand#12, SM, size 1-5
        else if (grp == 1) match = (qty >= 10.0f && qty <= 20.0f);  // Brand#23, MED, size 1-10
        else               match = (qty >= 20.0f && qty <= 30.0f);  // Brand#34, LG, size 1-15

        if (match) {
            local_revenue += l_extendedprice[i] * (1.0f - l_discount[i]);
        }
    }

    threadgroup float shared[32];
    float total = tg_reduce_float(local_revenue, thread_id_in_group, threads_per_group, shared);
    if (thread_id_in_group == 0) {
        partial_revenue[group_id] = total;
    }
}

kernel void q19_final_sum_stage2(
    const device float* partial_revenue   [[buffer(0)]],
    device float* final_revenue           [[buffer(1)]],
    constant uint& num_threadgroups       [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index == 0) {
        float total = 0.0f;
        for (uint i = 0; i < num_threadgroups; ++i) {
            total += partial_revenue[i];
        }
        final_revenue[0] = total;
    }
}

// ===================================================================
// Pre-computation: Build part group map on GPU
// Classifies each part into group 0/1/2 or 0xFF based on brand+container+size
// ===================================================================
kernel void q19_build_part_group_map_kernel(
    const device int*   p_partkey         [[buffer(0)]],
    const device char*  p_brand           [[buffer(1)]],  // fixed-width
    const device char*  p_container       [[buffer(2)]],  // fixed-width
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

    // Brand#12, SM CASE/SM BOX/SM PACK/SM PKG, size 1-5 → group 0
    // Brand#23, MED BAG/MED BOX/MED PKG/MED PACK, size 1-10 → group 1
    // Brand#34, LG CASE/LG BOX/LG PACK/LG PKG, size 1-15 → group 2

    // Helper: check container suffix (after prefix+space) for CASE/BOX/PACK/PKG
    // SM containers start suffix at index 3, LG at index 3, MED at index 4
    uchar grp = 0xFF;

    // Check Brand#12 + SM containers
    if (brand[6]=='1' && brand[7]=='2' && sz >= 1 && sz <= 5 &&
        cont[0]=='S' && cont[1]=='M' && cont[2]==' ') {
        // SM CASE(C,A,S), SM BOX(B,O,X), SM PACK(P,A,C), SM PKG(P,K,G)
        char c3 = cont[3]; char c4 = cont[4]; char c5 = cont[5];
        if ((c3=='C' && c4=='A' && c5=='S') || (c3=='B' && c4=='O') ||
            (c3=='P' && c4=='A' && c5=='C') || (c3=='P' && c4=='K')) grp = 0;
    }
    // Check Brand#23 + MED containers
    else if (brand[6]=='2' && brand[7]=='3' && sz >= 1 && sz <= 10 &&
             cont[0]=='M' && cont[1]=='E' && cont[2]=='D' && cont[3]==' ') {
        // MED BAG(B,A,G), MED BOX(B,O,X), MED PKG(P,K,G), MED PACK(P,A,C)
        char c4 = cont[4]; char c5 = cont[5]; char c6 = cont[6];
        if ((c4=='B' && c5=='A' && c6=='G') || (c4=='B' && c5=='O') ||
            (c4=='P' && c5=='K') || (c4=='P' && c5=='A' && c6=='C')) grp = 1;
    }
    // Check Brand#34 + LG containers
    else if (brand[6]=='3' && brand[7]=='4' && sz >= 1 && sz <= 15 &&
             cont[0]=='L' && cont[1]=='G' && cont[2]==' ') {
        // LG CASE(C,A,S), LG BOX(B,O,X), LG PACK(P,A,C), LG PKG(P,K,G)
        char c3 = cont[3]; char c4 = cont[4]; char c5 = cont[5];
        if ((c3=='C' && c4=='A' && c5=='S') || (c3=='B' && c4=='O') ||
            (c3=='P' && c4=='A' && c5=='C') || (c3=='P' && c4=='K')) grp = 2;
    }
    part_group_map[pk] = grp;
}

// Pre-computation: Compute lineitem qualifies flag on GPU
// AIR or REG AIR + DELIVER IN PERSON
kernel void q19_shipmode_filter_kernel(
    const device char*  l_shipmode        [[buffer(0)]],
    const device char*  l_shipinstruct    [[buffer(1)]],
    device uchar*       l_qualifies       [[buffer(2)]],
    constant uint& data_size              [[buffer(3)]],
    constant uint& shipmode_stride        [[buffer(4)]],
    constant uint& shipinstruct_stride    [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= data_size) return;
    const device char* sm = l_shipmode + (uint64_t)tid * shipmode_stride;
    const device char* si = l_shipinstruct + (uint64_t)tid * shipinstruct_stride;

    // Check shipinstruct starts with 'DELIVER IN PERSON'
    bool instruct_ok = (si[0]=='D' && si[1]=='E' && si[2]=='L');
    // Check shipmode is 'AIR' (exact: 'A','I','R' then NUL/space) or 'REG AIR'
    bool is_air     = (sm[0]=='A' && sm[1]=='I' && sm[2]=='R' && (sm[3]=='\0' || sm[3]==' '));
    bool is_reg_air = (sm[0]=='R' && sm[1]=='E' && sm[2]=='G' && sm[3]==' ' && sm[4]=='A');
    l_qualifies[tid] = (instruct_ok && (is_air || is_reg_air)) ? 1 : 0;
}

kernel void q19_chunked_shipmode_filter_kernel(
    const device char*  l_shipmode        [[buffer(0)]],
    const device char*  l_shipinstruct    [[buffer(1)]],
    device uchar*       l_qualifies       [[buffer(2)]],
    constant uint& chunk_size             [[buffer(3)]],
    constant uint& shipmode_stride        [[buffer(4)]],
    constant uint& shipinstruct_stride    [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= chunk_size) return;
    const device char* sm = l_shipmode + (uint64_t)tid * shipmode_stride;
    const device char* si = l_shipinstruct + (uint64_t)tid * shipinstruct_stride;
    bool instruct_ok = (si[0]=='D' && si[1]=='E' && si[2]=='L');
    bool is_air     = (sm[0]=='A' && sm[1]=='I' && sm[2]=='R' && (sm[3]=='\0' || sm[3]==' '));
    bool is_reg_air = (sm[0]=='R' && sm[1]=='E' && sm[2]=='G' && sm[3]==' ' && sm[4]=='A');
    l_qualifies[tid] = (instruct_ok && (is_air || is_reg_air)) ? 1 : 0;
}

// --- Chunked variants ---
kernel void q19_chunked_stage1(
    const device int*   l_partkey         [[buffer(0)]],
    const device float* l_quantity        [[buffer(1)]],
    const device float* l_extendedprice   [[buffer(2)]],
    const device float* l_discount        [[buffer(3)]],
    const device uchar* l_qualifies       [[buffer(4)]],
    const device uchar* part_group_map    [[buffer(5)]],
    device float* partial_revenue         [[buffer(6)]],
    constant uint& chunk_size             [[buffer(7)]],
    constant uint& map_size               [[buffer(8)]],
    uint group_id           [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group  [[threads_per_threadgroup]],
    uint grid_size          [[threads_per_grid]])
{
    float local_revenue = 0.0f;

    for (uint i = (group_id * threads_per_group) + thread_id_in_group;
         i < chunk_size; i += grid_size) {
        if (!l_qualifies[i]) continue;

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
    if (thread_id_in_group == 0) {
        partial_revenue[group_id] = total;
    }
}

kernel void q19_chunked_stage2(
    const device float* partial_revenue   [[buffer(0)]],
    device float* final_revenue           [[buffer(1)]],
    constant uint& num_threadgroups       [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index == 0) {
        float total = 0.0f;
        for (uint i = 0; i < num_threadgroups; ++i) {
            total += partial_revenue[i];
        }
        final_revenue[0] = total;
    }
}
