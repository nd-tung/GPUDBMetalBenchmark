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
