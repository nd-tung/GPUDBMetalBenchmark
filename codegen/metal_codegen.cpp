#include "metal_codegen.h"
#include <sstream>
#include <stdexcept>

namespace codegen {

namespace {

// ===================================================================
// COMMON HEADER — embedded version of kernels/common.h
// ===================================================================

const char* METAL_COMMON_HEADER = R"METAL(
#include <metal_stdlib>
using namespace metal;

// --- SIMD reduction for long (int64) via 2×uint shuffle ---
inline long simd_reduce_add_long(long v) {
    for (uint d = 16; d >= 1; d >>= 1) {
        uint lo = simd_shuffle_down((uint)(v), d);
        uint hi = simd_shuffle_down((uint)((ulong)v >> 32), d);
        v += (long)(((ulong)hi << 32) | (ulong)lo);
    }
    return v;
}

inline float tg_reduce_float(float val, uint tid, uint tg_size,
                             threadgroup float* shared) {
    float sv = simd_sum(val);
    uint lane = tid & 31u;
    uint gid  = tid >> 5u;
    uint ng   = (tg_size + 31u) >> 5u;
    if (lane == 0u) shared[gid] = sv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = 0.0f;
    if (gid == 0u) {
        float v2 = (lane < ng) ? shared[lane] : 0.0f;
        r = simd_sum(v2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

inline uint tg_reduce_uint(uint val, uint tid, uint tg_size,
                           threadgroup uint* shared) {
    uint sv = simd_sum(val);
    uint lane = tid & 31u;
    uint gid  = tid >> 5u;
    uint ng   = (tg_size + 31u) >> 5u;
    if (lane == 0u) shared[gid] = sv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint r = 0u;
    if (gid == 0u) {
        uint v2 = (lane < ng) ? shared[lane] : 0u;
        r = simd_sum(v2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

inline long tg_reduce_long(long val, uint tid, uint tg_size,
                           threadgroup long* shared) {
    long sv = simd_reduce_add_long(val);
    uint lane = tid & 31u;
    uint gid  = tid >> 5u;
    uint ng   = (tg_size + 31u) >> 5u;
    if (lane == 0u) shared[gid] = sv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    long r = 0;
    if (gid == 0u) {
        long v2 = (lane < ng) ? shared[lane] : 0;
        r = simd_reduce_add_long(v2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

inline bool bitmap_test(const device uint* bitmap, int key) {
    return (bitmap[(uint)key >> 5] >> ((uint)key & 31u)) & 1u;
}

inline void bitmap_set(device atomic_uint* bitmap, int key) {
    atomic_fetch_or_explicit(&bitmap[(uint)key >> 5], 1u << ((uint)key & 31u), memory_order_relaxed);
}

inline void atomic_add_long_pair(device atomic_uint* lo, device atomic_uint* hi, long val) {
    ulong uval = as_type<ulong>(val);
    uint add_lo = (uint)(uval);
    uint add_hi = (uint)(uval >> 32);
    uint old_lo = atomic_fetch_add_explicit(lo, add_lo, memory_order_relaxed);
    uint new_lo = old_lo + add_lo;
    uint carry = (new_lo < old_lo) ? 1u : 0u;
    if (add_hi != 0 || carry != 0)
        atomic_fetch_add_explicit(hi, add_hi + carry, memory_order_relaxed);
}

inline long load_long_pair(const device uint* lo, const device uint* hi) {
    ulong v = ((ulong)(*hi) << 32) | (ulong)(*lo);
    return as_type<long>(v);
}

inline uint next_pow2(uint v) {
    v--; v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}

inline uint sf100_hash_key(int key) {
    uint h = as_type<uint>(key);
    h ^= h >> 16; h *= 0x45d9f3b; h ^= h >> 16;
    return h;
}
)METAL";

// ===================================================================
// Q1 KERNEL GENERATOR
// ===================================================================

void generateQ1Kernels(std::ostringstream& out, GeneratedKernels& result) {
    out << R"METAL(
// Q1 bin index helpers
inline int q1_rf_index(char rf) {
    return (rf == 'A') ? 0 : (rf == 'N') ? 1 : (rf == 'R') ? 2 : -1;
}
inline int q1_ls_index(char ls) {
    return (ls == 'F') ? 0 : (ls == 'O') ? 1 : -1;
}

kernel void gen_q1_fused(
    const device int*   l_shipdate        [[buffer(0)]],
    const device char*  l_returnflag      [[buffer(1)]],
    const device char*  l_linestatus      [[buffer(2)]],
    const device float* l_quantity        [[buffer(3)]],
    const device float* l_extendedprice   [[buffer(4)]],
    const device float* l_discount        [[buffer(5)]],
    const device float* l_tax             [[buffer(6)]],
    device atomic_uint* out_qty_lo        [[buffer(7)]],
    device atomic_uint* out_qty_hi        [[buffer(8)]],
    device atomic_uint* out_base_lo       [[buffer(9)]],
    device atomic_uint* out_base_hi       [[buffer(10)]],
    device atomic_uint* out_disc_lo       [[buffer(11)]],
    device atomic_uint* out_disc_hi       [[buffer(12)]],
    device atomic_uint* out_charge_lo     [[buffer(13)]],
    device atomic_uint* out_charge_hi     [[buffer(14)]],
    device atomic_uint* out_discount_bp   [[buffer(15)]],
    device atomic_uint* out_count         [[buffer(16)]],
    constant uint& data_size              [[buffer(17)]],
    constant int&  cutoff_date            [[buffer(18)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    const int BINS = 6;
    long sum_qty_c[BINS], sum_base_c[BINS], sum_disc_c[BINS], sum_charge_c[BINS];
    uint sum_disc_bp[BINS], cnt[BINS];
    for (int b = 0; b < BINS; ++b) {
        sum_qty_c[b]=0; sum_base_c[b]=0; sum_disc_c[b]=0; sum_charge_c[b]=0;
        sum_disc_bp[b]=0u; cnt[b]=0u;
    }
    for (uint i = (group_id * threads_per_group) + thread_id_in_group; i < data_size; i += grid_size) {
        if (l_shipdate[i] > cutoff_date) continue;
        int rfi = q1_rf_index(l_returnflag[i]); if (rfi < 0) continue;
        int lsi = q1_ls_index(l_linestatus[i]); if (lsi < 0) continue;
        int bin = rfi * 2 + lsi;
        float base = l_extendedprice[i];
        float qty  = l_quantity[i];
        float d    = l_discount[i];
        float t    = l_tax[i];
        long base_c = (long)floor(base * 100.0f + 0.5f);
        long qty_c  = (long)floor(qty  * 100.0f + 0.5f);
        int  d_bp   = (int)floor(d * 100.0f + 0.5f);
        int  t_bp   = (int)floor(t * 100.0f + 0.5f);
        long disc_c   = (base_c * (long)(100 - d_bp) + 50) / 100;
        long charge_c = (disc_c * (long)(100 + t_bp) + 50) / 100;
        sum_qty_c[bin] += qty_c; sum_base_c[bin] += base_c;
        sum_disc_c[bin] += disc_c; sum_charge_c[bin] += charge_c;
        sum_disc_bp[bin] += (uint)d_bp; cnt[bin] += 1u;
    }
    threadgroup long tg64[32];
    threadgroup uint tg32[32];
    for (int b = 0; b < BINS; ++b) {
        long r;
        r = tg_reduce_long(sum_qty_c[b], thread_id_in_group, threads_per_group, tg64);
        if (thread_id_in_group == 0) atomic_add_long_pair(&out_qty_lo[b], &out_qty_hi[b], r);
        r = tg_reduce_long(sum_base_c[b], thread_id_in_group, threads_per_group, tg64);
        if (thread_id_in_group == 0) atomic_add_long_pair(&out_base_lo[b], &out_base_hi[b], r);
        r = tg_reduce_long(sum_disc_c[b], thread_id_in_group, threads_per_group, tg64);
        if (thread_id_in_group == 0) atomic_add_long_pair(&out_disc_lo[b], &out_disc_hi[b], r);
        r = tg_reduce_long(sum_charge_c[b], thread_id_in_group, threads_per_group, tg64);
        if (thread_id_in_group == 0) atomic_add_long_pair(&out_charge_lo[b], &out_charge_hi[b], r);
        uint u;
        u = tg_reduce_uint(sum_disc_bp[b], thread_id_in_group, threads_per_group, tg32);
        if (thread_id_in_group == 0) atomic_fetch_add_explicit(&out_discount_bp[b], u, memory_order_relaxed);
        u = tg_reduce_uint(cnt[b], thread_id_in_group, threads_per_group, tg32);
        if (thread_id_in_group == 0) atomic_fetch_add_explicit(&out_count[b], u, memory_order_relaxed);
    }
}
)METAL";
    result.kernels.push_back({"gen_q1_fused", 1024, false});
}

// ===================================================================
// Q6 KERNEL GENERATOR
// ===================================================================

void generateQ6Kernels(std::ostringstream& out, GeneratedKernels& result) {
    out << R"METAL(
kernel void gen_q6_stage1(
    const device int*   l_shipdate        [[buffer(0)]],
    const device float* l_discount        [[buffer(1)]],
    const device float* l_quantity        [[buffer(2)]],
    const device float* l_extendedprice   [[buffer(3)]],
    device float* partial_revenues        [[buffer(4)]],
    constant uint& data_size              [[buffer(5)]],
    constant int&  start_date             [[buffer(6)]],
    constant int&  end_date               [[buffer(7)]],
    constant float& min_discount          [[buffer(8)]],
    constant float& max_discount          [[buffer(9)]],
    constant float& max_quantity          [[buffer(10)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    float local_revenue = 0.0f;
    for (uint i = (group_id * threads_per_group) + thread_id_in_group;
         i < data_size; i += grid_size) {
        if (l_shipdate[i] >= start_date &&
            l_shipdate[i] < end_date &&
            l_discount[i] >= min_discount &&
            l_discount[i] <= max_discount &&
            l_quantity[i] < max_quantity) {
            local_revenue += l_extendedprice[i] * l_discount[i];
        }
    }
    threadgroup float shared[32];
    float total = tg_reduce_float(local_revenue, thread_id_in_group, threads_per_group, shared);
    if (thread_id_in_group == 0) partial_revenues[group_id] = total;
}

kernel void gen_q6_stage2(
    const device float* partial_revenues  [[buffer(0)]],
    device float* final_revenue           [[buffer(1)]],
    constant uint& num_threadgroups       [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index == 0) {
        float total = 0.0f;
        for (uint i = 0; i < num_threadgroups; ++i) total += partial_revenues[i];
        final_revenue[0] = total;
    }
}
)METAL";
    result.kernels.push_back({"gen_q6_stage1", 1024, false});
    result.kernels.push_back({"gen_q6_stage2", 1, true});
}

// ===================================================================
// Q3 KERNEL GENERATOR
// ===================================================================

void generateQ3Kernels(std::ostringstream& out, GeneratedKernels& result) {
    out << R"METAL(
struct GenQ3Agg {
    atomic_int key;
    atomic_float revenue;
    atomic_uint orderdate;
    atomic_uint shippriority;
};

struct GenQ3Result {
    int key;
    float revenue;
    uint orderdate;
    uint shippriority;
};

kernel void gen_q3_build_customer_bitmap(
    const device int* c_custkey           [[buffer(0)]],
    const device char* c_mktsegment      [[buffer(1)]],
    device atomic_uint* customer_bitmap  [[buffer(2)]],
    constant uint& customer_size         [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= customer_size) return;
    if (c_mktsegment[index] == 'B') bitmap_set(customer_bitmap, c_custkey[index]);
}

kernel void gen_q3_build_orders_map(
    const device int* o_orderkey         [[buffer(0)]],
    const device int* o_orderdate        [[buffer(1)]],
    device int* orders_map               [[buffer(2)]],
    constant uint& orders_size           [[buffer(3)]],
    constant int& cutoff_date            [[buffer(4)]],
    const device int* o_custkey          [[buffer(5)]],
    const device uint* customer_bitmap   [[buffer(6)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= orders_size) return;
    if (o_orderdate[index] >= cutoff_date) return;
    int ck = o_custkey[index];
    if (!bitmap_test(customer_bitmap, ck)) return;
    orders_map[o_orderkey[index]] = (int)index;
}

kernel void gen_q3_probe_agg(
    const device int* l_orderkey         [[buffer(0)]],
    const device int* l_shipdate         [[buffer(1)]],
    const device float* l_extendedprice  [[buffer(2)]],
    const device float* l_discount       [[buffer(3)]],
    const device int* orders_map         [[buffer(4)]],
    const device int* o_custkey          [[buffer(5)]],
    const device int* o_orderdate        [[buffer(6)]],
    const device int* o_shippriority     [[buffer(7)]],
    device GenQ3Agg* final_ht            [[buffer(8)]],
    constant uint& lineitem_size         [[buffer(9)]],
    constant int& cutoff_date            [[buffer(10)]],
    constant uint& final_ht_size         [[buffer(11)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    uint global_id = (group_id * threads_per_group) + thread_id_in_group;
    for (uint i = global_id; i < lineitem_size; i += grid_size) {
        if (l_shipdate[i] <= cutoff_date) continue;
        int okey = l_orderkey[i];
        int oidx = orders_map[okey];
        if (oidx == -1) continue;
        float revenue = l_extendedprice[i] * (1.0f - l_discount[i]);
        uint ht_mask = final_ht_size - 1;
        uint hash = (uint)okey & ht_mask;
        for (uint p = 0; p <= ht_mask; p++) {
            uint pidx = (hash + p) & ht_mask;
            int expected = 0;
            if (atomic_compare_exchange_weak_explicit(
                    &final_ht[pidx].key, &expected, okey,
                    memory_order_relaxed, memory_order_relaxed)) {
                atomic_store_explicit(&final_ht[pidx].orderdate,
                                     (uint)o_orderdate[oidx], memory_order_relaxed);
                atomic_store_explicit(&final_ht[pidx].shippriority,
                                     (uint)o_shippriority[oidx], memory_order_relaxed);
            }
            int cur = atomic_load_explicit(&final_ht[pidx].key, memory_order_relaxed);
            if (cur == okey) {
                atomic_fetch_add_explicit(&final_ht[pidx].revenue, revenue, memory_order_relaxed);
                break;
            }
        }
    }
}

kernel void gen_q3_compact(
    device GenQ3Result* ht               [[buffer(0)]],
    device GenQ3Result* output           [[buffer(1)]],
    device atomic_uint& result_count     [[buffer(2)]],
    constant uint& ht_size               [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= ht_size) return;
    if (ht[index].key > 0) {
        uint pos = atomic_fetch_add_explicit(&result_count, 1, memory_order_relaxed);
        output[pos] = ht[index];
    }
}
)METAL";
    result.kernels.push_back({"gen_q3_build_customer_bitmap", 1024, false});
    result.kernels.push_back({"gen_q3_build_orders_map", 1024, false});
    result.kernels.push_back({"gen_q3_probe_agg", 1024, false});
    result.kernels.push_back({"gen_q3_compact", 1024, false});
}

// ===================================================================
// Q14 KERNEL GENERATOR
// ===================================================================

void generateQ14Kernels(std::ostringstream& out, GeneratedKernels& result) {
    out << R"METAL(
kernel void gen_q14_stage1(
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
            float rev = l_extendedprice[i] * (1.0f - l_discount[i]);
            local_total += rev;
            if (bitmap_test(promo_bitmap, l_partkey[i]))
                local_promo += rev;
        }
    }
    threadgroup float shared[32];
    float ps = tg_reduce_float(local_promo, thread_id_in_group, threads_per_group, shared);
    float ts = tg_reduce_float(local_total, thread_id_in_group, threads_per_group, shared);
    if (thread_id_in_group == 0) {
        partial_promo[group_id] = ps;
        partial_total[group_id] = ts;
    }
}

kernel void gen_q14_stage2(
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
)METAL";
    result.kernels.push_back({"gen_q14_stage1", 1024, false});
    result.kernels.push_back({"gen_q14_stage2", 1, true});
}

// ===================================================================
// Q13 KERNEL GENERATOR
// ===================================================================

void generateQ13Kernels(std::ostringstream& out, GeneratedKernels& result) {
    out << R"METAL(
kernel void gen_q13_count_orders(
    const device int*  o_custkey          [[buffer(0)]],
    const device char* o_comment          [[buffer(1)]],
    device atomic_uint* custorder_counts  [[buffer(2)]],
    constant uint& orders_size            [[buffer(3)]],
    constant uint& comment_width          [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= orders_size) return;
    const device char* c = o_comment + (uint64_t)index * comment_width;
    // Check NOT LIKE '%special%requests%'
    bool match = false;
    for (uint i = 0; i + 7 <= comment_width; i++) {
        if (c[i]=='s' && c[i+1]=='p' && c[i+2]=='e' && c[i+3]=='c' &&
            c[i+4]=='i' && c[i+5]=='a' && c[i+6]=='l') {
            for (uint j = i + 7; j + 8 <= comment_width; j++) {
                if (c[j]=='r' && c[j+1]=='e' && c[j+2]=='q' && c[j+3]=='u' &&
                    c[j+4]=='e' && c[j+5]=='s' && c[j+6]=='t' && c[j+7]=='s') {
                    match = true; break;
                }
            }
            if (match) break;
        }
    }
    if (!match) {
        atomic_fetch_add_explicit(&custorder_counts[o_custkey[index]], 1u, memory_order_relaxed);
    }
}

kernel void gen_q13_histogram(
    const device uint* custorder_counts   [[buffer(0)]],
    device atomic_uint* histogram         [[buffer(1)]],
    constant uint& max_custkey            [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index > max_custkey) return;
    uint cnt = custorder_counts[index];
    atomic_fetch_add_explicit(&histogram[cnt], 1u, memory_order_relaxed);
}
)METAL";
    result.kernels.push_back({"gen_q13_count_orders", 1024, false});
    result.kernels.push_back({"gen_q13_histogram", 1024, false});
}

// ===================================================================
// Q4 KERNEL GENERATOR
// ===================================================================

void generateQ4Kernels(std::ostringstream& out, GeneratedKernels& result) {
    out << R"METAL(
// Q4 Phase 1: Build late-delivery bitmap from lineitem
kernel void gen_q4_build_late_bitmap(
    const device int*  l_orderkey         [[buffer(0)]],
    const device int*  l_commitdate       [[buffer(1)]],
    const device int*  l_receiptdate      [[buffer(2)]],
    device atomic_uint* late_bitmap       [[buffer(3)]],
    constant uint& data_size              [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= data_size) return;
    if (l_commitdate[tid] < l_receiptdate[tid]) {
        bitmap_set(late_bitmap, l_orderkey[tid]);
    }
}

// Q4 Phase 2: Scan orders, filter by date + bitmap, count by priority
kernel void gen_q4_count_stage1(
    const device int*  o_orderkey         [[buffer(0)]],
    const device int*  o_orderdate        [[buffer(1)]],
    const device char* o_orderpriority    [[buffer(2)]],
    const device uint* late_bitmap        [[buffer(3)]],
    device uint* partial_counts           [[buffer(4)]],
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
        int bin = o_orderpriority[i] - '1';
        if (bin >= 0 && bin < 5) local_counts[bin] += 1;
    }
    threadgroup uint shared[32];
    for (int b = 0; b < 5; b++) {
        uint r = tg_reduce_uint(local_counts[b], thread_id_in_group, threads_per_group, shared);
        if (thread_id_in_group == 0) partial_counts[group_id * 5 + b] = r;
    }
}

kernel void gen_q4_count_stage2(
    const device uint* partial_counts     [[buffer(0)]],
    device uint* final_counts             [[buffer(1)]],
    constant uint& num_threadgroups       [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index == 0) {
        uint totals[5] = {0, 0, 0, 0, 0};
        for (uint tg = 0; tg < num_threadgroups; tg++)
            for (int b = 0; b < 5; b++)
                totals[b] += partial_counts[tg * 5 + b];
        for (int b = 0; b < 5; b++) final_counts[b] = totals[b];
    }
}
)METAL";
    result.kernels.push_back({"gen_q4_build_late_bitmap", 1024, false});
    result.kernels.push_back({"gen_q4_count_stage1", 1024, false});
    result.kernels.push_back({"gen_q4_count_stage2", 1, true});
}

// ===================================================================
// Q12 KERNEL GENERATOR
// ===================================================================

void generateQ12Kernels(std::ostringstream& out, GeneratedKernels& result) {
    out << R"METAL(
// Q12 Phase 1: Build priority bitmap from orders (orderpriority '1' or '2')
kernel void gen_q12_build_priority_bitmap(
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

// Q12 Phase 2: Filter lineitem and count by {MAIL,SHIP} x {HIGH,LOW}
kernel void gen_q12_filter_stage1(
    const device int*   l_orderkey        [[buffer(0)]],
    const device char*  l_shipmode        [[buffer(1)]],
    const device int*   l_shipdate        [[buffer(2)]],
    const device int*   l_commitdate      [[buffer(3)]],
    const device int*   l_receiptdate     [[buffer(4)]],
    const device uint*  priority_bitmap   [[buffer(5)]],
    device uint* partial_counts           [[buffer(6)]],
    constant uint& data_size              [[buffer(7)]],
    constant int&  receipt_start          [[buffer(8)]],
    constant int&  receipt_end            [[buffer(9)]],
    uint group_id           [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group  [[threads_per_threadgroup]],
    uint grid_size          [[threads_per_grid]])
{
    uint local_counts[4] = {0, 0, 0, 0};
    for (uint i = (group_id * threads_per_group) + thread_id_in_group;
         i < data_size; i += grid_size) {
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
        if (thread_id_in_group == 0) partial_counts[group_id * 4 + b] = r;
    }
}

kernel void gen_q12_filter_stage2(
    const device uint* partial_counts     [[buffer(0)]],
    device uint* final_counts             [[buffer(1)]],
    constant uint& num_threadgroups       [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index == 0) {
        uint totals[4] = {0, 0, 0, 0};
        for (uint tg = 0; tg < num_threadgroups; tg++)
            for (int b = 0; b < 4; b++)
                totals[b] += partial_counts[tg * 4 + b];
        for (int b = 0; b < 4; b++) final_counts[b] = totals[b];
    }
}
)METAL";
    result.kernels.push_back({"gen_q12_build_priority_bitmap", 1024, false});
    result.kernels.push_back({"gen_q12_filter_stage1", 1024, false});
    result.kernels.push_back({"gen_q12_filter_stage2", 1, true});
}

// ===================================================================
// Q19 KERNEL GENERATOR
// ===================================================================

void generateQ19Kernels(std::ostringstream& out, GeneratedKernels& result) {
    out << R"METAL(
// Q19 Phase 1: Build part group map — classify parts into groups 0/1/2 or 0xFF
kernel void gen_q19_build_part_group_map(
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
    // Brand#12 + SM containers + size 1-5 → group 0
    if (brand[6]=='1' && brand[7]=='2' && sz >= 1 && sz <= 5 &&
        cont[0]=='S' && cont[1]=='M' && cont[2]==' ') {
        char c3 = cont[3]; char c4 = cont[4]; char c5 = cont[5];
        if ((c3=='C' && c4=='A' && c5=='S') || (c3=='B' && c4=='O') ||
            (c3=='P' && c4=='A' && c5=='C') || (c3=='P' && c4=='K')) grp = 0;
    }
    // Brand#23 + MED containers + size 1-10 → group 1
    else if (brand[6]=='2' && brand[7]=='3' && sz >= 1 && sz <= 10 &&
             cont[0]=='M' && cont[1]=='E' && cont[2]=='D' && cont[3]==' ') {
        char c4 = cont[4]; char c5 = cont[5]; char c6 = cont[6];
        if ((c4=='B' && c5=='A' && c6=='G') || (c4=='B' && c5=='O') ||
            (c4=='P' && c5=='K') || (c4=='P' && c5=='A' && c6=='C')) grp = 1;
    }
    // Brand#34 + LG containers + size 1-15 → group 2
    else if (brand[6]=='3' && brand[7]=='4' && sz >= 1 && sz <= 15 &&
             cont[0]=='L' && cont[1]=='G' && cont[2]==' ') {
        char c3 = cont[3]; char c4 = cont[4]; char c5 = cont[5];
        if ((c3=='C' && c4=='A' && c5=='S') || (c3=='B' && c4=='O') ||
            (c3=='P' && c4=='A' && c5=='C') || (c3=='P' && c4=='K')) grp = 2;
    }
    part_group_map[pk] = grp;
}

// Q19 Phase 2: Compute lineitem shipmode/shipinstruct qualifies flag
kernel void gen_q19_shipmode_filter(
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
    bool instruct_ok = (si[0]=='D' && si[1]=='E' && si[2]=='L');
    bool is_air     = (sm[0]=='A' && sm[1]=='I' && sm[2]=='R' && (sm[3]=='\0' || sm[3]==' '));
    bool is_reg_air = (sm[0]=='R' && sm[1]=='E' && sm[2]=='G' && sm[3]==' ' && sm[4]=='A');
    l_qualifies[tid] = (instruct_ok && (is_air || is_reg_air)) ? 1 : 0;
}

// Q19 Phase 3: Filter+sum with part group map + quantity range check
kernel void gen_q19_sum_stage1(
    const device int*   l_partkey         [[buffer(0)]],
    const device float* l_quantity        [[buffer(1)]],
    const device float* l_extendedprice   [[buffer(2)]],
    const device float* l_discount        [[buffer(3)]],
    const device uchar* l_qualifies       [[buffer(4)]],
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
        if (match) local_revenue += l_extendedprice[i] * (1.0f - l_discount[i]);
    }
    threadgroup float shared[32];
    float total = tg_reduce_float(local_revenue, thread_id_in_group, threads_per_group, shared);
    if (thread_id_in_group == 0) partial_revenue[group_id] = total;
}

kernel void gen_q19_sum_stage2(
    const device float* partial_revenue   [[buffer(0)]],
    device float* final_revenue           [[buffer(1)]],
    constant uint& num_threadgroups       [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index == 0) {
        float total = 0.0f;
        for (uint i = 0; i < num_threadgroups; ++i) total += partial_revenue[i];
        final_revenue[0] = total;
    }
}
)METAL";
    result.kernels.push_back({"gen_q19_build_part_group_map", 1024, false});
    result.kernels.push_back({"gen_q19_shipmode_filter", 1024, false});
    result.kernels.push_back({"gen_q19_sum_stage1", 1024, false});
    result.kernels.push_back({"gen_q19_sum_stage2", 1, true});
}

// ===================================================================
// Q15 KERNEL GENERATOR
// ===================================================================

void generateQ15Kernels(std::ostringstream& out, GeneratedKernels& result) {
    out << R"METAL(
// Q15: Single kernel — lineitem date filter → atomic_add revenue to map[suppkey]
kernel void gen_q15_aggregate_revenue(
    const device int*    l_suppkey         [[buffer(0)]],
    const device int*    l_shipdate        [[buffer(1)]],
    const device float*  l_extendedprice   [[buffer(2)]],
    const device float*  l_discount        [[buffer(3)]],
    device atomic_float* revenue_map       [[buffer(4)]],
    constant uint& data_size              [[buffer(5)]],
    constant int&  date_start             [[buffer(6)]],
    constant int&  date_end               [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= data_size) return;
    int date = l_shipdate[tid];
    if (date < date_start || date >= date_end) return;
    float revenue = l_extendedprice[tid] * (1.0f - l_discount[tid]);
    int sk = l_suppkey[tid];
    atomic_fetch_add_explicit(&revenue_map[sk], revenue, memory_order_relaxed);
}
)METAL";
    result.kernels.push_back({"gen_q15_aggregate_revenue", 256, true});
}

// ===================================================================
// Q11 KERNEL GENERATOR
// ===================================================================

void generateQ11Kernels(std::ostringstream& out, GeneratedKernels& result) {
    out << R"METAL(
// Q11: Scan partsupp, filter by supplier bitmap (GERMANY), atomic_add value per partkey
// Also produces threadgroup partial sums for global sum computation
kernel void gen_q11_aggregate(
    const device int*    ps_partkey        [[buffer(0)]],
    const device int*    ps_suppkey        [[buffer(1)]],
    const device float*  ps_supplycost     [[buffer(2)]],
    const device int*    ps_availqty       [[buffer(3)]],
    const device uint*   supp_bitmap       [[buffer(4)]],
    device atomic_float* value_map         [[buffer(5)]],
    device float*        partial_sums      [[buffer(6)]],
    constant uint& data_size              [[buffer(7)]],
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
    threadgroup float shared[32];
    float total = tg_reduce_float(local_sum, thread_id_in_group, threads_per_group, shared);
    if (thread_id_in_group == 0) {
        partial_sums[group_id] = total;
    }
}
)METAL";
    result.kernels.push_back({"gen_q11_aggregate", 256, true});
}

// ===================================================================
// Q10 KERNEL GENERATOR
// ===================================================================

void generateQ10Kernels(std::ostringstream& out, GeneratedKernels& result) {
    out << R"METAL(
// Q10 Phase 1: Build orders direct map — orderkey → custkey (date filtered)
kernel void gen_q10_build_orders_map(
    const device int* o_orderkey     [[buffer(0)]],
    const device int* o_custkey      [[buffer(1)]],
    const device int* o_orderdate    [[buffer(2)]],
    device int* orders_map           [[buffer(3)]],
    constant uint& orders_size       [[buffer(4)]],
    constant int& date_start         [[buffer(5)]],
    constant int& date_end           [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= orders_size) return;
    int date = o_orderdate[tid];
    if (date < date_start || date >= date_end) return;
    int okey = o_orderkey[tid];
    orders_map[okey] = o_custkey[tid];
}

// Q10 Phase 2: Probe lineitem, filter returnflag='R', aggregate revenue per custkey
kernel void gen_q10_probe_and_aggregate(
    const device int*    l_orderkey        [[buffer(0)]],
    const device char*   l_returnflag      [[buffer(1)]],
    const device float*  l_extendedprice   [[buffer(2)]],
    const device float*  l_discount        [[buffer(3)]],
    const device int*    orders_map        [[buffer(4)]],
    device atomic_float* cust_revenue      [[buffer(5)]],
    constant uint& lineitem_size          [[buffer(6)]],
    constant uint& map_size               [[buffer(7)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    uint global_id = (group_id * threads_per_group) + thread_id_in_group;
    for (uint i = global_id; i < lineitem_size; i += grid_size) {
        if (l_returnflag[i] != 'R') continue;
        int okey = l_orderkey[i];
        if ((uint)okey >= map_size) continue;
        int ck = orders_map[okey];
        if (ck == -1) continue;
        float revenue = l_extendedprice[i] * (1.0f - l_discount[i]);
        atomic_fetch_add_explicit(&cust_revenue[ck], revenue, memory_order_relaxed);
    }
}
)METAL";
    result.kernels.push_back({"gen_q10_build_orders_map", 256, true});
    result.kernels.push_back({"gen_q10_probe_and_aggregate", 1024, false});
}

// ===================================================================
// Q5 KERNEL GENERATOR
// ===================================================================

void generateQ5Kernels(std::ostringstream& out, GeneratedKernels& result) {
    out << R"METAL(
// Q5 Kernel 1: Build customer->nationkey direct map (ASIA nations only)
kernel void gen_q5_build_customer_nation_map(
    const device int* c_custkey     [[buffer(0)]],
    const device int* c_nationkey   [[buffer(1)]],
    device int* customer_nation_map [[buffer(2)]],
    const device uint* nation_bitmap [[buffer(3)]],
    constant uint& customer_size    [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= customer_size) return;
    int nk = c_nationkey[index];
    if (!bitmap_test(nation_bitmap, nk)) return;
    customer_nation_map[c_custkey[index]] = nk;
}

// Q5 Kernel 2: Build supplier->nationkey direct map (ASIA nations only)
kernel void gen_q5_build_supplier_nation_map(
    const device int* s_suppkey      [[buffer(0)]],
    const device int* s_nationkey    [[buffer(1)]],
    device int* supplier_nation_map  [[buffer(2)]],
    const device uint* nation_bitmap [[buffer(3)]],
    constant uint& supplier_size     [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= supplier_size) return;
    int nk = s_nationkey[index];
    if (!bitmap_test(nation_bitmap, nk)) return;
    supplier_nation_map[s_suppkey[index]] = nk;
}

// Q5 Kernel 3: Build orders->nationkey direct map (date+customer filter)
kernel void gen_q5_build_orders_map(
    const device int* o_orderkey           [[buffer(0)]],
    const device int* o_custkey            [[buffer(1)]],
    const device int* o_orderdate          [[buffer(2)]],
    device int* orders_nation_map          [[buffer(3)]],
    constant uint& orders_size             [[buffer(4)]],
    constant int& date_start               [[buffer(5)]],
    constant int& date_end                 [[buffer(6)]],
    constant uint& map_size                [[buffer(7)]],
    const device int* customer_nation_map  [[buffer(8)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= orders_size) return;
    int date = o_orderdate[index];
    if (date < date_start || date >= date_end) return;
    int ck = o_custkey[index];
    int cust_nk = customer_nation_map[ck];
    if (cust_nk == -1) return;
    int key = o_orderkey[index];
    if ((uint)key < map_size)
        orders_nation_map[key] = cust_nk;
}

// Q5 Kernel 4: Probe lineitem, same-nation check, aggregate revenue per nation
kernel void gen_q5_probe_and_aggregate(
    const device int* l_orderkey        [[buffer(0)]],
    const device int* l_suppkey         [[buffer(1)]],
    const device float* l_extendedprice [[buffer(2)]],
    const device float* l_discount      [[buffer(3)]],
    const device int* orders_nation_map [[buffer(4)]],
    const device int* supplier_nation_map [[buffer(5)]],
    device atomic_float* nation_revenue [[buffer(6)]],
    constant uint& lineitem_size        [[buffer(7)]],
    constant uint& map_size             [[buffer(8)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    uint global_id = (group_id * threads_per_group) + thread_id_in_group;
    const int BATCH_SIZE = 4;
    for (uint i = global_id; i < lineitem_size; i += grid_size * BATCH_SIZE) {
        for (int k = 0; k < BATCH_SIZE; k++) {
            uint idx = i + k * grid_size;
            if (idx >= lineitem_size) break;
            int orderkey = l_orderkey[idx];
            if ((uint)orderkey >= map_size) continue;
            int cust_nationkey = orders_nation_map[orderkey];
            if (cust_nationkey == -1) continue;
            int suppkey = l_suppkey[idx];
            int supp_nationkey = supplier_nation_map[suppkey];
            if (supp_nationkey != cust_nationkey) continue;
            float revenue = l_extendedprice[idx] * (1.0f - l_discount[idx]);
            atomic_fetch_add_explicit(&nation_revenue[cust_nationkey], revenue, memory_order_relaxed);
        }
    }
}
)METAL";
    result.kernels.push_back({"gen_q5_build_customer_nation_map", 256, true});
    result.kernels.push_back({"gen_q5_build_supplier_nation_map", 256, true});
    result.kernels.push_back({"gen_q5_build_orders_map", 256, true});
    result.kernels.push_back({"gen_q5_probe_and_aggregate", 1024, false});
}

// ===================================================================
// Q7 KERNEL GENERATOR
// ===================================================================

void generateQ7Kernels(std::ostringstream& out, GeneratedKernels& result) {
    out << R"METAL(
// Q7 Kernel 1: Build supplier->nationkey map (FRANCE/GERMANY only)
kernel void gen_q7_build_supplier_map(
    const device int* s_suppkey     [[buffer(0)]],
    const device int* s_nationkey   [[buffer(1)]],
    device int* supp_nation_map     [[buffer(2)]],
    constant int& france_nk         [[buffer(3)]],
    constant int& germany_nk        [[buffer(4)]],
    constant uint& supplier_size    [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= supplier_size) return;
    int nk = s_nationkey[tid];
    if (nk == france_nk || nk == germany_nk)
        supp_nation_map[s_suppkey[tid]] = nk;
}

// Q7 Kernel 2: Build customer->nationkey map (FRANCE/GERMANY only)
kernel void gen_q7_build_customer_map(
    const device int* c_custkey     [[buffer(0)]],
    const device int* c_nationkey   [[buffer(1)]],
    device int* cust_nation_map     [[buffer(2)]],
    constant int& france_nk         [[buffer(3)]],
    constant int& germany_nk        [[buffer(4)]],
    constant uint& customer_size    [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= customer_size) return;
    int nk = c_nationkey[tid];
    if (nk == france_nk || nk == germany_nk)
        cust_nation_map[c_custkey[tid]] = nk;
}

// Q7 Kernel 3: Build orders map orderkey->custkey
kernel void gen_q7_build_orders_map(
    const device int* o_orderkey    [[buffer(0)]],
    const device int* o_custkey     [[buffer(1)]],
    device int* orders_map          [[buffer(2)]],
    constant uint& orders_size      [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= orders_size) return;
    orders_map[o_orderkey[tid]] = o_custkey[tid];
}

// Q7 Kernel 4: Probe lineitem, date+chain checks, aggregate 4 bins
kernel void gen_q7_probe_and_aggregate(
    const device int* l_orderkey        [[buffer(0)]],
    const device int* l_suppkey         [[buffer(1)]],
    const device int* l_shipdate        [[buffer(2)]],
    const device float* l_extendedprice [[buffer(3)]],
    const device float* l_discount      [[buffer(4)]],
    const device int* orders_map        [[buffer(5)]],
    const device int* cust_nation_map   [[buffer(6)]],
    const device int* supp_nation_map   [[buffer(7)]],
    device atomic_float* revenue_bins   [[buffer(8)]],
    constant uint& lineitem_size        [[buffer(9)]],
    constant uint& orders_map_size      [[buffer(10)]],
    constant int& france_nk             [[buffer(11)]],
    constant int& germany_nk            [[buffer(12)]],
    constant int& date_start            [[buffer(13)]],
    constant int& date_end              [[buffer(14)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    uint global_id = (group_id * threads_per_group) + thread_id_in_group;
    for (uint i = global_id; i < lineitem_size; i += grid_size) {
        int shipdate = l_shipdate[i];
        if (shipdate < date_start || shipdate > date_end) continue;
        int okey = l_orderkey[i];
        if ((uint)okey >= orders_map_size) continue;
        int ck = orders_map[okey];
        if (ck == -1) continue;
        int supp_nk = supp_nation_map[l_suppkey[i]];
        if (supp_nk == -1) continue;
        int cust_nk = cust_nation_map[ck];
        if (cust_nk == -1) continue;
        int pair_idx;
        if (supp_nk == france_nk && cust_nk == germany_nk) pair_idx = 0;
        else if (supp_nk == germany_nk && cust_nk == france_nk) pair_idx = 1;
        else continue;
        int year = shipdate / 10000;
        int year_idx;
        if (year == 1995) year_idx = 0;
        else if (year == 1996) year_idx = 1;
        else continue;
        float revenue = l_extendedprice[i] * (1.0f - l_discount[i]);
        atomic_fetch_add_explicit(&revenue_bins[pair_idx * 2 + year_idx], revenue, memory_order_relaxed);
    }
}
)METAL";
    result.kernels.push_back({"gen_q7_build_supplier_map", 256, true});
    result.kernels.push_back({"gen_q7_build_customer_map", 256, true});
    result.kernels.push_back({"gen_q7_build_orders_map", 256, true});
    result.kernels.push_back({"gen_q7_probe_and_aggregate", 1024, false});
}

// ===================================================================
// Q8 KERNEL GENERATOR
// ===================================================================

void generateQ8Kernels(std::ostringstream& out, GeneratedKernels& result) {
    out << R"METAL(
// Q8 Kernel 1: Build orders map (date+AMERICA customer filter) → custkey+year
kernel void gen_q8_build_orders_map(
    const device int* o_orderkey        [[buffer(0)]],
    const device int* o_custkey         [[buffer(1)]],
    const device int* o_orderdate       [[buffer(2)]],
    device int* orders_custkey_map      [[buffer(3)]],
    device int* orders_year_map         [[buffer(4)]],
    const device int* cust_nation_map   [[buffer(5)]],
    constant uint& orders_size          [[buffer(6)]],
    constant int& date_start            [[buffer(7)]],
    constant int& date_end              [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= orders_size) return;
    int date = o_orderdate[tid];
    if (date < date_start || date > date_end) return;
    int ck = o_custkey[tid];
    int cust_nk = cust_nation_map[ck];
    if (cust_nk == -1) return;
    int okey = o_orderkey[tid];
    orders_custkey_map[okey] = ck;
    orders_year_map[okey] = date / 10000;
}

// Q8 Kernel 2: Probe lineitem, part bitmap check, aggregate 4 bins
kernel void gen_q8_probe_and_aggregate(
    const device int* l_orderkey        [[buffer(0)]],
    const device int* l_partkey         [[buffer(1)]],
    const device int* l_suppkey         [[buffer(2)]],
    const device float* l_extendedprice [[buffer(3)]],
    const device float* l_discount      [[buffer(4)]],
    const device uint* part_bitmap      [[buffer(5)]],
    const device int* orders_custkey_map [[buffer(6)]],
    const device int* orders_year_map   [[buffer(7)]],
    const device int* supp_nation_map   [[buffer(8)]],
    device atomic_float* result_bins    [[buffer(9)]],
    constant uint& lineitem_size        [[buffer(10)]],
    constant uint& orders_map_size      [[buffer(11)]],
    constant int& brazil_nk             [[buffer(12)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    uint global_id = (group_id * threads_per_group) + thread_id_in_group;
    for (uint i = global_id; i < lineitem_size; i += grid_size) {
        int pk = l_partkey[i];
        if (!bitmap_test(part_bitmap, pk)) continue;
        int okey = l_orderkey[i];
        if ((uint)okey >= orders_map_size) continue;
        int ck = orders_custkey_map[okey];
        if (ck == -1) continue;
        int year = orders_year_map[okey];
        int year_idx = year - 1995;
        if (year_idx < 0 || year_idx > 1) continue;
        float revenue = l_extendedprice[i] * (1.0f - l_discount[i]);
        atomic_fetch_add_explicit(&result_bins[2 + year_idx], revenue, memory_order_relaxed);
        int supp_nk = supp_nation_map[l_suppkey[i]];
        if (supp_nk == brazil_nk) {
            atomic_fetch_add_explicit(&result_bins[year_idx], revenue, memory_order_relaxed);
        }
    }
}
)METAL";
    result.kernels.push_back({"gen_q8_build_orders_map", 256, true});
    result.kernels.push_back({"gen_q8_probe_and_aggregate", 1024, false});
}

// ===================================================================
// Q17: Small-Quantity-Order Revenue
// ===================================================================
static void generateQ17Kernels(std::ostringstream& out, GeneratedKernels& result) {
    out << R"METAL(
// Q17 Pass 1: Aggregate quantity stats per qualifying partkey
kernel void gen_q17_aggregate_qty_stats(
    const device int* l_partkey          [[buffer(0)]],
    const device float* l_quantity       [[buffer(1)]],
    const device uint* part_bitmap       [[buffer(2)]],
    device atomic_float* sum_qty_map     [[buffer(3)]],
    device atomic_uint* count_map        [[buffer(4)]],
    constant uint& lineitem_size         [[buffer(5)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    uint global_id = (group_id * threads_per_group) + thread_id_in_group;
    for (uint i = global_id; i < lineitem_size; i += grid_size) {
        int pk = l_partkey[i];
        if (!bitmap_test(part_bitmap, pk)) continue;
        float qty = l_quantity[i];
        atomic_fetch_add_explicit(&sum_qty_map[pk], qty, memory_order_relaxed);
        atomic_fetch_add_explicit(&count_map[pk], 1u, memory_order_relaxed);
    }
}

// Q17 Pass 2: Sum extendedprice where qty < threshold
kernel void gen_q17_sum_revenue(
    const device int* l_partkey          [[buffer(0)]],
    const device float* l_quantity       [[buffer(1)]],
    const device float* l_extendedprice  [[buffer(2)]],
    const device uint* part_bitmap       [[buffer(3)]],
    const device float* threshold_map    [[buffer(4)]],
    device atomic_float& total_revenue   [[buffer(5)]],
    constant uint& lineitem_size         [[buffer(6)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    uint global_id = (group_id * threads_per_group) + thread_id_in_group;
    for (uint i = global_id; i < lineitem_size; i += grid_size) {
        int pk = l_partkey[i];
        if (!bitmap_test(part_bitmap, pk)) continue;
        float thresh = threshold_map[pk];
        if (thresh <= 0.0f) continue;
        if (l_quantity[i] < thresh) {
            atomic_fetch_add_explicit(&total_revenue, l_extendedprice[i], memory_order_relaxed);
        }
    }
}
)METAL";
    result.kernels.push_back({"gen_q17_aggregate_qty_stats", 1024, false});
    result.kernels.push_back({"gen_q17_sum_revenue", 1024, false});
}

// ===================================================================
// Q22: Global Sales Opportunity
// ===================================================================
static void generateQ22Kernels(std::ostringstream& out, GeneratedKernels& result) {
    out << R"METAL(
// Q22 Phase 1: Sum + count acctbal for qualifying customers with bal > 0
kernel void gen_q22_avg_balance(
    const device int* c_phone_prefix    [[buffer(0)]],
    const device float* c_acctbal       [[buffer(1)]],
    device atomic_float& sum_bal        [[buffer(2)]],
    device atomic_uint& count_bal       [[buffer(3)]],
    constant uint& cust_size            [[buffer(4)]],
    constant uint& valid_prefix_mask    [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= cust_size) return;
    float bal = c_acctbal[tid];
    if (bal <= 0.0f) return;
    int prefix = c_phone_prefix[tid];
    if (prefix < 0 || prefix > 31) return;
    if (!((valid_prefix_mask >> (uint)prefix) & 1u)) return;
    atomic_fetch_add_explicit(&sum_bal, bal, memory_order_relaxed);
    atomic_fetch_add_explicit(&count_bal, 1u, memory_order_relaxed);
}

// Q22 Phase 2: Build orders custkey bitmap
kernel void gen_q22_build_orders_bitmap(
    const device int* o_custkey         [[buffer(0)]],
    device atomic_uint* cust_bitmap     [[buffer(1)]],
    constant uint& orders_size          [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= orders_size) return;
    bitmap_set(cust_bitmap, o_custkey[tid]);
}

// Q22 Phase 3: Final aggregate — count/sum per country code (7 bins)
kernel void gen_q22_final_aggregate(
    const device int* c_phone_prefix    [[buffer(0)]],
    const device float* c_acctbal       [[buffer(1)]],
    const device int* c_custkey         [[buffer(2)]],
    const device uint* cust_bitmap      [[buffer(3)]],
    device atomic_uint* result_count    [[buffer(4)]],
    device atomic_float* result_sum     [[buffer(5)]],
    constant uint& cust_size            [[buffer(6)]],
    constant float& avg_bal             [[buffer(7)]],
    constant uint& valid_prefix_mask    [[buffer(8)]],
    const device int* prefix_to_bin     [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= cust_size) return;
    float bal = c_acctbal[tid];
    if (bal <= avg_bal) return;
    int prefix = c_phone_prefix[tid];
    if (prefix < 0 || prefix > 31) return;
    if (!((valid_prefix_mask >> (uint)prefix) & 1u)) return;
    int ck = c_custkey[tid];
    if (bitmap_test(cust_bitmap, ck)) return;
    int bin = prefix_to_bin[prefix];
    if (bin < 0 || bin > 6) return;
    atomic_fetch_add_explicit(&result_count[bin], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&result_sum[bin], bal, memory_order_relaxed);
}
)METAL";
    result.kernels.push_back({"gen_q22_avg_balance", 256, false});
    result.kernels.push_back({"gen_q22_build_orders_bitmap", 256, false});
    result.kernels.push_back({"gen_q22_final_aggregate", 256, false});
}

// ===================================================================
// Q2: Minimum Cost Supplier
// ===================================================================
static void generateQ2Kernels(std::ostringstream& out, GeneratedKernels& result) {
    out << R"METAL(
// Q2 result struct for compact output
struct GenQ2MatchResult {
    int partkey;
    int suppkey;
    uint supplycost_cents;
};

// Q2 Kernel 1: Filter parts by p_size = 15 AND p_type LIKE '%BRASS' → bitmap
kernel void gen_q2_filter_part(
    const device int* p_partkey   [[buffer(0)]],
    const device int* p_size      [[buffer(1)]],
    const device char* p_type     [[buffer(2)]],
    device atomic_uint* part_bitmap [[buffer(3)]],
    constant uint& part_size      [[buffer(4)]],
    constant int& target_size     [[buffer(5)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= part_size) return;
    if (p_size[index] != target_size) return;

    const device char* type_str = p_type + index * 25;
    int len = 0;
    for (int i = 0; i < 25; i++) {
        if (type_str[i] != '\0') len = i + 1;
    }
    if (len < 5) return;
    if (type_str[len-5] != 'B' || type_str[len-4] != 'R' ||
        type_str[len-3] != 'A' || type_str[len-2] != 'S' ||
        type_str[len-1] != 'S') return;

    int key = p_partkey[index];
    bitmap_set(part_bitmap, key);
}

// Q2 Kernel 2: Find minimum supplycost per partkey (atomic_fetch_min on uint cents)
kernel void gen_q2_find_min_cost(
    const device int* ps_partkey      [[buffer(0)]],
    const device int* ps_suppkey      [[buffer(1)]],
    const device float* ps_supplycost [[buffer(2)]],
    const device uint* part_bitmap    [[buffer(3)]],
    const device uint* supplier_bitmap [[buffer(4)]],
    device atomic_uint* min_cost      [[buffer(5)]],
    constant uint& partsupp_size      [[buffer(6)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= partsupp_size) return;
    int pk = ps_partkey[index];
    if (!bitmap_test(part_bitmap, pk)) return;
    int sk = ps_suppkey[index];
    if (!bitmap_test(supplier_bitmap, sk)) return;
    uint cost_cents = (uint)(ps_supplycost[index] * 100.0f + 0.5f);
    atomic_fetch_min_explicit(&min_cost[pk], cost_cents, memory_order_relaxed);
}

// Q2 Kernel 3: Match suppliers with minimum cost → compact output
kernel void gen_q2_match_suppliers(
    const device int* ps_partkey      [[buffer(0)]],
    const device int* ps_suppkey      [[buffer(1)]],
    const device float* ps_supplycost [[buffer(2)]],
    const device uint* part_bitmap    [[buffer(3)]],
    const device uint* supplier_bitmap [[buffer(4)]],
    const device uint* min_cost       [[buffer(5)]],
    device GenQ2MatchResult* results  [[buffer(6)]],
    device atomic_uint& result_count  [[buffer(7)]],
    constant uint& partsupp_size      [[buffer(8)]],
    constant uint& max_results        [[buffer(9)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= partsupp_size) return;
    int pk = ps_partkey[index];
    if (!bitmap_test(part_bitmap, pk)) return;
    int sk = ps_suppkey[index];
    if (!bitmap_test(supplier_bitmap, sk)) return;
    uint cost_cents = (uint)(ps_supplycost[index] * 100.0f + 0.5f);
    if (cost_cents != min_cost[pk]) return;
    uint pos = atomic_fetch_add_explicit(&result_count, 1, memory_order_relaxed);
    if (pos < max_results) {
        results[pos].partkey = pk;
        results[pos].suppkey = sk;
        results[pos].supplycost_cents = cost_cents;
    }
}
)METAL";
    result.kernels.push_back({"gen_q2_filter_part", 256, false});
    result.kernels.push_back({"gen_q2_find_min_cost", 256, false});
    result.kernels.push_back({"gen_q2_match_suppliers", 256, false});
}

// ===================================================================
// Q18: Large Volume Customer
// ===================================================================
static void generateQ18Kernels(std::ostringstream& out, GeneratedKernels& result) {
    out << R"METAL(
// Q18 Kernel 1: Aggregate SUM(l_quantity) per orderkey
kernel void gen_q18_aggregate_quantity(
    const device int* l_orderkey        [[buffer(0)]],
    const device float* l_quantity      [[buffer(1)]],
    device atomic_float* qty_map        [[buffer(2)]],
    constant uint& lineitem_size        [[buffer(3)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    uint global_id = (group_id * threads_per_group) + thread_id_in_group;
    for (uint i = global_id; i < lineitem_size; i += grid_size) {
        int okey = l_orderkey[i];
        float qty = l_quantity[i];
        atomic_fetch_add_explicit(&qty_map[okey], qty, memory_order_relaxed);
    }
}

// Q18 output row struct
struct GenQ18OutputRow {
    int o_orderkey;
    int o_custkey;
    int o_orderdate;
    float o_totalprice;
    float sum_qty;
};

// Q18 Kernel 2: Filter orders with sum(qty) > threshold, compact output
kernel void gen_q18_filter_orders(
    const device int* o_orderkey        [[buffer(0)]],
    const device int* o_custkey         [[buffer(1)]],
    const device int* o_orderdate       [[buffer(2)]],
    const device float* o_totalprice    [[buffer(3)]],
    const device float* qty_map         [[buffer(4)]],
    device GenQ18OutputRow* output      [[buffer(5)]],
    device atomic_uint& output_count    [[buffer(6)]],
    constant uint& orders_size          [[buffer(7)]],
    constant uint& qty_map_size         [[buffer(8)]],
    constant float& threshold           [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= orders_size) return;
    int okey = o_orderkey[tid];
    if ((uint)okey >= qty_map_size) return;
    float sq = qty_map[okey];
    if (sq <= threshold) return;
    uint pos = atomic_fetch_add_explicit(&output_count, 1u, memory_order_relaxed);
    output[pos].o_orderkey = okey;
    output[pos].o_custkey = o_custkey[tid];
    output[pos].o_orderdate = o_orderdate[tid];
    output[pos].o_totalprice = o_totalprice[tid];
    output[pos].sum_qty = sq;
}
)METAL";
    result.kernels.push_back({"gen_q18_aggregate_quantity", 1024, false});
    result.kernels.push_back({"gen_q18_filter_orders", 256, false});
}

// ===================================================================
// Q9: Product Type Profit Measure
// ===================================================================
static void generateQ9Kernels(std::ostringstream& out, GeneratedKernels& result) {
    out << R"METAL(
// Q9 structs
struct GenQ9PartSuppEntry {
    atomic_int partkey;
    atomic_int suppkey;
    atomic_int idx;
    int _pad;
};

struct GenQ9Aggregates {
    atomic_uint key;
    atomic_float profit;
};

// Hash table entry for orders (key + value)
struct GenQ9OrdersHTEntry {
    atomic_int key;
    atomic_int value;
};

// Q9 Kernel 1: Build part bitmap — filter p_name LIKE '%green%' (SWAR)
kernel void gen_q9_build_part_bitmap(
    const device int* p_partkey     [[buffer(0)]],
    const device char* p_name       [[buffer(1)]],
    device atomic_uint* part_bitmap [[buffer(2)]],
    constant uint& part_size        [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= part_size) return;
    const device uchar* name = (const device uchar*)(p_name + index * 55);
    bool match = false;
    const uint g_spread = 0x67676767u;
    for (int w = 0; w < 13 && !match; ++w) {
        uint word = (uint)name[w*4]
                  | ((uint)name[w*4+1] << 8)
                  | ((uint)name[w*4+2] << 16)
                  | ((uint)name[w*4+3] << 24);
        uint xor_val = word ^ g_spread;
        if (((xor_val - 0x01010101u) & ~xor_val & 0x80808080u) == 0) continue;
        int base = w * 4;
        for (int b = 0; b < 4 && base + b <= 50; ++b) {
            if (name[base+b]=='g' && name[base+b+1]=='r' && name[base+b+2]=='e' &&
                name[base+b+3]=='e' && name[base+b+4]=='n') { match = true; break; }
        }
    }
    if (match) bitmap_set(part_bitmap, p_partkey[index]);
}

// Q9 Kernel 2: Build supplier -> nationkey direct map
kernel void gen_q9_build_supplier_map(
    const device int* s_suppkey     [[buffer(0)]],
    const device int* s_nationkey   [[buffer(1)]],
    device int* supplier_nation_map [[buffer(2)]],
    constant uint& supplier_size    [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= supplier_size) return;
    supplier_nation_map[s_suppkey[index]] = s_nationkey[index];
}

// Q9 Kernel 3: Build partsupp HT (open-addressing with CAS)
kernel void gen_q9_build_partsupp_ht(
    const device int* ps_partkey        [[buffer(0)]],
    const device int* ps_suppkey        [[buffer(1)]],
    device GenQ9PartSuppEntry* ht       [[buffer(2)]],
    constant uint& partsupp_size        [[buffer(3)]],
    constant uint& ht_size              [[buffer(4)]],
    const device uint* part_bitmap      [[buffer(5)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= partsupp_size) return;
    int pk = ps_partkey[index];
    if (!bitmap_test(part_bitmap, pk)) return;
    int sk = ps_suppkey[index];
    uint mix = (uint)pk * 0x9E3779B1u ^ (uint)sk * 0x85EBCA77u;
    uint mask = ht_size - 1;
    uint h = mix & mask;
    for (uint i = 0; i <= mask; ++i) {
        uint probe = (h + i) & mask;
        int expected = -1;
        if (atomic_compare_exchange_weak_explicit(&ht[probe].partkey, &expected, pk,
                memory_order_relaxed, memory_order_relaxed)) {
            atomic_store_explicit(&ht[probe].suppkey, sk, memory_order_relaxed);
            atomic_store_explicit(&ht[probe].idx, (int)index, memory_order_relaxed);
            return;
        }
        int cur_pk = atomic_load_explicit(&ht[probe].partkey, memory_order_relaxed);
        if (cur_pk == -1) return;
        if (cur_pk == pk) {
            int cur_sk = atomic_load_explicit(&ht[probe].suppkey, memory_order_relaxed);
            if (cur_sk == sk) {
                atomic_store_explicit(&ht[probe].idx, (int)index, memory_order_relaxed);
                return;
            }
        }
    }
}

// Q9 Kernel 4: Build orders HT (orderkey -> year)
kernel void gen_q9_build_orders_ht(
    const device int* o_orderkey    [[buffer(0)]],
    const device int* o_orderdate   [[buffer(1)]],
    device GenQ9OrdersHTEntry* ht   [[buffer(2)]],
    constant uint& orders_size      [[buffer(3)]],
    constant uint& ht_size          [[buffer(4)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= orders_size) return;
    int key = o_orderkey[index];
    int year = o_orderdate[index] / 10000;
    uint mask = ht_size - 1;
    uint h = sf100_hash_key(key) & mask;
    for (uint i = 0; i <= mask; ++i) {
        uint probe = (h + i) & mask;
        int expected = -1;
        if (atomic_compare_exchange_weak_explicit(&ht[probe].key, &expected, key,
                memory_order_relaxed, memory_order_relaxed)) {
            atomic_store_explicit(&ht[probe].value, year, memory_order_relaxed);
            return;
        }
    }
}

// Q9 Kernel 5: Probe lineitem + direct global aggregation
kernel void gen_q9_probe_and_aggregate(
    const device int* l_suppkey         [[buffer(0)]],
    const device int* l_partkey         [[buffer(1)]],
    const device int* l_orderkey        [[buffer(2)]],
    const device float* l_extendedprice [[buffer(3)]],
    const device float* l_discount      [[buffer(4)]],
    const device float* l_quantity      [[buffer(5)]],
    const device float* ps_supplycost   [[buffer(6)]],
    const device uint* part_bitmap      [[buffer(7)]],
    const device int* supplier_nation_map [[buffer(8)]],
    const device GenQ9PartSuppEntry* partsupp_ht [[buffer(9)]],
    const device GenQ9OrdersHTEntry* orders_ht   [[buffer(10)]],
    device GenQ9Aggregates* global_agg  [[buffer(11)]],
    constant uint& lineitem_size        [[buffer(12)]],
    constant uint& partsupp_ht_size     [[buffer(13)]],
    constant uint& orders_ht_size       [[buffer(14)]],
    constant uint& agg_size             [[buffer(15)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    const uint global_tid = (group_id * threads_per_group) + thread_id_in_group;
    const uint BATCH = 4;
    const uint stride = grid_size * BATCH;
    const uint agg_mask = agg_size - 1;
    const uint ps_mask = partsupp_ht_size - 1;
    const uint ord_mask = orders_ht_size - 1;

    for (uint base = global_tid * BATCH; base < lineitem_size; base += stride) {
        for (uint k = 0; k < BATCH; ++k) {
            uint i = base + k;
            if (i >= lineitem_size) break;

            int partkey = l_partkey[i];
            if (!bitmap_test(part_bitmap, partkey)) continue;
            int suppkey = l_suppkey[i];
            int nationkey = supplier_nation_map[suppkey];
            if (nationkey == -1) continue;

            // Probe partsupp HT
            int ps_idx = -1;
            uint ps_hash = ((uint)partkey * 0x9E3779B1u ^ (uint)suppkey * 0x85EBCA77u) & ps_mask;
            for (uint j = 0; j <= ps_mask; ++j) {
                uint probe = (ps_hash + j) & ps_mask;
                int pk2 = atomic_load_explicit(&partsupp_ht[probe].partkey, memory_order_relaxed);
                if (pk2 == -1) break;
                if (pk2 == partkey) {
                    int sk2 = atomic_load_explicit(&partsupp_ht[probe].suppkey, memory_order_relaxed);
                    if (sk2 == suppkey) {
                        ps_idx = atomic_load_explicit(&partsupp_ht[probe].idx, memory_order_relaxed);
                        break;
                    }
                }
            }
            if (ps_idx == -1) continue;

            // Probe orders HT
            int orderkey = l_orderkey[i];
            int year = -1;
            uint ord_hash = sf100_hash_key(orderkey) & ord_mask;
            for (uint j = 0; j <= ord_mask; ++j) {
                uint probe = (ord_hash + j) & ord_mask;
                int o_key = atomic_load_explicit(&orders_ht[probe].key, memory_order_relaxed);
                if (o_key == orderkey) {
                    year = atomic_load_explicit(&orders_ht[probe].value, memory_order_relaxed);
                    break;
                }
                if (o_key == -1) break;
            }
            if (year == -1) continue;

            float profit = l_extendedprice[i] * (1.0f - l_discount[i]) - ps_supplycost[ps_idx] * l_quantity[i];
            uint agg_key = (uint)(nationkey << 16) | (uint)year;
            uint agg_hash = agg_key & agg_mask;
            for (uint m = 0; m <= agg_mask; ++m) {
                uint probe = (agg_hash + m) & agg_mask;
                uint expected = 0;
                if (atomic_compare_exchange_weak_explicit(&global_agg[probe].key, &expected, agg_key,
                        memory_order_relaxed, memory_order_relaxed)) {
                    atomic_fetch_add_explicit(&global_agg[probe].profit, profit, memory_order_relaxed);
                    break;
                }
                if (atomic_load_explicit(&global_agg[probe].key, memory_order_relaxed) == agg_key) {
                    atomic_fetch_add_explicit(&global_agg[probe].profit, profit, memory_order_relaxed);
                    break;
                }
            }
        }
    }
}
)METAL";
    result.kernels.push_back({"gen_q9_build_part_bitmap", 256, true});
    result.kernels.push_back({"gen_q9_build_supplier_map", 256, true});
    result.kernels.push_back({"gen_q9_build_partsupp_ht", 256, true});
    result.kernels.push_back({"gen_q9_build_orders_ht", 256, true});
    result.kernels.push_back({"gen_q9_probe_and_aggregate", 1024, false});
}

// ===================================================================
// Q16: Parts/Supplier Relationship
// ===================================================================
static void generateQ16Kernels(std::ostringstream& out, GeneratedKernels& result) {
    out << R"METAL(
// Q16 Kernel 1: Scan partsupp, set bits in per-group bitmaps
kernel void gen_q16_scan_and_bitmap(
    const device int* ps_partkey        [[buffer(0)]],
    const device int* ps_suppkey        [[buffer(1)]],
    const device int* part_group_map    [[buffer(2)]],
    const device uint* complaint_bitmap [[buffer(3)]],
    device atomic_uint* group_bitmaps   [[buffer(4)]],
    constant uint& partsupp_size        [[buffer(5)]],
    constant uint& part_map_size        [[buffer(6)]],
    constant uint& bv_ints              [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= partsupp_size) return;
    int pk = ps_partkey[tid];
    if ((uint)pk >= part_map_size) return;
    int gid = part_group_map[pk];
    if (gid < 0) return;
    int sk = ps_suppkey[tid];
    if (bitmap_test(complaint_bitmap, sk)) return;
    uint word_idx = (uint)gid * bv_ints + ((uint)sk >> 5);
    uint bit = 1u << ((uint)sk & 31u);
    atomic_fetch_or_explicit(&group_bitmaps[word_idx], bit, memory_order_relaxed);
}

// Q16 Kernel 2: Popcount each group's bitmap
kernel void gen_q16_popcount(
    const device uint* group_bitmaps    [[buffer(0)]],
    device uint* group_counts           [[buffer(1)]],
    constant uint& num_groups           [[buffer(2)]],
    constant uint& bv_ints              [[buffer(3)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint tid_in_group [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint gid = group_id;
    if (gid >= num_groups) return;
    const device uint* bv = group_bitmaps + (uint64_t)gid * bv_ints;
    uint local_count = 0;
    for (uint w = tid_in_group; w < bv_ints; w += tg_size) {
        local_count += popcount(bv[w]);
    }
    threadgroup uint shared[32];
    uint r = tg_reduce_uint(local_count, tid_in_group, tg_size, shared);
    if (tid_in_group == 0) group_counts[gid] = r;
}
)METAL";
    result.kernels.push_back({"gen_q16_scan_and_bitmap", 256, true});
    result.kernels.push_back({"gen_q16_popcount", 256, false});
}

// ===================================================================
// Q20: Potential Part Promotion
// ===================================================================
static void generateQ20Kernels(std::ostringstream& out, GeneratedKernels& result) {
    out << R"METAL(
struct GenQ20HTEntry {
    atomic_int key_hi;
    atomic_int key_lo;
    atomic_float value;
};

// Q20 Kernel 1: Aggregate lineitem quantity into HT keyed by (partkey, suppkey)
kernel void gen_q20_aggregate_lineitem(
    const device int* l_partkey     [[buffer(0)]],
    const device int* l_suppkey     [[buffer(1)]],
    const device float* l_quantity  [[buffer(2)]],
    const device int* l_shipdate    [[buffer(3)]],
    const device uint* part_bitmap  [[buffer(4)]],
    device GenQ20HTEntry* ht        [[buffer(5)]],
    constant uint& lineitem_size    [[buffer(6)]],
    constant uint& ht_mask          [[buffer(7)]],
    constant int& date_start        [[buffer(8)]],
    constant int& date_end          [[buffer(9)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    uint global_id = (group_id * threads_per_group) + thread_id_in_group;
    for (uint i = global_id; i < lineitem_size; i += grid_size) {
        int date = l_shipdate[i];
        if (date < date_start || date >= date_end) continue;
        int pk = l_partkey[i];
        if (!bitmap_test(part_bitmap, pk)) continue;
        int sk = l_suppkey[i];
        float qty = l_quantity[i];
        uint h = sf100_hash_key(pk * 100001 + sk);
        for (uint j = 0; j <= ht_mask; j++) {
            uint slot = (h + j) & ht_mask;
            int expected = -1;
            if (atomic_compare_exchange_weak_explicit(&ht[slot].key_hi, &expected, pk,
                    memory_order_relaxed, memory_order_relaxed)) {
                atomic_store_explicit(&ht[slot].key_lo, sk, memory_order_relaxed);
                atomic_fetch_add_explicit(&ht[slot].value, qty, memory_order_relaxed);
                break;
            }
            int cur_pk = atomic_load_explicit(&ht[slot].key_hi, memory_order_relaxed);
            int cur_sk = atomic_load_explicit(&ht[slot].key_lo, memory_order_relaxed);
            if (cur_pk == pk && cur_sk == sk) {
                atomic_fetch_add_explicit(&ht[slot].value, qty, memory_order_relaxed);
                break;
            }
        }
    }
}

// Q20 Kernel 2: Probe partsupp against HT, check threshold, set qualifying bitmap
kernel void gen_q20_probe_partsupp(
    const device int* ps_partkey        [[buffer(0)]],
    const device int* ps_suppkey        [[buffer(1)]],
    const device int* ps_availqty       [[buffer(2)]],
    const device uint* part_bitmap      [[buffer(3)]],
    const device uint* canada_bitmap    [[buffer(4)]],
    const device GenQ20HTEntry* ht      [[buffer(5)]],
    device atomic_uint* qual_bitmap     [[buffer(6)]],
    constant uint& partsupp_size        [[buffer(7)]],
    constant uint& ht_mask              [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= partsupp_size) return;
    int pk = ps_partkey[tid];
    if (!bitmap_test(part_bitmap, pk)) return;
    int sk = ps_suppkey[tid];
    if (!bitmap_test(canada_bitmap, sk)) return;
    uint h = sf100_hash_key(pk * 100001 + sk);
    for (uint j = 0; j <= ht_mask; j++) {
        uint slot = (h + j) & ht_mask;
        int cur_pk = atomic_load_explicit(&ht[slot].key_hi, memory_order_relaxed);
        if (cur_pk == -1) break;
        if (cur_pk == pk) {
            int cur_sk = atomic_load_explicit(&ht[slot].key_lo, memory_order_relaxed);
            if (cur_sk == sk) {
                float sum_qty = atomic_load_explicit(&ht[slot].value, memory_order_relaxed);
                if ((float)ps_availqty[tid] > 0.5f * sum_qty) {
                    atomic_fetch_or_explicit(&qual_bitmap[(uint)sk >> 5],
                                            1u << ((uint)sk & 31u), memory_order_relaxed);
                }
                return;
            }
        }
    }
}
)METAL";
    result.kernels.push_back({"gen_q20_aggregate_lineitem", 1024, false});
    result.kernels.push_back({"gen_q20_probe_partsupp", 256, true});
}

// ===================================================================
// Q21: Suppliers Who Kept Orders Waiting
// ===================================================================
static void generateQ21Kernels(std::ostringstream& out, GeneratedKernels& result) {
    out << R"METAL(
// Q21 Pass 1: Build per-order supplier tracking (CAS multi-value)
kernel void gen_q21_build_order_tracking(
    const device int* l_orderkey        [[buffer(0)]],
    const device int* l_suppkey         [[buffer(1)]],
    const device int* l_receiptdate     [[buffer(2)]],
    const device int* l_commitdate      [[buffer(3)]],
    const device int* orders_status_map [[buffer(4)]],
    device atomic_int* first_supp       [[buffer(5)]],
    device atomic_uint* multi_supp_bm   [[buffer(6)]],
    device atomic_int* late_supp        [[buffer(7)]],
    device atomic_uint* multi_late_bm   [[buffer(8)]],
    constant uint& lineitem_size        [[buffer(9)]],
    constant uint& map_size             [[buffer(10)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    uint global_id = (group_id * threads_per_group) + thread_id_in_group;
    for (uint i = global_id; i < lineitem_size; i += grid_size) {
        int okey = l_orderkey[i];
        if ((uint)okey >= map_size) continue;
        int status = orders_status_map[okey];
        if (status != 1) continue;
        int sk = l_suppkey[i];
        int expected = -1;
        if (!atomic_compare_exchange_weak_explicit(&first_supp[okey], &expected, sk,
                memory_order_relaxed, memory_order_relaxed)) {
            if (expected != sk) bitmap_set(multi_supp_bm, okey);
        }
        bool is_late = (l_receiptdate[i] > l_commitdate[i]);
        if (is_late) {
            int exp_late = -1;
            if (!atomic_compare_exchange_weak_explicit(&late_supp[okey], &exp_late, sk,
                    memory_order_relaxed, memory_order_relaxed)) {
                if (exp_late != sk) bitmap_set(multi_late_bm, okey);
            }
        }
    }
}

// Q21 Pass 2: Count qualifying lineitems per SAUDI ARABIA supplier
kernel void gen_q21_count_qualifying(
    const device int* l_orderkey        [[buffer(0)]],
    const device int* l_suppkey         [[buffer(1)]],
    const device int* l_receiptdate     [[buffer(2)]],
    const device int* l_commitdate      [[buffer(3)]],
    const device int* orders_status_map [[buffer(4)]],
    const device int* first_supp        [[buffer(5)]],
    const device uint* multi_supp_bm    [[buffer(6)]],
    const device int* late_supp         [[buffer(7)]],
    const device uint* multi_late_bm    [[buffer(8)]],
    const device uint* sa_supp_bitmap   [[buffer(9)]],
    device atomic_uint* supp_count      [[buffer(10)]],
    constant uint& lineitem_size        [[buffer(11)]],
    constant uint& map_size             [[buffer(12)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    uint global_id = (group_id * threads_per_group) + thread_id_in_group;
    for (uint i = global_id; i < lineitem_size; i += grid_size) {
        int okey = l_orderkey[i];
        if ((uint)okey >= map_size) continue;
        int status = orders_status_map[okey];
        if (status != 1) continue;
        int sk = l_suppkey[i];
        if (l_receiptdate[i] <= l_commitdate[i]) continue;
        if (!bitmap_test(sa_supp_bitmap, sk)) continue;
        if (!bitmap_test(multi_supp_bm, okey)) continue;
        if (bitmap_test(multi_late_bm, okey)) continue;
        if (late_supp[okey] != sk) continue;
        atomic_fetch_add_explicit(&supp_count[sk], 1u, memory_order_relaxed);
    }
}
)METAL";
    result.kernels.push_back({"gen_q21_build_order_tracking", 1024, false});
    result.kernels.push_back({"gen_q21_count_qualifying", 1024, false});
}

} // anonymous namespace

// ===================================================================
// PUBLIC: generateMetal
// ===================================================================

GeneratedKernels generateMetal(const QueryPlan& plan) {
    GeneratedKernels result;
    std::ostringstream out;

    // Common header
    out << METAL_COMMON_HEADER;

    // Dispatch to query-specific generator
    if (plan.name == "Q1") generateQ1Kernels(out, result);
    else if (plan.name == "Q6") generateQ6Kernels(out, result);
    else if (plan.name == "Q3") generateQ3Kernels(out, result);
    else if (plan.name == "Q14") generateQ14Kernels(out, result);
    else if (plan.name == "Q13") generateQ13Kernels(out, result);
    else if (plan.name == "Q4") generateQ4Kernels(out, result);
    else if (plan.name == "Q12") generateQ12Kernels(out, result);
    else if (plan.name == "Q19") generateQ19Kernels(out, result);
    else if (plan.name == "Q15") generateQ15Kernels(out, result);
    else if (plan.name == "Q11") generateQ11Kernels(out, result);
    else if (plan.name == "Q10") generateQ10Kernels(out, result);
    else if (plan.name == "Q5") generateQ5Kernels(out, result);
    else if (plan.name == "Q7") generateQ7Kernels(out, result);
    else if (plan.name == "Q8") generateQ8Kernels(out, result);
    else if (plan.name == "Q17") generateQ17Kernels(out, result);
    else if (plan.name == "Q22") generateQ22Kernels(out, result);
    else if (plan.name == "Q2") generateQ2Kernels(out, result);
    else if (plan.name == "Q18") generateQ18Kernels(out, result);
    else if (plan.name == "Q9") generateQ9Kernels(out, result);
    else if (plan.name == "Q16") generateQ16Kernels(out, result);
    else if (plan.name == "Q20") generateQ20Kernels(out, result);
    else if (plan.name == "Q21") generateQ21Kernels(out, result);
    else throw std::runtime_error("No Metal codegen for plan: " + plan.name);

    result.metalSource = out.str();
    return result;
}

} // namespace codegen
