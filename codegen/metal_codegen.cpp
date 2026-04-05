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
    else throw std::runtime_error("No Metal codegen for plan: " + plan.name);

    result.metalSource = out.str();
    return result;
}

} // namespace codegen
