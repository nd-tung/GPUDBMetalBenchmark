#include "common.h"

// ===================================================================
// TPC-H Q1 KERNELS — Pricing Summary Report
// ===================================================================
/*
SELECT 
    l_returnflag,
    l_linestatus,
    SUM(l_quantity) AS sum_qty,
    SUM(l_extendedprice) AS sum_base_price,
    SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
    SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
    AVG(l_quantity) AS avg_qty,
    AVG(l_extendedprice) AS avg_price,
    AVG(l_discount) AS avg_disc,
    COUNT(*) AS count_order
FROM lineitem
WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus;
*/

// --- Q1 Specialized low-cardinality bins path ---
// Exploits the fact that Q1 groups only by l_returnflag in {A,N,R} and l_linestatus in {F,O}.
// We maintain 6 bins: (A,F)=0, (A,O)=1, (N,F)=2, (N,O)=3, (R,F)=4, (R,O)=5.
inline int q1_rf_index(char rf) {
    return (rf == 'A') ? 0 : (rf == 'N') ? 1 : (rf == 'R') ? 2 : -1;
}
inline int q1_ls_index(char ls) {
    return (ls == 'F') ? 0 : (ls == 'O') ? 1 : -1;
}

// --- Q1 Integer-cent two-pass path ---
// Stage 1: Per-thread local accumulation into 6 bins using integer cents (and basis points),
// then threadgroup reduction to one partial per bin per threadgroup. No atomics used.
// Monetary fields are accumulated in cents (int64). Discount is accumulated in basis points (bp, int32).
kernel void q1_bins_accumulate_int_stage1(
    const device int*   l_shipdate,
    const device char*  l_returnflag,
    const device char*  l_linestatus,
    const device float* l_quantity,
    const device float* l_extendedprice,
    const device float* l_discount,
    const device float* l_tax,
    // Outputs: one partial per threadgroup per bin (size = num_threadgroups * 6)
    device long*  p_sum_qty_cents,        // int64
    device long*  p_sum_base_cents,       // int64
    device long*  p_sum_disc_price_cents, // int64
    device long*  p_sum_charge_cents,     // int64
    device uint*  p_sum_discount_bp,      // uint32 (sum of basis points)
    device uint*  p_count,                // uint32
    constant uint& data_size,
    constant int&  cutoff_date,
    constant uint& num_threadgroups,
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    const int BINS = 6;
    // Per-thread local accumulators in integer units
    long sum_qty_c[BINS];
    long sum_base_c[BINS];
    long sum_disc_c[BINS];
    long sum_charge_c[BINS];
    uint sum_disc_bp[BINS];
    uint cnt[BINS];
    for (int b = 0; b < BINS; ++b) { sum_qty_c[b]=0; sum_base_c[b]=0; sum_disc_c[b]=0; sum_charge_c[b]=0; sum_disc_bp[b]=0u; cnt[b]=0u; }

    // Grid stride loop using actual dispatched num_threadgroups
    for (uint i = (group_id * threads_per_group) + thread_id_in_group; i < data_size; i += grid_size) {
        // [SEL] Selection: filter rows by shipdate and bin by (returnflag, linestatus)
        if (l_shipdate[i] > cutoff_date) continue;
        int rfi = q1_rf_index(l_returnflag[i]); if (rfi < 0) continue;
        int lsi = q1_ls_index(l_linestatus[i]); if (lsi < 0) continue;
        int bin = rfi * 2 + lsi; // 0..5

        // [PROJ] Projection: convert to fixed-point (cents for currency, basis points for percentages)
        float base = l_extendedprice[i];
        float qty  = l_quantity[i];
        float d    = l_discount[i];
        float t    = l_tax[i];

        long base_c = (long)floor(base * 100.0f + 0.5f);  // cents (int64)
        long qty_c  = (long)floor(qty  * 100.0f + 0.5f);  // cents (int64)
        int  d_bp   = (int)floor(d * 100.0f + 0.5f);      // basis points (int)
        int  t_bp   = (int)floor(t * 100.0f + 0.5f);      // basis points (int)

        // Compute derived columns in fixed-point
        long disc_c = (base_c * (long)(100 - d_bp) + 50) / 100;  // disc_price_cents
        long charge_c = (disc_c * (long)(100 + t_bp) + 50) / 100; // charge_cents

        // [AGG-LOCAL] Local accumulation: per-thread accumulators (6 bins, no atomics)
        sum_qty_c[bin]      += qty_c;
        sum_base_c[bin]     += base_c;
        sum_disc_c[bin]     += disc_c;
        sum_charge_c[bin]   += charge_c;
        sum_disc_bp[bin]    += (uint)d_bp;
        cnt[bin]            += 1u;
    }

    // SIMD-accelerated threadgroup reduction (2 barriers per metric instead of ~12)
    threadgroup long tg64[32];  // one per SIMD group
    threadgroup uint tg32[32];

    // [AGG-LOCAL] Threadgroup reduction for each bin and metric via SIMD groups
    for (int b = 0; b < BINS; ++b) {
        long r;
        r = tg_reduce_long(sum_qty_c[b], thread_id_in_group, threads_per_group, tg64);
        if (thread_id_in_group == 0) p_sum_qty_cents[group_id * BINS + b] = r;

        r = tg_reduce_long(sum_base_c[b], thread_id_in_group, threads_per_group, tg64);
        if (thread_id_in_group == 0) p_sum_base_cents[group_id * BINS + b] = r;

        r = tg_reduce_long(sum_disc_c[b], thread_id_in_group, threads_per_group, tg64);
        if (thread_id_in_group == 0) p_sum_disc_price_cents[group_id * BINS + b] = r;

        r = tg_reduce_long(sum_charge_c[b], thread_id_in_group, threads_per_group, tg64);
        if (thread_id_in_group == 0) p_sum_charge_cents[group_id * BINS + b] = r;

        uint u;
        u = tg_reduce_uint(sum_disc_bp[b], thread_id_in_group, threads_per_group, tg32);
        if (thread_id_in_group == 0) p_sum_discount_bp[group_id * BINS + b] = u;

        u = tg_reduce_uint(cnt[b], thread_id_in_group, threads_per_group, tg32);
        if (thread_id_in_group == 0) p_count[group_id * BINS + b] = u;
    }
}

// Stage 2: Parallel reduce per-threadgroup partials into final 6-bin results.
// Dispatched as 6 threadgroups × 1024 threads. Each threadgroup reduces one bin
// using grid-stride loop + SIMD two-level reduction (no atomics).
kernel void q1_bins_reduce_int_stage2(
    const device long* p_sum_qty_cents,
    const device long* p_sum_base_cents,
    const device long* p_sum_disc_price_cents,
    const device long* p_sum_charge_cents,
    const device uint* p_sum_discount_bp,
    const device uint* p_count,
    device long* out_sum_qty_cents,
    device long* out_sum_base_cents,
    device long* out_sum_disc_price_cents,
    device long* out_sum_charge_cents,
    device uint* out_sum_discount_bp,
    device uint* out_count,
    constant uint& num_threadgroups,
    uint group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    const uint BINS = 6;
    uint b = group_id; // one threadgroup per bin

    // Grid-stride accumulation across threadgroups' partials for this bin
    long s_qty = 0, s_base = 0, s_disc = 0, s_charge = 0;
    uint s_dbp = 0, s_cnt = 0;
    for (uint g = tid; g < num_threadgroups; g += tg_size) {
        uint idx = g * BINS + b;
        s_qty    += p_sum_qty_cents[idx];
        s_base   += p_sum_base_cents[idx];
        s_disc   += p_sum_disc_price_cents[idx];
        s_charge += p_sum_charge_cents[idx];
        s_dbp    += p_sum_discount_bp[idx];
        s_cnt    += p_count[idx];
    }

    // Two-level SIMD + threadgroup reduction
    threadgroup long tg64[32];
    threadgroup uint tg32[32];

    long r;
    r = tg_reduce_long(s_qty, tid, tg_size, tg64);
    if (tid == 0) out_sum_qty_cents[b] = r;
    r = tg_reduce_long(s_base, tid, tg_size, tg64);
    if (tid == 0) out_sum_base_cents[b] = r;
    r = tg_reduce_long(s_disc, tid, tg_size, tg64);
    if (tid == 0) out_sum_disc_price_cents[b] = r;
    r = tg_reduce_long(s_charge, tid, tg_size, tg64);
    if (tid == 0) out_sum_charge_cents[b] = r;

    uint u;
    u = tg_reduce_uint(s_dbp, tid, tg_size, tg32);
    if (tid == 0) out_sum_discount_bp[b] = u;
    u = tg_reduce_uint(s_cnt, tid, tg_size, tg32);
    if (tid == 0) out_count[b] = u;
}

// ===================================================================
// Q1 SF100 CHUNKED EXECUTION KERNELS
// ===================================================================

kernel void q1_chunked_stage1(
    const device int*   l_shipdate        [[buffer(0)]],
    const device char*  l_returnflag      [[buffer(1)]],
    const device char*  l_linestatus      [[buffer(2)]],
    const device float* l_quantity        [[buffer(3)]],
    const device float* l_extendedprice   [[buffer(4)]],
    const device float* l_discount        [[buffer(5)]],
    const device float* l_tax             [[buffer(6)]],
    device long*  p_sum_qty_cents         [[buffer(7)]],
    device long*  p_sum_base_cents        [[buffer(8)]],
    device long*  p_sum_disc_price_cents  [[buffer(9)]],
    device long*  p_sum_charge_cents      [[buffer(10)]],
    device uint*  p_sum_discount_bp       [[buffer(11)]],
    device uint*  p_count                 [[buffer(12)]],
    constant uint& chunk_size             [[buffer(13)]],
    constant int&  cutoff_date            [[buffer(14)]],
    constant uint& num_threadgroups       [[buffer(15)]],
    uint group_id         [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group  [[threads_per_threadgroup]],
    uint grid_size          [[threads_per_grid]])
{
    const int BINS = 6;
    long sum_qty_c[BINS], sum_base_c[BINS], sum_disc_c[BINS], sum_charge_c[BINS];
    uint sum_disc_bp[BINS], cnt[BINS];
    for (int b = 0; b < BINS; ++b) {
        sum_qty_c[b]=0; sum_base_c[b]=0; sum_disc_c[b]=0; sum_charge_c[b]=0;
        sum_disc_bp[b]=0u; cnt[b]=0u;
    }

    auto rf_index = [](char rf) -> int { return (rf=='A')?0:(rf=='N')?1:(rf=='R')?2:-1; };
    auto ls_index = [](char ls) -> int { return (ls=='F')?0:(ls=='O')?1:-1; };

    for (uint i = (group_id * threads_per_group) + thread_id_in_group;
         i < chunk_size; i += grid_size) {
        if (l_shipdate[i] > cutoff_date) continue;
        int rfi = rf_index(l_returnflag[i]); if (rfi < 0) continue;
        int lsi = ls_index(l_linestatus[i]); if (lsi < 0) continue;
        int bin = rfi * 2 + lsi;
        float base = l_extendedprice[i], qty = l_quantity[i], d = l_discount[i], t = l_tax[i];
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

    // SIMD-accelerated threadgroup reduction (2 barriers per metric instead of ~10)
    threadgroup long tg64[32];  // one per SIMD group (max 1024/32 = 32)
    threadgroup uint tg32[32];

    for (int b = 0; b < BINS; ++b) {
        long r;
        r = tg_reduce_long(sum_qty_c[b], thread_id_in_group, threads_per_group, tg64);
        if (thread_id_in_group == 0) p_sum_qty_cents[group_id * BINS + b] = r;

        r = tg_reduce_long(sum_base_c[b], thread_id_in_group, threads_per_group, tg64);
        if (thread_id_in_group == 0) p_sum_base_cents[group_id * BINS + b] = r;

        r = tg_reduce_long(sum_disc_c[b], thread_id_in_group, threads_per_group, tg64);
        if (thread_id_in_group == 0) p_sum_disc_price_cents[group_id * BINS + b] = r;

        r = tg_reduce_long(sum_charge_c[b], thread_id_in_group, threads_per_group, tg64);
        if (thread_id_in_group == 0) p_sum_charge_cents[group_id * BINS + b] = r;

        uint u;
        u = tg_reduce_uint(sum_disc_bp[b], thread_id_in_group, threads_per_group, tg32);
        if (thread_id_in_group == 0) p_sum_discount_bp[group_id * BINS + b] = u;

        u = tg_reduce_uint(cnt[b], thread_id_in_group, threads_per_group, tg32);
        if (thread_id_in_group == 0) p_count[group_id * BINS + b] = u;
    }
}

// Q1 chunked stage2: identical to q1_bins_reduce_int_stage2
kernel void q1_chunked_stage2(
    const device long* p_sum_qty_cents,
    const device long* p_sum_base_cents,
    const device long* p_sum_disc_price_cents,
    const device long* p_sum_charge_cents,
    const device uint* p_sum_discount_bp,
    const device uint* p_count,
    device long* out_sum_qty_cents,
    device long* out_sum_base_cents,
    device long* out_sum_disc_price_cents,
    device long* out_sum_charge_cents,
    device uint* out_sum_discount_bp,
    device uint* out_count,
    constant uint& num_threadgroups,
    uint index [[thread_position_in_grid]])
{
    if (index != 0) return;
    const uint BINS = 6;
    for (uint b = 0; b < BINS; ++b) {
        long s_qty = 0, s_base = 0, s_disc = 0, s_charge = 0;
        uint s_dbp = 0, s_cnt = 0;
        for (uint g = 0; g < num_threadgroups; ++g) {
            uint idx = g * BINS + b;
            s_qty    += p_sum_qty_cents[idx];
            s_base   += p_sum_base_cents[idx];
            s_disc   += p_sum_disc_price_cents[idx];
            s_charge += p_sum_charge_cents[idx];
            s_dbp    += p_sum_discount_bp[idx];
            s_cnt    += p_count[idx];
        }
        out_sum_qty_cents[b]        = s_qty;
        out_sum_base_cents[b]       = s_base;
        out_sum_disc_price_cents[b] = s_disc;
        out_sum_charge_cents[b]     = s_charge;
        out_sum_discount_bp[b]      = s_dbp;
        out_count[b]                = s_cnt;
    }
}
