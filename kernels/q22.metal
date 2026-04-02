#include "common.h"

// ===================================================================
// TPC-H Q22 KERNELS — Global Sales Opportunity
// ===================================================================
// Phase 1: Compute average acctbal for qualifying country codes with bal > 0
// Phase 2: Build orders custkey bitmap
// Phase 3: Count/sum customers per country code (7 bins) with bal > avg and no orders

// Phase 1: Sum + count acctbal for qualifying customers with bal > 0
kernel void q22_avg_balance_kernel(
    const device int* c_phone_prefix    [[buffer(0)]],  // 2-digit prefix as int
    const device float* c_acctbal       [[buffer(1)]],
    device atomic_float& sum_bal        [[buffer(2)]],
    device atomic_uint& count_bal       [[buffer(3)]],
    constant uint& cust_size            [[buffer(4)]],
    constant uint& valid_prefix_mask    [[buffer(5)]],  // bitmap of valid 2-digit prefixes  
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

// Phase 2: Build orders custkey bitmap
kernel void q22_build_orders_bitmap_kernel(
    const device int* o_custkey         [[buffer(0)]],
    device atomic_uint* cust_bitmap     [[buffer(1)]],
    constant uint& orders_size          [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= orders_size) return;
    bitmap_set(cust_bitmap, o_custkey[tid]);
}

// Phase 3: Final aggregate — count/sum per country code
kernel void q22_final_aggregate_kernel(
    const device int* c_phone_prefix    [[buffer(0)]],
    const device float* c_acctbal       [[buffer(1)]],
    const device int* c_custkey         [[buffer(2)]],
    const device uint* cust_bitmap      [[buffer(3)]],  // orders bitmap
    device atomic_uint* result_count    [[buffer(4)]],  // 7 bins
    device atomic_float* result_sum     [[buffer(5)]],  // 7 bins
    constant uint& cust_size            [[buffer(6)]],
    constant float& avg_bal             [[buffer(7)]],
    constant uint& valid_prefix_mask    [[buffer(8)]],
    const device int* prefix_to_bin     [[buffer(9)]],  // prefix → bin index (0-6, -1 = invalid)
    uint tid [[thread_position_in_grid]])
{
    if (tid >= cust_size) return;
    float bal = c_acctbal[tid];
    if (bal <= avg_bal) return;
    int prefix = c_phone_prefix[tid];
    if (prefix < 0 || prefix > 31) return;
    if (!((valid_prefix_mask >> (uint)prefix) & 1u)) return;
    int ck = c_custkey[tid];
    if (bitmap_test(cust_bitmap, ck)) return;  // has orders → skip
    int bin = prefix_to_bin[prefix];
    if (bin < 0 || bin > 6) return;
    atomic_fetch_add_explicit(&result_count[bin], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&result_sum[bin], bal, memory_order_relaxed);
}
