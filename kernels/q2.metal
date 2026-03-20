#include "common.h"

// ===================================================================
// TPC-H Q2 KERNELS — Minimum Cost Supplier Query
// ===================================================================
/*
SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr,
       s_address, s_phone, s_comment
FROM part, supplier, partsupp, nation, region
WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey
  AND p_size = 15 AND p_type LIKE '%BRASS'
  AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey
  AND r_name = 'EUROPE'
  AND ps_supplycost = (
      SELECT MIN(ps_supplycost) FROM partsupp, supplier, nation, region
      WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey
        AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey
        AND r_name = 'EUROPE')
ORDER BY s_acctbal DESC, n_name, s_name, p_partkey
LIMIT 100;
*/

// Result struct for Q2 matching rows
struct Q2MatchResult {
    int partkey;
    int suppkey;
    uint supplycost_cents; // supplycost * 100 encoded as uint
};

// KERNEL 1: Filter parts by p_size = 15 AND p_type LIKE '%BRASS'
// Builds a bitmap on qualifying partkeys.
// p_type is stored as fixed-width 25-char field.
kernel void q2_filter_part_kernel(
    const device int* p_partkey   [[buffer(0)]],
    const device int* p_size      [[buffer(1)]],
    const device char* p_type     [[buffer(2)]],
    device atomic_uint* part_bitmap [[buffer(3)]],
    constant uint& part_size      [[buffer(4)]],
    constant int& target_size     [[buffer(5)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= part_size) return;

    // Filter: p_size = target_size
    if (p_size[index] != target_size) return;

    // Filter: p_type LIKE '%BRASS' (suffix match on 25-char field)
    const device char* type_str = p_type + index * 25;

    // Find the actual string length (last non-null char position + 1)
    int len = 0;
    for (int i = 0; i < 25; i++) {
        if (type_str[i] != '\0') len = i + 1;
    }

    // Check suffix "BRASS" (5 chars)
    if (len < 5) return;
    if (type_str[len-5] != 'B' || type_str[len-4] != 'R' ||
        type_str[len-3] != 'A' || type_str[len-2] != 'S' ||
        type_str[len-1] != 'S') return;

    int key = p_partkey[index];
    bitmap_set(part_bitmap, key);
}

// KERNEL 2: Find minimum supplycost per partkey
// Scans partsupp; for qualifying rows (part_bitmap & supplier_bitmap),
// uses atomic_fetch_min on uint-encoded supplycost (cents * 100).
kernel void q2_find_min_cost_kernel(
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

    // Encode supplycost as uint cents for atomic_min
    uint cost_cents = (uint)(ps_supplycost[index] * 100.0f + 0.5f);
    atomic_fetch_min_explicit(&min_cost[pk], cost_cents, memory_order_relaxed);
}

// KERNEL 3: Match suppliers with minimum cost
// Scans partsupp again, outputs rows where cost matches minimum.
kernel void q2_match_suppliers_kernel(
    const device int* ps_partkey      [[buffer(0)]],
    const device int* ps_suppkey      [[buffer(1)]],
    const device float* ps_supplycost [[buffer(2)]],
    const device uint* part_bitmap    [[buffer(3)]],
    const device uint* supplier_bitmap [[buffer(4)]],
    const device uint* min_cost       [[buffer(5)]],
    device Q2MatchResult* results     [[buffer(6)]],
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
