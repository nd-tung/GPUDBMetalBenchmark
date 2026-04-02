#include "common.h"

// ===================================================================
// TPC-H Q15 KERNELS — Top Supplier
// ===================================================================
// SELECT s_suppkey, s_name, s_address, s_phone, total_revenue
// FROM supplier, revenue0
// WHERE s_suppkey = supplier_no
//   AND total_revenue = (SELECT MAX(total_revenue) FROM revenue0)
// revenue0: SUM(l_extendedprice * (1-l_discount)) per l_suppkey
//           WHERE l_shipdate in [1996-01-01, 1996-04-01)

kernel void q15_aggregate_revenue_kernel(
    const device int* l_suppkey         [[buffer(0)]],
    const device int* l_shipdate        [[buffer(1)]],
    const device float* l_extendedprice [[buffer(2)]],
    const device float* l_discount      [[buffer(3)]],
    device atomic_float* revenue_map    [[buffer(4)]],
    constant uint& data_size            [[buffer(5)]],
    constant int& date_start            [[buffer(6)]],
    constant int& date_end              [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= data_size) return;

    int date = l_shipdate[tid];
    if (date < date_start || date >= date_end) return;

    float revenue = l_extendedprice[tid] * (1.0f - l_discount[tid]);
    int sk = l_suppkey[tid];
    atomic_fetch_add_explicit(&revenue_map[sk], revenue, memory_order_relaxed);
}
