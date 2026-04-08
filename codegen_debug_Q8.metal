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
    atomic_fetch_or_explicit(&bitmap[(uint)key >> 5],
                             1u << ((uint)key & 31u),
                             memory_order_relaxed);
}

inline void atomic_add_long_pair(device atomic_uint* lo,
                                 device atomic_uint* hi,
                                 long val) {
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


// === Phase 0: Q8_build_nation_bitmap ===
kernel void Q8_build_nation_bitmap(
    constant int& america_rk [[buffer(0)]],
    device atomic_uint* d_america_bitmap [[buffer(1)]],
    device const int* n_nationkey [[buffer(2)]],
    device const int* n_regionkey [[buffer(3)]],
    constant uint& n_nation [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    for (uint i = tid; i < n_nation; i += tpg) {
        if (n_regionkey[i] == america_rk) {
            bitmap_set(d_america_bitmap, n_nationkey[i]);
        }
    }
}

// === Phase 1: Q8_build_part_bitmap ===
kernel void Q8_build_part_bitmap(
    device atomic_uint* d_part_bitmap [[buffer(0)]],
    device const int* p_partkey [[buffer(1)]],
    device const char* p_type [[buffer(2)]],
    constant uint& n_part [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    for (uint i = tid; i < n_part; i += tpg) {
        if (p_type[i * 25] == 'E' && p_type[i * 25 + 1] == 'C' && p_type[i * 25 + 2] == 'O' && p_type[i * 25 + 3] == 'N' && p_type[i * 25 + 4] == 'O' && p_type[i * 25 + 5] == 'M' && p_type[i * 25 + 6] == 'Y' && p_type[i * 25 + 7] == ' ' && p_type[i * 25 + 8] == 'A' && p_type[i * 25 + 9] == 'N' && p_type[i * 25 + 10] == 'O' && p_type[i * 25 + 11] == 'D' && p_type[i * 25 + 12] == 'I' && p_type[i * 25 + 13] == 'Z' && p_type[i * 25 + 14] == 'E' && p_type[i * 25 + 15] == 'D' && p_type[i * 25 + 16] == ' ' && p_type[i * 25 + 17] == 'S' && p_type[i * 25 + 18] == 'T' && p_type[i * 25 + 19] == 'E' && p_type[i * 25 + 20] == 'E' && p_type[i * 25 + 21] == 'L') {
            bitmap_set(d_part_bitmap, p_partkey[i]);
        }
    }
}

// === Phase 2: Q8_build_cust_map ===
kernel void Q8_build_cust_map(
    device int* d_cust_nation_map [[buffer(0)]],
    device const uint* d_america_bitmap [[buffer(1)]],
    device const int* c_custkey [[buffer(2)]],
    device const int* c_nationkey [[buffer(3)]],
    constant uint& n_customer [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    for (uint i = tid; i < n_customer; i += tpg) {
        if (bitmap_test(d_america_bitmap, c_nationkey[i])) {
            d_cust_nation_map[c_custkey[i]] = c_nationkey[i];
        }
    }
}

// === Phase 3: Q8_build_supp_map ===
kernel void Q8_build_supp_map(
    device int* d_supp_nation_map [[buffer(0)]],
    device const int* s_suppkey [[buffer(1)]],
    device const int* s_nationkey [[buffer(2)]],
    constant uint& n_supplier [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    for (uint i = tid; i < n_supplier; i += tpg) {
        d_supp_nation_map[s_suppkey[i]] = s_nationkey[i];
    }
}

// === Phase 4: Q8_build_orders_map ===
kernel void Q8_build_orders_map(
    device int* d_orders_year_map [[buffer(0)]],
    device int* d_cust_nation_map [[buffer(1)]],
    device const int* o_orderkey [[buffer(2)]],
    device const int* o_custkey [[buffer(3)]],
    device const int* o_orderdate [[buffer(4)]],
    constant uint& n_orders [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    for (uint i = tid; i < n_orders; i += tpg) {
        if (o_orderdate[i] >= 19950101 && o_orderdate[i] <= 19961231) {
            int _cust_nk = d_cust_nation_map[o_custkey[i]];
            if (_cust_nk != -1) {
                d_orders_year_map[o_orderkey[i]] = o_orderdate[i] / 10000;
            }
        }
    }
}

// === Phase 5: Q8_probe_aggregate ===
kernel void Q8_probe_aggregate(
    constant int& brazil_nk [[buffer(0)]],
    device atomic_float* d_result_bins [[buffer(1)]],
    device int* d_supp_nation_map [[buffer(2)]],
    device int* d_orders_year_map [[buffer(3)]],
    device const uint* d_part_bitmap [[buffer(4)]],
    device const int* l_orderkey [[buffer(5)]],
    device const int* l_partkey [[buffer(6)]],
    device const int* l_suppkey [[buffer(7)]],
    device const float* l_extendedprice [[buffer(8)]],
    device const float* l_discount [[buffer(9)]],
    constant uint& n_lineitem [[buffer(10)]],
    uint tid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    for (uint i = tid; i < n_lineitem; i += tpg) {
        if (bitmap_test(d_part_bitmap, l_partkey[i])) {
            int _year = d_orders_year_map[l_orderkey[i]];
            if (_year != -1) {
                int _supp_nk = d_supp_nation_map[l_suppkey[i]];
                if (_supp_nk != -1) {
                    atomic_fetch_add_explicit(&d_result_bins[2 + (_year - 1995)], (float)(l_extendedprice[i] * (1.0f - l_discount[i])), memory_order_relaxed);
                    if (_supp_nk == brazil_nk) {
                        atomic_fetch_add_explicit(&d_result_bins[_year - 1995], (float)(l_extendedprice[i] * (1.0f - l_discount[i])), memory_order_relaxed);
                    }
                }
            }
        }
    }
}
