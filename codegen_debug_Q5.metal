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


// === Phase 0: Q5_build_nation_bitmap ===
kernel void Q5_build_nation_bitmap(
    constant int& asia_rk [[buffer(0)]],
    device atomic_uint* d_nation_bitmap [[buffer(1)]],
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
        if (n_regionkey[i] == asia_rk) {
            bitmap_set(d_nation_bitmap, n_nationkey[i]);
        }
    }
}

// === Phase 1: Q5_build_cust_map ===
kernel void Q5_build_cust_map(
    device int* d_cust_nation_map [[buffer(0)]],
    device const uint* d_nation_bitmap [[buffer(1)]],
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
        if (bitmap_test(d_nation_bitmap, c_nationkey[i])) {
            d_cust_nation_map[c_custkey[i]] = c_nationkey[i];
        }
    }
}

// === Phase 2: Q5_build_supp_map ===
kernel void Q5_build_supp_map(
    device int* d_supp_nation_map [[buffer(0)]],
    device const uint* d_nation_bitmap [[buffer(1)]],
    device const int* s_suppkey [[buffer(2)]],
    device const int* s_nationkey [[buffer(3)]],
    constant uint& n_supplier [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    for (uint i = tid; i < n_supplier; i += tpg) {
        if (bitmap_test(d_nation_bitmap, s_nationkey[i])) {
            d_supp_nation_map[s_suppkey[i]] = s_nationkey[i];
        }
    }
}

// === Phase 3: Q5_build_orders_map ===
kernel void Q5_build_orders_map(
    device int* d_orders_nation_map [[buffer(0)]],
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
        if (o_orderdate[i] >= 19940101 && o_orderdate[i] < 19950101) {
            int _cust_nk = d_cust_nation_map[o_custkey[i]];
            if (_cust_nk != -1) {
                d_orders_nation_map[o_orderkey[i]] = _cust_nk;
            }
        }
    }
}

// === Phase 4: Q5_probe_aggregate ===
kernel void Q5_probe_aggregate(
    device atomic_float* d_nation_revenue [[buffer(0)]],
    device int* d_supp_nation_map [[buffer(1)]],
    device int* d_orders_nation_map [[buffer(2)]],
    device const int* l_orderkey [[buffer(3)]],
    device const int* l_suppkey [[buffer(4)]],
    device const float* l_extendedprice [[buffer(5)]],
    device const float* l_discount [[buffer(6)]],
    constant uint& n_lineitem [[buffer(7)]],
    uint tid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    for (uint i = tid; i < n_lineitem; i += tpg) {
        int _cust_nk = d_orders_nation_map[l_orderkey[i]];
        if (_cust_nk != -1) {
            int _supp_nk = d_supp_nation_map[l_suppkey[i]];
            if (_supp_nk != -1) {
                if (_cust_nk == _supp_nk) {
                    atomic_fetch_add_explicit(&d_nation_revenue[_cust_nk], (float)(l_extendedprice[i] * (1.0f - l_discount[i])), memory_order_relaxed);
                }
            }
        }
    }
}
