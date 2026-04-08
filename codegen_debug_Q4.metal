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


// === Phase 0: Q4_build_bitmap ===
kernel void Q4_build_bitmap(
    device atomic_uint* d_late_bitmap [[buffer(0)]],
    device const int* l_orderkey [[buffer(1)]],
    device const int* l_commitdate [[buffer(2)]],
    device const int* l_receiptdate [[buffer(3)]],
    constant uint& n_lineitem [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    for (uint i = tid; i < n_lineitem; i += tpg) {
        if (l_commitdate[i] < l_receiptdate[i]) {
            bitmap_set(d_late_bitmap, l_orderkey[i]);
        }
    }
}

// === Phase 1: Q4_count ===
kernel void Q4_count(
    device atomic_uint* d_q4_counts [[buffer(0)]],
    device const uint* d_late_bitmap [[buffer(1)]],
    device const int* o_orderkey [[buffer(2)]],
    device const int* o_orderdate [[buffer(3)]],
    device const char* o_orderpriority [[buffer(4)]],
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
        if ((o_orderdate[i] >= 19930701) && (o_orderdate[i] < 19931001)) {
            if (bitmap_test(d_late_bitmap, o_orderkey[i])) {
                int _bucket = (o_orderpriority[i * 15] - '1');
                atomic_fetch_add_explicit(&d_q4_counts[_bucket * 1 + 0], (uint)(1u), memory_order_relaxed);
            }
        }
    }
}
