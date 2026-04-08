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


// === Phase 0: Q14_build_bitmap ===
kernel void Q14_build_bitmap(
    device atomic_uint* d_promo_bitmap [[buffer(0)]],
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
        if (p_type[i * 25] == 'P' && p_type[i * 25 + 1] == 'R' && p_type[i * 25 + 2] == 'O' && p_type[i * 25 + 3] == 'M' && p_type[i * 25 + 4] == 'O') {
            bitmap_set(d_promo_bitmap, p_partkey[i]);
        }
    }
}

// === Phase 1: Q14_reduce ===
kernel void Q14_reduce(
    device const uint* d_promo_bitmap [[buffer(0)]],
    device atomic_uint* d_q14_promo_lo [[buffer(1)]],
    device atomic_uint* d_q14_promo_hi [[buffer(2)]],
    device atomic_uint* d_q14_total_lo [[buffer(3)]],
    device atomic_uint* d_q14_total_hi [[buffer(4)]],
    device const int* l_partkey [[buffer(5)]],
    device const int* l_shipdate [[buffer(6)]],
    device const float* l_extendedprice [[buffer(7)]],
    device const float* l_discount [[buffer(8)]],
    constant uint& n_lineitem [[buffer(9)]],
    uint tid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    long local_promo = 0;
    long local_total = 0;
    for (uint i = tid; i < n_lineitem; i += tpg) {
        if ((l_shipdate[i] >= 19950901) && (l_shipdate[i] < 19951001)) {
            local_promo += (long)((long)(bitmap_test(d_promo_bitmap, l_partkey[i]) ? l_extendedprice[i] * (1.0f - l_discount[i] * 0.01f) * 100.0f : 0));
            local_total += (long)((long)(l_extendedprice[i] * (1.0f - l_discount[i] * 0.01f) * 100.0f));
        }
    }
    // --- Threadgroup reduction ---
    threadgroup long tg_shared_promo[32];
    long tg_promo = tg_reduce_long(local_promo, lid, tg_size, tg_shared_promo);
    if (lid == 0) {
        atomic_add_long_pair(d_q14_promo_lo, d_q14_promo_hi, tg_promo);
    }
    threadgroup long tg_shared_total[32];
    long tg_total = tg_reduce_long(local_total, lid, tg_size, tg_shared_total);
    if (lid == 0) {
        atomic_add_long_pair(d_q14_total_lo, d_q14_total_hi, tg_total);
    }
}
