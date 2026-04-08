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

// --- Helper functions ---

static bool q2_type_ends_brass(const device char* p_type, uint idx) {
    const device char* tp = p_type + (uint)idx * 25u;
    int len = 25;
    while (len > 0 && (tp[len-1] == ' ' || tp[len-1] == '\0')) len--;
    return len >= 5 && tp[len-5]=='B' && tp[len-4]=='R' &&
           tp[len-3]=='A' && tp[len-2]=='S' && tp[len-1]=='S';
}


static void q2_atomic_min(device atomic_uint* min_cost, uint partkey, float cost) {
    uint cost_uint = as_type<uint>(cost);
    atomic_fetch_min_explicit(&min_cost[partkey], cost_uint, memory_order_relaxed);
}



// === Phase 0: Q2_build_part_bitmap ===
kernel void Q2_build_part_bitmap(
    device atomic_uint* d_q2_part_bitmap [[buffer(0)]],
    device const int* p_partkey [[buffer(1)]],
    device const int* p_size [[buffer(2)]],
    device const char* p_type [[buffer(3)]],
    constant uint& n_part [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    for (uint i = tid; i < n_part; i += tpg) {
        if (p_size[i] == 15) {
            if (q2_type_ends_brass(p_type, i)) {
                bitmap_set(d_q2_part_bitmap, p_partkey[i]);
            }
        }
    }
}

// === Phase 1: Q2_find_min_cost ===
kernel void Q2_find_min_cost(
    device atomic_uint* d_q2_min_cost [[buffer(0)]],
    device const uint* d_q2_supp_bitmap [[buffer(1)]],
    device const uint* d_q2_part_bitmap [[buffer(2)]],
    device const int* ps_partkey [[buffer(3)]],
    device const int* ps_suppkey [[buffer(4)]],
    device const float* ps_supplycost [[buffer(5)]],
    constant uint& n_partsupp [[buffer(6)]],
    uint tid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    for (uint i = tid; i < n_partsupp; i += tpg) {
        if (bitmap_test(d_q2_part_bitmap, ps_partkey[i])) {
            if (bitmap_test(d_q2_supp_bitmap, ps_suppkey[i])) {
                int _unused = (q2_atomic_min(d_q2_min_cost, (uint)ps_partkey[i], ps_supplycost[i]), 0);
            }
        }
    }
}
