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

static void q21_track_supplier(device atomic_int* first_supp,
                                device atomic_uint* multi_supp_bmp,
                                device atomic_int* first_late,
                                device atomic_uint* multi_late_bmp,
                                int ok, int sk, bool is_late) {
    // Track multi-supplier orders
    int expected = -1;
    bool was_first = atomic_compare_exchange_weak_explicit(
        &first_supp[ok], &expected, sk, memory_order_relaxed, memory_order_relaxed);
    if (!was_first && expected != sk) {
        atomic_fetch_or_explicit(&multi_supp_bmp[ok >> 5], 1u << (ok & 31), memory_order_relaxed);
    }
    // Track multi-late orders
    if (is_late) {
        expected = -1;
        was_first = atomic_compare_exchange_weak_explicit(
            &first_late[ok], &expected, sk, memory_order_relaxed, memory_order_relaxed);
        if (!was_first && expected != sk) {
            atomic_fetch_or_explicit(&multi_late_bmp[ok >> 5], 1u << (ok & 31), memory_order_relaxed);
        }
    }
}



// === Phase 0: Q21_build_f_orders ===
kernel void Q21_build_f_orders(
    device atomic_uint* d_q21_f_orders [[buffer(0)]],
    device const int* o_orderkey [[buffer(1)]],
    device const char* o_orderstatus [[buffer(2)]],
    constant uint& n_orders [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    for (uint i = tid; i < n_orders; i += tpg) {
        if (o_orderstatus[i] == 'F') {
            bitmap_set(d_q21_f_orders, o_orderkey[i]);
        }
    }
}

// === Phase 1: Q21_build_bitmaps ===
kernel void Q21_build_bitmaps(
    device atomic_int* d_q21_first_supp [[buffer(0)]],
    device atomic_int* d_q21_first_late [[buffer(1)]],
    device atomic_uint* d_q21_multi_supp [[buffer(2)]],
    device atomic_uint* d_q21_multi_late [[buffer(3)]],
    device const uint* d_q21_f_orders [[buffer(4)]],
    device const int* l_orderkey [[buffer(5)]],
    device const int* l_suppkey [[buffer(6)]],
    device const int* l_receiptdate [[buffer(7)]],
    device const int* l_commitdate [[buffer(8)]],
    constant uint& n_lineitem [[buffer(9)]],
    uint tid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    for (uint i = tid; i < n_lineitem; i += tpg) {
        if (bitmap_test(d_q21_f_orders, l_orderkey[i])) {
            int _unused = (q21_track_supplier(d_q21_first_supp, d_q21_multi_supp, d_q21_first_late, d_q21_multi_late, l_orderkey[i], l_suppkey[i], l_receiptdate[i] > l_commitdate[i]), 0);
        }
    }
}

// === Phase 2: Q21_count_qualifying ===
kernel void Q21_count_qualifying(
    device atomic_uint* d_q21_supp_count [[buffer(0)]],
    device const uint* d_q21_multi_late [[buffer(1)]],
    device const uint* d_q21_multi_supp [[buffer(2)]],
    device const uint* d_q21_sa_supp [[buffer(3)]],
    device const uint* d_q21_f_orders [[buffer(4)]],
    device const int* l_orderkey [[buffer(5)]],
    device const int* l_suppkey [[buffer(6)]],
    device const int* l_receiptdate [[buffer(7)]],
    device const int* l_commitdate [[buffer(8)]],
    constant uint& n_lineitem [[buffer(9)]],
    uint tid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    for (uint i = tid; i < n_lineitem; i += tpg) {
        if (bitmap_test(d_q21_f_orders, l_orderkey[i])) {
            if (bitmap_test(d_q21_sa_supp, l_suppkey[i])) {
                if (l_receiptdate[i] > l_commitdate[i]) {
                    if (bitmap_test(d_q21_multi_supp, l_orderkey[i])) {
                        if (!bitmap_test(d_q21_multi_late, l_orderkey[i])) {
                            atomic_fetch_add_explicit(&d_q21_supp_count[l_suppkey[i]], 1u, memory_order_relaxed);
                        }
                    }
                }
            }
        }
    }
}
