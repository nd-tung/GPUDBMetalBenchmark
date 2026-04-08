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

static bool brand_eq(const device char* brand, uint idx, char d1, char d2) {
    const device char* b = brand + idx * 10;
    return b[0]=='B' && b[1]=='r' && b[2]=='a' && b[3]=='n' && b[4]=='d' && b[5]=='#' && b[6]==d1 && b[7]==d2;
}
static int container_match(const device char* cont, uint idx) {
    const device char* c = cont + idx * 10;
    // SM CASE/BOX/PACK/PKG -> 1
    if (c[0]=='S' && c[1]=='M' && c[2]==' ') {
        char c3=c[3],c4=c[4],c5=c[5];
        if ((c3=='C'&&c4=='A'&&c5=='S') || (c3=='B'&&c4=='O') ||
            (c3=='P'&&c4=='A'&&c5=='C') || (c3=='P'&&c4=='K')) return 1;
    }
    // MED BAG/BOX/PKG/PACK -> 2
    if (c[0]=='M' && c[1]=='E' && c[2]=='D' && c[3]==' ') {
        char c4=c[4],c5=c[5],c6=c[6];
        if ((c4=='B'&&c5=='A'&&c6=='G') || (c4=='B'&&c5=='O') ||
            (c4=='P'&&c5=='K') || (c4=='P'&&c5=='A'&&c6=='C')) return 2;
    }
    // LG CASE/BOX/PACK/PKG -> 3
    if (c[0]=='L' && c[1]=='G' && c[2]==' ') {
        char c3=c[3],c4=c[4],c5=c[5];
        if ((c3=='C'&&c4=='A'&&c5=='S') || (c3=='B'&&c4=='O') ||
            (c3=='P'&&c4=='A'&&c5=='C') || (c3=='P'&&c4=='K')) return 3;
    }
    return 0;
}



// === Phase 0: Q19_build_part_cond ===
kernel void Q19_build_part_cond(
    device int* d_part_cond [[buffer(0)]],
    device const int* p_partkey [[buffer(1)]],
    device const char* p_brand [[buffer(2)]],
    device const char* p_container [[buffer(3)]],
    device const int* p_size [[buffer(4)]],
    constant uint& n_part [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    for (uint i = tid; i < n_part; i += tpg) {
        int _cond = (brand_eq(p_brand, i, '1', '2') && container_match(p_container, i) == 1 && p_size[i] >= 1 && p_size[i] <= 5 ? 1 : 0) | (brand_eq(p_brand, i, '2', '3') && container_match(p_container, i) == 2 && p_size[i] >= 1 && p_size[i] <= 10 ? 2 : 0) | (brand_eq(p_brand, i, '3', '4') && container_match(p_container, i) == 3 && p_size[i] >= 1 && p_size[i] <= 15 ? 4 : 0);
        if (_cond > 0) {
            d_part_cond[p_partkey[i]] = _cond;
        }
    }
}

// === Phase 1: Q19_reduce ===
kernel void Q19_reduce(
    device atomic_uint* d_q19_revenue_lo [[buffer(0)]],
    device atomic_uint* d_q19_revenue_hi [[buffer(1)]],
    device int* d_part_cond [[buffer(2)]],
    device const int* l_partkey [[buffer(3)]],
    device const float* l_quantity [[buffer(4)]],
    device const float* l_extendedprice [[buffer(5)]],
    device const float* l_discount [[buffer(6)]],
    device const char* l_shipmode [[buffer(7)]],
    device const char* l_shipinstruct [[buffer(8)]],
    constant uint& n_lineitem [[buffer(9)]],
    uint tid [[thread_position_in_grid]],
    uint tpg [[threads_per_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    long local_revenue = 0;
    for (uint i = tid; i < n_lineitem; i += tpg) {
        if ((l_shipmode[i * 10] == 'A' || (l_shipmode[i * 10] == 'R' && l_shipmode[i * 10 + 1] == 'E')) && l_shipinstruct[i * 25] == 'D') {
            int _cond = d_part_cond[l_partkey[i]];
            if (_cond != -1) {
                if (((_cond & 1) && l_quantity[i] >= 1.0f && l_quantity[i] <= 11.0f) || ((_cond & 2) && l_quantity[i] >= 10.0f && l_quantity[i] <= 20.0f) || ((_cond & 4) && l_quantity[i] >= 20.0f && l_quantity[i] <= 30.0f)) {
                    local_revenue += (long)((long)(l_extendedprice[i] * (1.0f - l_discount[i]) * 100.0f));
                }
            }
        }
    }
    // --- Threadgroup reduction ---
    threadgroup long tg_shared_revenue[32];
    long tg_revenue = tg_reduce_long(local_revenue, lid, tg_size, tg_shared_revenue);
    if (lid == 0) {
        atomic_add_long_pair(d_q19_revenue_lo, d_q19_revenue_hi, tg_revenue);
    }
}
