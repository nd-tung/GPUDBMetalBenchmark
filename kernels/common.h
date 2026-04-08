#ifndef COMMON_METAL_H
#define COMMON_METAL_H

#include <metal_stdlib>
using namespace metal;

// ===================================================================
// SIMD GROUP REDUCTION HELPERS
// ===================================================================
// Apple GPUs have 32-wide SIMD groups. Two-level reduction:
//   Level 1: simd_sum within each SIMD group (no barrier, no shared memory)
//   Level 2: simd_sum across SIMD-group partials in shared memory (1 barrier)
// Result: 2 barriers instead of log2(N) ≈ 10 for a 1024-thread group.

// --- long (int64) SIMD reduction via 2×uint shuffle ---
inline long simd_reduce_add_long(long v) {
    for (uint d = 16; d >= 1; d >>= 1) {
        uint lo = simd_shuffle_down((uint)(v), d);
        uint hi = simd_shuffle_down((uint)((ulong)v >> 32), d);
        v += (long)(((ulong)hi << 32) | (ulong)lo);
    }
    return v;
}

// --- Threadgroup-level reductions (threads_per_group <= 1024 → max 32 SIMD groups) ---
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

// ===================================================================
// SHARED HASH UTILITIES
// ===================================================================

inline uint sf100_hash_key(int key) {
    uint h = as_type<uint>(key);
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;
    return h;
}

// Represents an entry in our simple hash table (used by microbenchmarks and Q9)
struct HashTableEntry {
    atomic_int key;   // o_orderkey
    atomic_int value; // Row ID (payload)
};

// Non-atomic read-only aliases for probe-phase reads (enables L2 caching on Apple GPUs)
struct HashTableEntryRO {
    int key;
    int value;
};

struct PartSuppEntryRO {
    int partkey;
    int suppkey;
    int idx;
    int _pad;
};

// ===================================================================
// DATE RANGE HELPERS (YYYYMMDD integer dates)
// ===================================================================

// Half-open: date in [lo, hi)  — standard for year ranges
inline bool date_in_range_ho(int date, int lo, int hi) { return date >= lo && date < hi; }

// Closed:   date in [lo, hi]  — for inclusive end dates
inline bool date_in_range(int date, int lo, int hi) { return date >= lo && date <= hi; }

// ===================================================================
// BITMAP HELPERS
// ===================================================================

inline bool bitmap_test(const device uint* bitmap, int key) {
    return (bitmap[(uint)key >> 5] >> ((uint)key & 31u)) & 1u;
}

inline void bitmap_set(device atomic_uint* bitmap, int key) {
    atomic_fetch_or_explicit(&bitmap[(uint)key >> 5], 1u << ((uint)key & 31u), memory_order_relaxed);
}

// ===================================================================
// 64-BIT ATOMIC ADD EMULATION (Metal lacks native atomic_long)
// ===================================================================
// Uses a pair of device atomic_uint (lo, hi) to represent a 64-bit value.
// CAS loop on the low word with carry propagation to the high word.
inline void atomic_add_long_pair(device atomic_uint* lo, device atomic_uint* hi, long val) {
    ulong uval = as_type<ulong>(val);
    uint add_lo = (uint)(uval);
    uint add_hi = (uint)(uval >> 32);
    uint old_lo = atomic_fetch_add_explicit(lo, add_lo, memory_order_relaxed);
    // Detect carry: if old_lo + add_lo wrapped (overflow)
    uint new_lo = old_lo + add_lo;
    uint carry = (new_lo < old_lo) ? 1u : 0u;
    if (add_hi != 0 || carry != 0) {
        atomic_fetch_add_explicit(hi, add_hi + carry, memory_order_relaxed);
    }
}

inline long load_long_pair(const device uint* lo, const device uint* hi) {
    ulong v = ((ulong)(*hi) << 32) | (ulong)(*lo);
    return as_type<long>(v);
}

// ===================================================================
// HASH TABLE PROBE HELPERS — always use power-of-2 capacity + mask
// ===================================================================

// Round up to next power of 2 (compile-time or runtime)
inline uint next_pow2(uint v) {
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}

// Probe a simple (key, value) hash table. Returns value or -1 if not found.
inline int ht_probe_kv(const device HashTableEntry* ht, int key, uint mask) {
    uint h = sf100_hash_key(key);
    for (uint i = 0; i <= mask; i++) {
        uint slot = (h + i) & mask;
        int k = atomic_load_explicit(&ht[slot].key, memory_order_relaxed);
        if (k == key) return atomic_load_explicit(&ht[slot].value, memory_order_relaxed);
        if (k == -1) return -1;
    }
    return -1;
}

#endif // COMMON_METAL_H
