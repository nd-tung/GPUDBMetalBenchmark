#include "common.h"

// ===================================================================
// TPC-H Q13 KERNELS — Customer Distribution
// ===================================================================
/*
SELECT
    c_count,
    COUNT(*) AS custdist
FROM (
    SELECT
        c_custkey,
        COUNT(o_orderkey) AS c_count
    FROM
        customer
    LEFT OUTER JOIN
        orders ON c_custkey = o_custkey
        AND o_comment NOT LIKE '%special%requests%'
    GROUP BY
        c_custkey
) AS c_orders
GROUP BY
    c_count
ORDER BY
    custdist DESC,
    c_count DESC;
*/



// --- Q13 substring matching helpers (SWAR — SIMD Within A Register) ---
// Process 4 bytes per iteration using classic Mycroft null-byte / byte-match
// detection on packed uint words. Each o_comment record is 100 bytes (25 uints)
// and starts at a 4-byte aligned offset (100 % 4 == 0).

// Returns non-zero with the high bit set in each zero byte
inline uint swar_has_zero(uint v) {
    return (v - 0x01010101u) & ~v & 0x80808080u;
}

// Returns non-zero with the high bit set in each byte equal to c (broadcast)
inline uint swar_has_byte(uint v, uint c_bcast) {
    uint x = v ^ c_bcast;
    return (x - 0x01010101u) & ~x & 0x80808080u;
}

// SWAR: Find first null byte in 100-byte fixed-width field.
// Returns effective length, or -1 if too short for pattern matching (< 15).
// Processes 4 bytes per iteration (25 iterations for 100 bytes vs 100 scalar).
inline int q13_effective_len_swar(const device uint* words) {
    const int min_pattern_len = 15;
    for (int i = 0; i < 25; i++) {  // 25 * 4 = 100 bytes
        uint z = swar_has_zero(words[i]);
        if (z) {
            int pos = i * 4 + (ctz(z) >> 3);
            return (pos < min_pattern_len) ? -1 : pos;
        }
    }
    return 100;
}

// SWAR: Search for "%special%requests%" in comment string.
// Scans for discriminant characters ('c' in "special", 'q' in "requests")
// 4 bytes at a time, then verifies full pattern only at candidate positions.
inline bool q13_has_special_requests_swar(const device uchar* s,
                                          const device uint* words,
                                          int comment_len) {
    const int last_special = comment_len - 15;
    if (last_special < 0) return false;

    // SWAR scan for 'c' byte (offset 3 in "special") — 4 positions per iteration
    const uint c_bcast = 0x63636363u; // 'c' = 0x63

    // Words covering byte positions [0 .. last_special + 3]
    int scan_words = (last_special + 3) / 4 + 1;
    if (scan_words > 25) scan_words = 25;

    for (int wi = 0; wi < scan_words; wi++) {
        uint m = swar_has_byte(words[wi], c_bcast);
        if (!m) continue;

        // Iterate 'c' candidates in this word (up to 4)
        while (m) {
            int c_pos = wi * 4 + (ctz(m) >> 3);
            m &= m - 1;                     // clear lowest set hit
            int i = c_pos - 3;              // potential start of "special"
            if (i < 0 || i > last_special) continue;

            // Verify full "special" at position i
            if (s[i]   == 's' && s[i+1] == 'p' && s[i+2] == 'e' &&
                s[i+4] == 'i' && s[i+5] == 'a' && s[i+6] == 'l') {

                // "special" confirmed — SWAR scan for "requests" after it
                int req_start = i + 7;
                int req_end = comment_len - 8;
                const uint q_bcast = 0x71717171u; // 'q' = 0x71

                // Words covering byte positions [req_start+2 .. req_end+2]
                int rw_lo = (req_start + 2) / 4;
                int rw_hi = (req_end + 2) / 4;
                if (rw_hi >= 25) rw_hi = 24;

                for (int rw = rw_lo; rw <= rw_hi; rw++) {
                    uint rm = swar_has_byte(words[rw], q_bcast);
                    if (!rm) continue;

                    while (rm) {
                        int q_pos = rw * 4 + (ctz(rm) >> 3);
                        rm &= rm - 1;
                        int j = q_pos - 2;  // potential start of "requests"
                        if (j < req_start || j > req_end) continue;

                        if (s[j]   == 'r' && s[j+1] == 'e' &&
                            s[j+3] == 'u' && s[j+4] == 'e' &&
                            s[j+5] == 's' && s[j+6] == 't' && s[j+7] == 's') {
                            return true;
                        }
                    }
                }
                return false; // "special" found but no "requests" after it
            }
        }
    }
    return false;
}


kernel void q13_fused_direct_count_kernel(
    const device int* o_custkey,
    const device char* o_comment,
    device atomic_uint* customer_order_counts,
    constant uint& orders_size,
    constant uint& customer_size,
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    const int comment_len = 100;
    const uint global_tid = (group_id * threads_per_group) + thread_id_in_group;
    const uint BATCH = 4;
    const uint stride = grid_size * BATCH;

    for (uint base = global_tid * BATCH; base < orders_size; base += stride) {
        if (base + 0 < orders_size) {
            const uint i = base + 0;
            const uint ck = (uint)o_custkey[i];
            if (ck >= 1u && ck <= customer_size) {
                const device uchar* row = (const device uchar*)(o_comment + (i * comment_len));
                const device uint* words = (const device uint*)(o_comment + (i * comment_len));
                int effective_len = q13_effective_len_swar(words);
                if (effective_len > 0) {
                    bool skip = q13_has_special_requests_swar(row, words, effective_len);
                    if (!skip) {
                        atomic_fetch_add_explicit(&customer_order_counts[ck - 1u], 1u, memory_order_relaxed);
                    }
                } else {
                    atomic_fetch_add_explicit(&customer_order_counts[ck - 1u], 1u, memory_order_relaxed);
                }
            }
        }
        if (base + 1 < orders_size) {
            const uint i = base + 1;
            const uint ck = (uint)o_custkey[i];
            if (ck >= 1u && ck <= customer_size) {
                const device uchar* row = (const device uchar*)(o_comment + (i * comment_len));
                const device uint* words = (const device uint*)(o_comment + (i * comment_len));
                int effective_len = q13_effective_len_swar(words);
                if (effective_len > 0) {
                    bool skip = q13_has_special_requests_swar(row, words, effective_len);
                    if (!skip) {
                        atomic_fetch_add_explicit(&customer_order_counts[ck - 1u], 1u, memory_order_relaxed);
                    }
                } else {
                    atomic_fetch_add_explicit(&customer_order_counts[ck - 1u], 1u, memory_order_relaxed);
                }
            }
        }
        if (base + 2 < orders_size) {
            const uint i = base + 2;
            const uint ck = (uint)o_custkey[i];
            if (ck >= 1u && ck <= customer_size) {
                const device uchar* row = (const device uchar*)(o_comment + (i * comment_len));
                const device uint* words = (const device uint*)(o_comment + (i * comment_len));
                int effective_len = q13_effective_len_swar(words);
                if (effective_len > 0) {
                    bool skip = q13_has_special_requests_swar(row, words, effective_len);
                    if (!skip) {
                        atomic_fetch_add_explicit(&customer_order_counts[ck - 1u], 1u, memory_order_relaxed);
                    }
                } else {
                    atomic_fetch_add_explicit(&customer_order_counts[ck - 1u], 1u, memory_order_relaxed);
                }
            }
        }
        if (base + 3 < orders_size) {
            const uint i = base + 3;
            const uint ck = (uint)o_custkey[i];
            if (ck >= 1u && ck <= customer_size) {
                const device uchar* row = (const device uchar*)(o_comment + (i * comment_len));
                const device uint* words = (const device uint*)(o_comment + (i * comment_len));
                int effective_len = q13_effective_len_swar(words);
                if (effective_len > 0) {
                    bool skip = q13_has_special_requests_swar(row, words, effective_len);
                    if (!skip) {
                        atomic_fetch_add_explicit(&customer_order_counts[ck - 1u], 1u, memory_order_relaxed);
                    }
                } else {
                    atomic_fetch_add_explicit(&customer_order_counts[ck - 1u], 1u, memory_order_relaxed);
                }
            }
        }
    }
}


// =============================================================
// Q13 GPU Histogram Kernel
// Scans per-customer order counts and builds histogram on GPU.
// Eliminates CPU scan of all customers.
// =============================================================
kernel void q13_build_histogram_kernel(
    const device uint* customer_order_counts [[buffer(0)]],
    device atomic_uint* histogram            [[buffer(1)]],
    constant uint& customer_size             [[buffer(2)]],
    constant uint& max_bins                  [[buffer(3)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    uint global_id = (group_id * threads_per_group) + thread_id_in_group;

    for (uint i = global_id; i < customer_size; i += grid_size) {
        uint count = customer_order_counts[i];
        if (count < max_bins) {
            atomic_fetch_add_explicit(&histogram[count], 1u, memory_order_relaxed);
        }
    }
}
