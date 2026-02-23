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
// SF100 CHUNKED EXECUTION KERNELS
// ===================================================================

// --- Q1 Chunked: processes a chunk of lineitem rows, accumulating into
//     per-threadgroup partial buffers (same two-pass integer-cent approach) ---
kernel void q1_chunked_stage1(
    const device int*   l_shipdate        [[buffer(0)]],
    const device char*  l_returnflag      [[buffer(1)]],
    const device char*  l_linestatus      [[buffer(2)]],
    const device float* l_quantity        [[buffer(3)]],
    const device float* l_extendedprice   [[buffer(4)]],
    const device float* l_discount        [[buffer(5)]],
    const device float* l_tax             [[buffer(6)]],
    device long*  p_sum_qty_cents         [[buffer(7)]],
    device long*  p_sum_base_cents        [[buffer(8)]],
    device long*  p_sum_disc_price_cents  [[buffer(9)]],
    device long*  p_sum_charge_cents      [[buffer(10)]],
    device uint*  p_sum_discount_bp       [[buffer(11)]],
    device uint*  p_count                 [[buffer(12)]],
    constant uint& chunk_size             [[buffer(13)]],
    constant int&  cutoff_date            [[buffer(14)]],
    constant uint& num_threadgroups       [[buffer(15)]],
    uint group_id         [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group  [[threads_per_threadgroup]],
    uint grid_size          [[threads_per_grid]])
{
    const int BINS = 6;
    long sum_qty_c[BINS], sum_base_c[BINS], sum_disc_c[BINS], sum_charge_c[BINS];
    uint sum_disc_bp[BINS], cnt[BINS];
    for (int b = 0; b < BINS; ++b) {
        sum_qty_c[b]=0; sum_base_c[b]=0; sum_disc_c[b]=0; sum_charge_c[b]=0;
        sum_disc_bp[b]=0u; cnt[b]=0u;
    }

    auto rf_index = [](char rf) -> int { return (rf=='A')?0:(rf=='N')?1:(rf=='R')?2:-1; };
    auto ls_index = [](char ls) -> int { return (ls=='F')?0:(ls=='O')?1:-1; };

    for (uint i = (group_id * threads_per_group) + thread_id_in_group;
         i < chunk_size; i += grid_size) {
        if (l_shipdate[i] > cutoff_date) continue;
        int rfi = rf_index(l_returnflag[i]); if (rfi < 0) continue;
        int lsi = ls_index(l_linestatus[i]); if (lsi < 0) continue;
        int bin = rfi * 2 + lsi;
        float base = l_extendedprice[i], qty = l_quantity[i], d = l_discount[i], t = l_tax[i];
        long base_c = (long)floor(base * 100.0f + 0.5f);
        long qty_c  = (long)floor(qty  * 100.0f + 0.5f);
        int  d_bp   = (int)floor(d * 100.0f + 0.5f);
        int  t_bp   = (int)floor(t * 100.0f + 0.5f);
        long disc_c   = (base_c * (long)(100 - d_bp) + 50) / 100;
        long charge_c = (disc_c * (long)(100 + t_bp) + 50) / 100;
        sum_qty_c[bin] += qty_c; sum_base_c[bin] += base_c;
        sum_disc_c[bin] += disc_c; sum_charge_c[bin] += charge_c;
        sum_disc_bp[bin] += (uint)d_bp; cnt[bin] += 1u;
    }

    // SIMD-accelerated threadgroup reduction (2 barriers per metric instead of ~10)
    threadgroup long tg64[32];  // one per SIMD group (max 1024/32 = 32)
    threadgroup uint tg32[32];

    for (int b = 0; b < BINS; ++b) {
        long r;
        r = tg_reduce_long(sum_qty_c[b], thread_id_in_group, threads_per_group, tg64);
        if (thread_id_in_group == 0) p_sum_qty_cents[group_id * BINS + b] = r;

        r = tg_reduce_long(sum_base_c[b], thread_id_in_group, threads_per_group, tg64);
        if (thread_id_in_group == 0) p_sum_base_cents[group_id * BINS + b] = r;

        r = tg_reduce_long(sum_disc_c[b], thread_id_in_group, threads_per_group, tg64);
        if (thread_id_in_group == 0) p_sum_disc_price_cents[group_id * BINS + b] = r;

        r = tg_reduce_long(sum_charge_c[b], thread_id_in_group, threads_per_group, tg64);
        if (thread_id_in_group == 0) p_sum_charge_cents[group_id * BINS + b] = r;

        uint u;
        u = tg_reduce_uint(sum_disc_bp[b], thread_id_in_group, threads_per_group, tg32);
        if (thread_id_in_group == 0) p_sum_discount_bp[group_id * BINS + b] = u;

        u = tg_reduce_uint(cnt[b], thread_id_in_group, threads_per_group, tg32);
        if (thread_id_in_group == 0) p_count[group_id * BINS + b] = u;
    }
}

// Q1 chunked stage2: identical to existing q1_bins_reduce_int_stage2
kernel void q1_chunked_stage2(
    const device long* p_sum_qty_cents,
    const device long* p_sum_base_cents,
    const device long* p_sum_disc_price_cents,
    const device long* p_sum_charge_cents,
    const device uint* p_sum_discount_bp,
    const device uint* p_count,
    device long* out_sum_qty_cents,
    device long* out_sum_base_cents,
    device long* out_sum_disc_price_cents,
    device long* out_sum_charge_cents,
    device uint* out_sum_discount_bp,
    device uint* out_count,
    constant uint& num_threadgroups,
    uint index [[thread_position_in_grid]])
{
    if (index != 0) return;
    const uint BINS = 6;
    for (uint b = 0; b < BINS; ++b) {
        long s_qty = 0, s_base = 0, s_disc = 0, s_charge = 0;
        uint s_dbp = 0, s_cnt = 0;
        for (uint g = 0; g < num_threadgroups; ++g) {
            uint idx = g * BINS + b;
            s_qty    += p_sum_qty_cents[idx];
            s_base   += p_sum_base_cents[idx];
            s_disc   += p_sum_disc_price_cents[idx];
            s_charge += p_sum_charge_cents[idx];
            s_dbp    += p_sum_discount_bp[idx];
            s_cnt    += p_count[idx];
        }
        out_sum_qty_cents[b]        = s_qty;
        out_sum_base_cents[b]       = s_base;
        out_sum_disc_price_cents[b] = s_disc;
        out_sum_charge_cents[b]     = s_charge;
        out_sum_discount_bp[b]      = s_dbp;
        out_count[b]                = s_cnt;
    }
}

// --- Q6 Chunked: processes a chunk of lineitem rows ---
kernel void q6_chunked_stage1(
    const device int*   l_shipdate        [[buffer(0)]],
    const device float* l_discount        [[buffer(1)]],
    const device float* l_quantity        [[buffer(2)]],
    const device float* l_extendedprice   [[buffer(3)]],
    device float* partial_revenues        [[buffer(4)]],
    constant uint& chunk_size             [[buffer(5)]],
    constant int&  start_date             [[buffer(6)]],
    constant int&  end_date               [[buffer(7)]],
    constant float& min_discount          [[buffer(8)]],
    constant float& max_discount          [[buffer(9)]],
    constant float& max_quantity          [[buffer(10)]],
    uint group_id           [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group  [[threads_per_threadgroup]],
    uint grid_size          [[threads_per_grid]])
{
    float local_revenue = 0.0f;
    for (uint i = (group_id * threads_per_group) + thread_id_in_group;
         i < chunk_size; i += grid_size) {
        if (l_shipdate[i] >= start_date && l_shipdate[i] < end_date &&
            l_discount[i] >= min_discount && l_discount[i] <= max_discount &&
            l_quantity[i] < max_quantity) {
            local_revenue += l_extendedprice[i] * l_discount[i];
        }
    }
    // SIMD-accelerated threadgroup reduction
    threadgroup float shared[32];
    float result = tg_reduce_float(local_revenue, thread_id_in_group, threads_per_group, shared);
    if (thread_id_in_group == 0) partial_revenues[group_id] = result;
}

// Q6 chunked stage2: reduce partials to a single value
kernel void q6_chunked_stage2(
    const device float* partial_revenues [[buffer(0)]],
    device float* final_revenue          [[buffer(1)]],
    constant uint& num_threadgroups      [[buffer(2)]],
    uint index [[thread_position_in_grid]])
{
    if (index != 0) return;
    float total = 0.0f;
    for (uint i = 0; i < num_threadgroups; ++i) total += partial_revenues[i];
    final_revenue[0] = total;
}

// ===================================================================
// SF100 RADIX PARTITIONED HASH JOIN KERNELS
// ===================================================================

constant uint SF100_RADIX_BITS = 10;
constant uint SF100_NUM_PARTITIONS = 1024; // 1 << 10

inline uint sf100_hash_key(int key) {
    uint h = as_type<uint>(key);
    h ^= h >> 16;
    h *= 0x45d9f3b;
    h ^= h >> 16;
    return h;
}

inline uint sf100_get_partition(int key) {
    return sf100_hash_key(key) & (SF100_NUM_PARTITIONS - 1);
}

// Phase 1: Build histogram — count rows per partition (with threadgroup-local histogram)
kernel void sf100_radix_histogram(
    device const int*  keys       [[buffer(0)]],
    device atomic_uint* histogram [[buffer(1)]],
    constant uint& num_rows       [[buffer(2)]],
    uint tid              [[thread_position_in_grid]],
    uint grid_size        [[threads_per_grid]],
    uint tg_tid           [[thread_index_in_threadgroup]],
    uint threads_per_tg   [[threads_per_threadgroup]])
{
    threadgroup uint local_hist[1024]; // SF100_NUM_PARTITIONS
    for (uint i = tg_tid; i < 1024; i += threads_per_tg) local_hist[i] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < num_rows; i += grid_size) {
        uint part = sf100_get_partition(keys[i]);
        atomic_fetch_add_explicit((threadgroup atomic_uint*)&local_hist[part], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tg_tid; i < 1024; i += threads_per_tg) {
        if (local_hist[i] > 0)
            atomic_fetch_add_explicit(&histogram[i], local_hist[i], memory_order_relaxed);
    }
}

// Phase 2: Exclusive prefix sum on histogram (single threadgroup for 1024 elements)
kernel void sf100_radix_prefix_sum(
    device uint* histogram       [[buffer(0)]],
    device uint* total_counts    [[buffer(1)]],
    constant uint& num_partitions [[buffer(2)]],
    uint tid         [[thread_index_in_threadgroup]],
    uint tg_size     [[threads_per_threadgroup]])
{
    threadgroup uint shared[1024];
    for (uint i = tid; i < num_partitions; i += tg_size) {
        shared[i] = histogram[i];
        total_counts[i] = histogram[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Blelloch exclusive scan — up-sweep
    for (uint stride = 1; stride < num_partitions; stride *= 2) {
        uint max_idx = num_partitions / (2 * stride);
        for (uint i = tid; i < max_idx; i += tg_size) {
            uint idx = (2 * stride) * (i + 1) - 1;
            shared[idx] += shared[idx - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) shared[num_partitions - 1] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down-sweep
    for (uint stride = num_partitions / 2; stride >= 1; stride /= 2) {
        uint max_idx = num_partitions / (2 * stride);
        for (uint i = tid; i < max_idx; i += tg_size) {
            uint idx = (2 * stride) * (i + 1) - 1;
            uint temp = shared[idx];
            shared[idx] += shared[idx - stride];
            shared[idx - stride] = temp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = tid; i < num_partitions; i += tg_size) histogram[i] = shared[i];
}

// Phase 3: Scatter keys and payloads into partitioned positions
kernel void sf100_radix_scatter(
    device const int*  keys_in     [[buffer(0)]],
    device const int*  payload_in  [[buffer(1)]],
    device int*        keys_out    [[buffer(2)]],
    device int*        payload_out [[buffer(3)]],
    device atomic_uint* offsets    [[buffer(4)]],
    constant uint& num_rows        [[buffer(5)]],
    uint tid        [[thread_position_in_grid]],
    uint grid_size  [[threads_per_grid]])
{
    for (uint i = tid; i < num_rows; i += grid_size) {
        int key = keys_in[i];
        uint part = sf100_get_partition(key);
        uint pos = atomic_fetch_add_explicit(&offsets[part], 1u, memory_order_relaxed);
        keys_out[pos] = key;
        payload_out[pos] = payload_in[i];
    }
}

// Phase 4: Per-partition local hash join (build + probe in threadgroup memory)
constant uint SF100_LOCAL_HT_CAP = 16384; // per-partition hash table capacity

kernel void sf100_partition_join(
    device const int* build_keys      [[buffer(0)]],
    device const int* build_payloads  [[buffer(1)]],
    constant uint& build_count        [[buffer(2)]],
    device const int* probe_keys      [[buffer(3)]],
    device const int* probe_payloads  [[buffer(4)]],
    constant uint& probe_count        [[buffer(5)]],
    device int*       out_build_idx   [[buffer(6)]],
    device int*       out_probe_idx   [[buffer(7)]],
    device atomic_uint* out_count     [[buffer(8)]],
    uint tid          [[thread_index_in_threadgroup]],
    uint tg_size      [[threads_per_threadgroup]])
{
    // Use threadgroup memory for hash table
    threadgroup int ht_keys[SF100_LOCAL_HT_CAP];
    threadgroup int ht_vals[SF100_LOCAL_HT_CAP];

    for (uint i = tid; i < SF100_LOCAL_HT_CAP; i += tg_size) ht_keys[i] = -1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Build phase
    for (uint i = tid; i < build_count; i += tg_size) {
        int key = build_keys[i];
        uint h = sf100_hash_key(key) & (SF100_LOCAL_HT_CAP - 1);
        for (uint j = 0; j < SF100_LOCAL_HT_CAP; ++j) {
            uint slot = (h + j) & (SF100_LOCAL_HT_CAP - 1);
            int expected = -1;
            if (atomic_compare_exchange_weak_explicit(
                    (threadgroup atomic_int*)&ht_keys[slot], &expected, key,
                    memory_order_relaxed, memory_order_relaxed)) {
                ht_vals[slot] = (int)i;
                break;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Probe phase
    for (uint i = tid; i < probe_count; i += tg_size) {
        int key = probe_keys[i];
        uint h = sf100_hash_key(key) & (SF100_LOCAL_HT_CAP - 1);
        for (uint j = 0; j < SF100_LOCAL_HT_CAP; ++j) {
            uint slot = (h + j) & (SF100_LOCAL_HT_CAP - 1);
            int sk = ht_keys[slot];
            if (sk == -1) break;
            if (sk == key) {
                uint out_idx = atomic_fetch_add_explicit(out_count, 1u, memory_order_relaxed);
                out_build_idx[out_idx] = ht_vals[slot];
                out_probe_idx[out_idx] = (int)i;
            }
        }
    }
}

// ===================================================================
// END SF100 KERNELS
// ===================================================================

// --- SELECTION KERNELS ---
// SELECT * FROM lineitem WHERE column < filterValue;
kernel void selection_kernel(const device int  *inData,        // Input data column
                           device uint *result,       // Output bitmap (0 or 1)
                           constant int &filterValue, // The value to compare against
                           uint index [[thread_position_in_grid]]) {

    // Each thread performs one comparison
    if (inData[index] < filterValue) {
        result[index] = 1;
    } else {
        result[index] = 0;
    }
}


// --- AGGREGATION KERNELS ---
// SELECT SUM(l_quantity) FROM lineitem;

// Stage 1: Reduces a partition of the input data into a single partial sum per threadgroup.
kernel void sum_kernel_stage1(const device float* inData,
                              device float* partialSums,
                              constant uint& dataSize, 
                              uint group_id [[threadgroup_position_in_grid]],
                              uint thread_id_in_group [[thread_index_in_threadgroup]],
                              uint threads_per_group [[threads_per_threadgroup]],
                              uint grid_size [[threads_per_grid]])
{
    // 1. Each thread computes a local sum
    float local_sum = 0.0f;
    // uint grid_size = threads_per_group * 2048; // Total threads in the grid
    for (uint index = (group_id * threads_per_group) + thread_id_in_group;
              index < dataSize;
              index += grid_size) {
        local_sum += inData[index];
    }
    
    // 2. SIMD-accelerated threadgroup reduction
    threadgroup float shared_memory[32];
    float total = tg_reduce_float(local_sum, thread_id_in_group, threads_per_group, shared_memory);

    // 3. The first thread in each group writes the final partial sum.
    if (thread_id_in_group == 0) {
        partialSums[group_id] = total;
    }
}


// Stage 2: Reduces the buffer of partial sums into a final single result.
kernel void sum_kernel_stage2(const device float* partialSums,
                              device float* finalResult,
                              constant uint& num_threadgroups,
                              uint index [[thread_position_in_grid]])
{
    // A single thread iterates through the partial sums and calculates the final total.
    if (index == 0) {
        float total_sum = 0.0f;
        for(uint i = 0; i < num_threadgroups; ++i) {
            total_sum += partialSums[i];
        }
        finalResult[0] = total_sum;
    }
}

// --- HASH JOIN KERNELS ---
// SELECT * FROM lineitem JOIN orders ON lineitem.l_orderkey = orders.o_orderkey;

// Represents an entry in our simple hash table
struct HashTableEntry {
    atomic_int key;   // o_orderkey
    atomic_int value; // Row ID (payload)
};

// Phase 1: Builds a hash table from the smaller table (orders)
kernel void hash_join_build(const device int* inKeys,      // Input: o_orderkey
                            const device int* inValues,    // Input: Row IDs
                            device HashTableEntry* hashTable,
                            constant uint& dataSize,
                            constant uint& hashTableSize,
                            uint index [[thread_position_in_grid]])
{
    if (index >= dataSize) {
        return;
    }

    int key = inKeys[index];
    int value = inValues[index];

    // Simple hash function
    uint hash_index = (uint)key % hashTableSize;

    // Linear probing with atomic operations to handle collisions
    for (uint i = 0; i < hashTableSize; ++i) {
        uint probe_index = (hash_index + i) % hashTableSize;

        // Try to insert the key if the slot is empty (key == -1)
        int expected = -1;
        if (atomic_compare_exchange_weak_explicit(&hashTable[probe_index].key,
                                                    &expected,
                                                    key,
                                                    memory_order_relaxed,
                                                    memory_order_relaxed)) {
            // If we successfully claimed the slot, write the value
            atomic_store_explicit(&hashTable[probe_index].value, value, memory_order_relaxed);
            return; // Exit the loop after successful insertion
        }
    }
}

// Phase 2: Probes the hash table using keys from the larger table (lineitem)
kernel void hash_join_probe(const device int* probeKeys,        // Input: l_orderkey
                            const device HashTableEntry* hashTable,
                            device atomic_uint* match_count, // Output: counter for successful joins
                            constant uint& probeDataSize,
                            constant uint& hashTableSize,
                            uint index [[thread_position_in_grid]])
{
    if (index >= probeDataSize) {
        return;
    }

    int key_to_find = probeKeys[index];

    // Simple hash function (must be identical to the build phase)
    uint hash_index = (uint)key_to_find % hashTableSize;

    // Linear probing to find the key
    for (uint i = 0; i < hashTableSize; ++i) {
        uint probe_index = (hash_index + i) % hashTableSize;
        
        int table_key = atomic_load_explicit(&hashTable[probe_index].key, memory_order_relaxed);

        // If we find our key, we have a match.
        if (table_key == key_to_find) {
            atomic_fetch_add_explicit(match_count, 1, memory_order_relaxed);
            return; // Found a match, this thread is done.
        }

        // If we find an empty slot, the key is not in the table.
        if (table_key == -1) {
            return; // Key not found, this thread is done.
        }
    }
}


// --- TPC-H Q1 KERNELS ---
// TPC-H Query 1: Pricing Summary Report Query
/*
SELECT 
    l_returnflag,
    l_linestatus,
    SUM(l_quantity) AS sum_qty,
    SUM(l_extendedprice) AS sum_base_price,
    SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
    SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
    AVG(l_quantity) AS avg_qty,
    AVG(l_extendedprice) AS avg_price,
    AVG(l_discount) AS avg_disc,
    COUNT(*) AS count_order
FROM lineitem
WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus;
*/

// --- Q1 Specialized low-cardinality bins path ---
// Exploits the fact that Q1 groups only by l_returnflag in {A,N,R} and l_linestatus in {F,O}.
// We maintain 6 bins: (A,F)=0, (A,O)=1, (N,F)=2, (N,O)=3, (R,F)=4, (R,O)=5.
inline int q1_rf_index(char rf) {
    return (rf == 'A') ? 0 : (rf == 'N') ? 1 : (rf == 'R') ? 2 : -1;
}
inline int q1_ls_index(char ls) {
    return (ls == 'F') ? 0 : (ls == 'O') ? 1 : -1;
}

// --- Q1 Integer-cent two-pass path ---
// Stage 1: Per-thread local accumulation into 6 bins using integer cents (and basis points),
// then threadgroup reduction to one partial per bin per threadgroup. No atomics used.
// Monetary fields are accumulated in cents (int64). Discount is accumulated in basis points (bp, int32).
kernel void q1_bins_accumulate_int_stage1(
    const device int*   l_shipdate,
    const device char*  l_returnflag,
    const device char*  l_linestatus,
    const device float* l_quantity,
    const device float* l_extendedprice,
    const device float* l_discount,
    const device float* l_tax,
    // Outputs: one partial per threadgroup per bin (size = num_threadgroups * 6)
    device long*  p_sum_qty_cents,        // int64
    device long*  p_sum_base_cents,       // int64
    device long*  p_sum_disc_price_cents, // int64
    device long*  p_sum_charge_cents,     // int64
    device uint*  p_sum_discount_bp,      // uint32 (sum of basis points)
    device uint*  p_count,                // uint32
    constant uint& data_size,
    constant int&  cutoff_date,
    constant uint& num_threadgroups,
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    const int BINS = 6;
    // Per-thread local accumulators in integer units
    long sum_qty_c[BINS];
    long sum_base_c[BINS];
    long sum_disc_c[BINS];
    long sum_charge_c[BINS];
    uint sum_disc_bp[BINS];
    uint cnt[BINS];
    for (int b = 0; b < BINS; ++b) { sum_qty_c[b]=0; sum_base_c[b]=0; sum_disc_c[b]=0; sum_charge_c[b]=0; sum_disc_bp[b]=0u; cnt[b]=0u; }

    // Grid stride loop using actual dispatched num_threadgroups
    // uint grid_size = threads_per_group * num_threadgroups;
    for (uint i = (group_id * threads_per_group) + thread_id_in_group; i < data_size; i += grid_size) {
        // [SEL] Selection: filter rows by shipdate and bin by (returnflag, linestatus)
        if (l_shipdate[i] > cutoff_date) continue;
        int rfi = q1_rf_index(l_returnflag[i]); if (rfi < 0) continue;
        int lsi = q1_ls_index(l_linestatus[i]); if (lsi < 0) continue;
        int bin = rfi * 2 + lsi; // 0..5

        // [PROJ] Projection: convert to fixed-point (cents for currency, basis points for percentages)
        float base = l_extendedprice[i];
        float qty  = l_quantity[i];
        float d    = l_discount[i];
        float t    = l_tax[i];

        long base_c = (long)floor(base * 100.0f + 0.5f);  // cents (int64)
        long qty_c  = (long)floor(qty  * 100.0f + 0.5f);  // cents (int64)
        int  d_bp   = (int)floor(d * 100.0f + 0.5f);      // basis points (int)
        int  t_bp   = (int)floor(t * 100.0f + 0.5f);      // basis points (int)

        // Compute derived columns in fixed-point
        long disc_c = (base_c * (long)(100 - d_bp) + 50) / 100;  // disc_price_cents
        long charge_c = (disc_c * (long)(100 + t_bp) + 50) / 100; // charge_cents

        // [AGG-LOCAL] Local accumulation: per-thread accumulators (6 bins, no atomics)
        sum_qty_c[bin]      += qty_c;
        sum_base_c[bin]     += base_c;
        sum_disc_c[bin]     += disc_c;
        sum_charge_c[bin]   += charge_c;
        sum_disc_bp[bin]    += (uint)d_bp;
        cnt[bin]            += 1u;
    }

    // SIMD-accelerated threadgroup reduction (2 barriers per metric instead of ~12)
    threadgroup long tg64[32];  // one per SIMD group
    threadgroup uint tg32[32];

    // [AGG-LOCAL] Threadgroup reduction for each bin and metric via SIMD groups
    for (int b = 0; b < BINS; ++b) {
        long r;
        r = tg_reduce_long(sum_qty_c[b], thread_id_in_group, threads_per_group, tg64);
        if (thread_id_in_group == 0) p_sum_qty_cents[group_id * BINS + b] = r;

        r = tg_reduce_long(sum_base_c[b], thread_id_in_group, threads_per_group, tg64);
        if (thread_id_in_group == 0) p_sum_base_cents[group_id * BINS + b] = r;

        r = tg_reduce_long(sum_disc_c[b], thread_id_in_group, threads_per_group, tg64);
        if (thread_id_in_group == 0) p_sum_disc_price_cents[group_id * BINS + b] = r;

        r = tg_reduce_long(sum_charge_c[b], thread_id_in_group, threads_per_group, tg64);
        if (thread_id_in_group == 0) p_sum_charge_cents[group_id * BINS + b] = r;

        uint u;
        u = tg_reduce_uint(sum_disc_bp[b], thread_id_in_group, threads_per_group, tg32);
        if (thread_id_in_group == 0) p_sum_discount_bp[group_id * BINS + b] = u;

        u = tg_reduce_uint(cnt[b], thread_id_in_group, threads_per_group, tg32);
        if (thread_id_in_group == 0) p_count[group_id * BINS + b] = u;
    }
}

// Stage 2: Reduce per-threadgroup partials into final 6-bin results (all on GPU).
kernel void q1_bins_reduce_int_stage2(
    const device long* p_sum_qty_cents,
    const device long* p_sum_base_cents,
    const device long* p_sum_disc_price_cents,
    const device long* p_sum_charge_cents,
    const device uint* p_sum_discount_bp,
    const device uint* p_count,
    device long* out_sum_qty_cents,
    device long* out_sum_base_cents,
    device long* out_sum_disc_price_cents,
    device long* out_sum_charge_cents,
    device uint* out_sum_discount_bp,
    device uint* out_count,
    constant uint& num_threadgroups,
    uint index [[thread_position_in_grid]])
{
    // [AGG-MERGE] Global reduction: single-thread final reduce over all threadgroups
    if (index != 0) return;
    const uint BINS = 6;
    for (uint b = 0; b < BINS; ++b) {
        long s_qty = 0, s_base = 0, s_disc = 0, s_charge = 0;
        uint s_dbp = 0, s_cnt = 0;
        for (uint g = 0; g < num_threadgroups; ++g) {
            uint idx = g * BINS + b;
            s_qty    += p_sum_qty_cents[idx];
            s_base   += p_sum_base_cents[idx];
            s_disc   += p_sum_disc_price_cents[idx];
            s_charge += p_sum_charge_cents[idx];
            s_dbp    += p_sum_discount_bp[idx];
            s_cnt    += p_count[idx];
        }
        out_sum_qty_cents[b]        = s_qty;
        out_sum_base_cents[b]       = s_base;
        out_sum_disc_price_cents[b] = s_disc;
        out_sum_charge_cents[b]     = s_charge;
        out_sum_discount_bp[b]      = s_dbp;
        out_count[b]                = s_cnt;
    }
}



// --- TPC-H Q6 KERNELS ---
// SELECT SUM(l_extendedprice * l_discount) AS revenue
// FROM lineitem
// WHERE l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01' 
//   AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24;

// Stage 1: Filter and compute partial revenue sums per threadgroup
kernel void q6_filter_and_sum_stage1(
    const device int* l_shipdate,        // Date as YYYYMMDD integer
    const device float* l_discount,      // Discount factor 
    const device float* l_quantity,      // Quantity
    const device float* l_extendedprice, // Extended price
    device float* partial_revenues,      // Output: partial sums per threadgroup
    constant uint& data_size,
    constant int& start_date,            // 19940101 (1994-01-01)
    constant int& end_date,              // 19950101 (1995-01-01)
    constant float& min_discount,        // 0.05
    constant float& max_discount,        // 0.07
    constant float& max_quantity,        // 24.0
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    // 1. Each thread computes a local revenue sum
    float local_revenue = 0.0f;
    // uint grid_size = threads_per_group * 2048; // Total threads in the grid
    
    for (uint index = (group_id * threads_per_group) + thread_id_in_group;
         index < data_size;
         index += grid_size) {
        
        // Apply all filter conditions
        if (l_shipdate[index] >= start_date && 
            l_shipdate[index] < end_date &&
            l_discount[index] >= min_discount && 
            l_discount[index] <= max_discount &&
            l_quantity[index] < max_quantity) {
            
            // Calculate revenue for this qualifying row
            local_revenue += l_extendedprice[index] * l_discount[index];
        }
    }
    
    // 2. SIMD-accelerated threadgroup reduction
    threadgroup float shared_memory[32];
    float total = tg_reduce_float(local_revenue, thread_id_in_group, threads_per_group, shared_memory);

    // 3. The first thread in each group writes the partial revenue sum
    if (thread_id_in_group == 0) {
        partial_revenues[group_id] = total;
    }
}

// Stage 2: Reduce partial revenue sums to final result
kernel void q6_final_sum_stage2(
    const device float* partial_revenues,
    device float* final_revenue,
    constant uint& num_threadgroups,
    uint index [[thread_position_in_grid]])
{
    // A single thread sums all partial revenues to get final result
    if (index == 0) {
        float total_revenue = 0.0f;
        for (uint i = 0; i < num_threadgroups; ++i) {
            total_revenue += partial_revenues[i];
        }
        final_revenue[0] = total_revenue;
    }
}


// --- TPC-H Q3 KERNELS ---
// TPC-H Query 3: Shipping Priority Query
/*
SELECT 
    l_orderkey,
    SUM(l_extendedprice * (1 - l_discount)) AS revenue,
    o_orderdate,
    o_shippriority
FROM customer, orders, lineitem
WHERE c_mktsegment = 'BUILDING'
  AND c_custkey = o_custkey
  AND l_orderkey = o_orderkey
  AND o_orderdate < '1995-03-15'
  AND l_shipdate > '1995-03-15'
GROUP BY l_orderkey, o_orderdate, o_shippriority
ORDER BY revenue DESC, o_orderdate
LIMIT 10;
*/

// Struct for the final aggregation results for Q3
struct Q3Aggregates {
    atomic_int key; // orderkey
    atomic_float revenue;
    atomic_uint orderdate;
    atomic_uint shippriority;
};


// KERNEL 1: Build a BITMAP on the CUSTOMER table.
// Replaces hash table with a simple bitmap for 'BUILDING' segment.
kernel void q3_build_customer_bitmap_kernel(
    const device int* c_custkey,
    const device char* c_mktsegment,
    device atomic_uint* customer_bitmap,
    constant uint& customer_size,
    uint index [[thread_position_in_grid]])
{
    if (index >= customer_size) return;

    if (c_mktsegment[index] == 'B') { // 'B' for BUILDING
        int key = c_custkey[index];
        // Set bit at 'key'
        uint word_idx = key / 32;
        uint bit_idx = key % 32;
        atomic_fetch_or_explicit(&customer_bitmap[word_idx], (1u << bit_idx), memory_order_relaxed);
    }
}

// KERNEL 2: Build a DIRECT MAP on the ORDERS table.
// Replaces hash table with a direct lookup array: orders_map[orderkey] = row_index
kernel void q3_build_orders_map_kernel(
    const device int* o_orderkey,
    const device int* o_orderdate,
    device int* orders_map,
    constant uint& orders_size,
    constant int& cutoff_date, // 19950315
    uint index [[thread_position_in_grid]])
{
    if (index >= orders_size) return;

    if (o_orderdate[index] < cutoff_date) {
        int key = o_orderkey[index];
        // Direct map: index by orderkey
        orders_map[key] = (int)index;
    }
}

// --- TPC-H Q9 KERNELS ---
// TPC-H Query 9: Product Type Profit Measure Query
/*
SELECT 
    nation,
    o_year,
    SUM(amount) AS sum_profit
FROM (
    SELECT 
        n_name AS nation,
        EXTRACT(year FROM o_orderdate) AS o_year,
        l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount
    FROM part, supplier, lineitem, partsupp, orders, nation
    WHERE s_suppkey = l_suppkey
      AND ps_suppkey = l_suppkey
      AND ps_partkey = l_partkey
      AND p_partkey = l_partkey
      AND o_orderkey = l_orderkey
      AND s_nationkey = n_nationkey
      AND p_name LIKE '%green%'
) AS profit
GROUP BY nation, o_year
ORDER BY nation, o_year DESC;
*/

// Struct for the final aggregation results for Q9
struct Q9Aggregates {
    atomic_uint key; // Packed (nation_key << 16) | year
    atomic_float profit;
};

// A non-atomic version for fast local aggregation
struct Q9Aggregates_Local {
    uint key;
    float profit;
};

// KERNEL 1: Build Bitmap on PART, filtering for p_name LIKE '%green%'
kernel void q9_build_part_ht_kernel(
    const device int* p_partkey [[buffer(0)]],
    const device char* p_name [[buffer(1)]], // Assuming p_name is a fixed-size char array
    device atomic_uint* part_bitmap [[buffer(2)]], // Bitmap: 1 bit per partkey
    constant uint& part_size [[buffer(3)]],
    constant uint& part_ht_size [[buffer(4)]], // Unused, kept for signature compatibility if needed, or remove
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    uint index = group_id * threads_per_group + thread_id_in_group;
    if (index >= part_size) return;
    bool match = false;
    for(int i = 0; i < 50; ++i) { // Simplified string search
        if (p_name[index * 55 + i] == 'g' && p_name[index * 55 + i + 1] == 'r' &&
            p_name[index * 55 + i + 2] == 'e' && p_name[index * 55 + i + 3] == 'e' &&
            p_name[index * 55 + i + 4] == 'n') {
            match = true;
            break;
        }
    }
    
    if (match) {
        int key = p_partkey[index];
        // Set bit in bitmap
        atomic_fetch_or_explicit(&part_bitmap[key / 32], (1u << (key % 32)), memory_order_relaxed);
    }
}

// KERNEL 2: Build Direct Map on SUPPLIER, storing nationkey as the value.
kernel void q9_build_supplier_ht_kernel(
    const device int* s_suppkey [[buffer(0)]],
    const device int* s_nationkey [[buffer(1)]],
    device int* supplier_nation_map [[buffer(2)]], // Direct map: index is suppkey
    constant uint& supplier_size [[buffer(3)]],
    constant uint& supplier_ht_size [[buffer(4)]], // Unused
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    uint index = group_id * threads_per_group + thread_id_in_group;
    if (index >= supplier_size) return;
    int key = s_suppkey[index];
    // Safety check (though we allocated max_suppkey + 1)
    // Assuming key is positive and within bounds.
    supplier_nation_map[key] = s_nationkey[index];
}

// KERNEL 3: Build HT on PARTSUPP, storing supplycost index as value
struct PartSuppEntry {
    atomic_int partkey;
    atomic_int suppkey;
    atomic_int idx; // row index into ps_supplycost array
    int _pad; // ensure 16-byte stride for predictable layout
};

kernel void q9_build_partsupp_ht_kernel(
    const device int* ps_partkey [[buffer(0)]],
    const device int* ps_suppkey [[buffer(1)]],
    device PartSuppEntry* partsupp_ht [[buffer(2)]],
    constant uint& partsupp_size [[buffer(3)]],
    constant uint& partsupp_ht_size [[buffer(4)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    uint index = group_id * threads_per_group + thread_id_in_group;
    if (index >= partsupp_size) return;
    int pk = ps_partkey[index];
    int sk = ps_suppkey[index];
    int val = (int)index;
    // Combined hash of (partkey, suppkey) to reduce probe lengths
    uint mix = (uint)pk * 0x9E3779B1u ^ (uint)sk * 0x85EBCA77u;
    uint hash_index = mix % partsupp_ht_size;
    for (uint i = 0; i < partsupp_ht_size; ++i) {
        uint probe_index = (hash_index + i) % partsupp_ht_size;
        // Try to claim empty slot by setting partkey from -1 to pk
        int expected_pk = -1;
        if (atomic_compare_exchange_weak_explicit(&partsupp_ht[probe_index].partkey, &expected_pk, pk, memory_order_relaxed, memory_order_relaxed)) {
            // Successfully claimed slot; write suppkey and idx
            atomic_store_explicit(&partsupp_ht[probe_index].suppkey, sk, memory_order_relaxed);
            atomic_store_explicit(&partsupp_ht[probe_index].idx, val, memory_order_relaxed);
            return;
        }
        // If slot has our (pk,sk), update idx and return (data is unique so this just sets once)
        int cur_pk = atomic_load_explicit(&partsupp_ht[probe_index].partkey, memory_order_relaxed);
        if (cur_pk == -1) { return; } // empty slot -> not found
        if (cur_pk == pk) {
            int cur_sk = atomic_load_explicit(&partsupp_ht[probe_index].suppkey, memory_order_relaxed);
            if (cur_sk == sk) {
                atomic_store_explicit(&partsupp_ht[probe_index].idx, val, memory_order_relaxed);
                return;
            }
        }
        // else continue probing
    }
}

// KERNEL 4: Build HT on ORDERS, storing year as value
kernel void q9_build_orders_ht_kernel(
    const device int* o_orderkey [[buffer(0)]],
    const device int* o_orderdate [[buffer(1)]],
    device HashTableEntry* orders_ht [[buffer(2)]],
    constant uint& orders_size [[buffer(3)]],
    constant uint& orders_ht_size [[buffer(4)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    uint index = group_id * threads_per_group + thread_id_in_group;
    if (index >= orders_size) return;
    int key = o_orderkey[index];
    int value = o_orderdate[index] / 10000; // Extract year
    uint hash_index = (uint)key % orders_ht_size;
    for (uint i = 0; i < orders_ht_size; ++i) {
        uint probe_index = (hash_index + i) % orders_ht_size;
        int expected = -1;
        if (atomic_compare_exchange_weak_explicit(&orders_ht[probe_index].key, &expected, key, memory_order_relaxed, memory_order_relaxed)) {
            atomic_store_explicit(&orders_ht[probe_index].value, value, memory_order_relaxed);
            return;
        }
    }
}


// KERNEL 5: The main kernel. Streams lineitem and probes all other hash tables.
kernel void q9_probe_and_local_agg_kernel(
    // lineitem columns
    const device int* l_suppkey, const device int* l_partkey, const device int* l_orderkey,
    const device float* l_extendedprice, const device float* l_discount, const device float* l_quantity,
    // partsupp supplycost array
    const device float* ps_supplycost,
    // Pre-built hash tables
    const device uint* part_bitmap, 
    const device int* supplier_nation_map,
    const device PartSuppEntry* partsupp_ht, const device HashTableEntry* orders_ht,
    // Output buffer
    device Q9Aggregates_Local* intermediate_results,
    // Parameters
    constant uint& lineitem_size, constant uint& part_ht_size, constant uint& supplier_ht_size,
    constant uint& partsupp_ht_size, constant uint& orders_ht_size,
    // Thread IDs
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    const int local_ht_size = 256;
    threadgroup Q9Aggregates_Local local_ht[local_ht_size];
    threadgroup atomic_int tg_locks[local_ht_size]; // 0=unlocked, 1=locked
    for (int i = thread_id_in_group; i < local_ht_size; i += threads_per_group) {
        local_ht[i].key = 0; local_ht[i].profit = 0.0f;
        atomic_store_explicit(&tg_locks[i], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // const uint grid_size = threads_per_group * 2048;
    const uint global_tid = (group_id * threads_per_group) + thread_id_in_group;
    const uint BATCH = 4;
    const uint stride = grid_size * BATCH;

    for (uint base = global_tid * BATCH; base < lineitem_size; base += stride) {
        for(int k=0; k<BATCH; ++k) {
            uint i = base + k;
            if (i >= lineitem_size) break;

            int partkey = l_partkey[i];

            // 1. Check Bitmap
            if (!((part_bitmap[partkey / 32] >> (partkey % 32)) & 1)) continue;

            int suppkey = l_suppkey[i];

            // 2. Direct Lookup
            int nationkey = supplier_nation_map[suppkey];
            if (nationkey == -1) continue;


            // 3. Probe partsupp_ht to get supply cost index (use combined hash of (partkey,suppkey))
            int ps_idx = -1;
            uint ps_hash = ((uint)partkey * 0x9E3779B1u ^ (uint)suppkey * 0x85EBCA77u) % partsupp_ht_size;
            for (uint j = 0; j < partsupp_ht_size; ++j) {
                uint probe_idx = (ps_hash + j) % partsupp_ht_size;
                int pk2 = atomic_load_explicit(&partsupp_ht[probe_idx].partkey, memory_order_relaxed);
                if (pk2 == -1) break; // empty slot -> not found
                if (pk2 == partkey) {
                    int sk2 = atomic_load_explicit(&partsupp_ht[probe_idx].suppkey, memory_order_relaxed);
                    if (sk2 == suppkey) {
                        ps_idx = atomic_load_explicit(&partsupp_ht[probe_idx].idx, memory_order_relaxed);
                        break;
                    }
                }
            }
            if (ps_idx == -1) continue;

            // 4. Probe orders_ht to get year
            int orderkey = l_orderkey[i];
            int year = -1;
            uint ord_hash = (uint)orderkey % orders_ht_size;
            for (uint j = 0; j < orders_ht_size; ++j) {
                uint probe_idx = (ord_hash + j) % orders_ht_size;
                int o_key = atomic_load_explicit(&orders_ht[probe_idx].key, memory_order_relaxed);
                if (o_key == orderkey) {
                    year = atomic_load_explicit(&orders_ht[probe_idx].value, memory_order_relaxed);
                    break;
                }
                if (o_key == -1) break;
            }
            if (year == -1) continue;

            // All probes succeeded!
            
            // --- AGGREGATE ---
            float profit = l_extendedprice[i] * (1.0f - l_discount[i]) - ps_supplycost[ps_idx] * l_quantity[i];
            uint agg_key = (uint)(nationkey << 16) | year;
            uint agg_hash = agg_key % local_ht_size;

            for(int m = 0; m < local_ht_size; ++m) {
                uint probe_idx = (agg_hash + m) % local_ht_size;
                int expected = 0;
                if (atomic_compare_exchange_weak_explicit(&tg_locks[probe_idx], &expected, 1, memory_order_relaxed, memory_order_relaxed)) {
                    if (local_ht[probe_idx].key == 0) {
                        local_ht[probe_idx].key = agg_key;
                        local_ht[probe_idx].profit = profit;
                        atomic_store_explicit(&tg_locks[probe_idx], 0, memory_order_relaxed);
                        break;
                    } else if (local_ht[probe_idx].key == agg_key) {
                        local_ht[probe_idx].profit += profit;
                        atomic_store_explicit(&tg_locks[probe_idx], 0, memory_order_relaxed);
                        break;
                    }
                    // release and continue probing
                    atomic_store_explicit(&tg_locks[probe_idx], 0, memory_order_relaxed);
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Write local results to global memory
    for (int i = thread_id_in_group; i < local_ht_size; i += threads_per_group) {
        if (local_ht[i].key != 0) {
            intermediate_results[group_id * local_ht_size + i] = local_ht[i];
        }
    }
}


// KERNEL 6: Final merge kernel
kernel void q9_merge_results_kernel(
    const device Q9Aggregates_Local* intermediate_results,
    device Q9Aggregates* final_hashtable,
    constant uint& intermediate_size,
    constant uint& final_hashtable_size,
    uint index [[thread_position_in_grid]])
{
    // This logic from your implementation was correct and can be reused.
    if (index >= intermediate_size) return;
    Q9Aggregates_Local local_result = intermediate_results[index];
    if (local_result.key == 0) return;

    uint hash_index = local_result.key % final_hashtable_size;
    for (uint i = 0; i < final_hashtable_size; ++i) {
        uint probe_index = (hash_index + i) % final_hashtable_size;
        uint expected = 0;
        if (atomic_compare_exchange_weak_explicit(&final_hashtable[probe_index].key, &expected, local_result.key, memory_order_relaxed, memory_order_relaxed)) {
            atomic_store_explicit(&final_hashtable[probe_index].profit, 0.0f, memory_order_relaxed);
        }
        if (atomic_load_explicit(&final_hashtable[probe_index].key, memory_order_relaxed) == local_result.key) {
            atomic_fetch_add_explicit(&final_hashtable[probe_index].profit, local_result.profit, memory_order_relaxed);
            return;
        }
    }
}

// --- TPC-H Query 13 Kernels ---
/*
SELECT
    c_count,
    COUNT(*) AS custdist
FROM (
    -- Inner Query: First, for each customer, count their non-special orders.
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
-- Outer Query: Then, group those results again to create a histogram.
GROUP BY
    c_count
ORDER BY
    custdist DESC,
    c_count DESC;

*/



// --- Q13 substring matching helpers ---

inline int q13_effective_len_fixed_100(const device uchar* s) {
    const int max_len = 100;
    const int min_pattern_len = 15;
    for (int i = 0; i < max_len; i++) {
        if (s[i] == 0) {
            return (i < min_pattern_len) ? -1 : i;
        }
    }
    return max_len;
}

inline bool q13_has_special_requests(const device uchar* s, int comment_len) {
    const int last_special = comment_len - 15;
    
    for (int i = 0; i <= last_special; i++) {
        if (s[i+3] == 'c') {
            if (s[i] == 's' && s[i+1] == 'p' && s[i+2] == 'e' && 
                s[i+4] == 'i' && s[i+5] == 'a' && s[i+6] == 'l') {
                
                int req_start = i + 7;
                int req_end = comment_len - 8;
                for (int j = req_start; j <= req_end; j++) {
                    if (s[j+2] == 'q') {
                        if (s[j] == 'r' && s[j+1] == 'e' && s[j+3] == 'u' &&
                            s[j+4] == 'e' && s[j+5] == 's' && s[j+6] == 't' && s[j+7] == 's') {
                            return true;
                        }
                    }
                }
                return false;
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
    // const uint grid_size = threads_per_group * 2048;
    const uint global_tid = (group_id * threads_per_group) + thread_id_in_group;
    const uint BATCH = 4;
    const uint stride = grid_size * BATCH;

    for (uint base = global_tid * BATCH; base < orders_size; base += stride) {
        if (base + 0 < orders_size) {
            const uint i = base + 0;
            const uint ck = (uint)o_custkey[i];
            if (ck >= 1u && ck <= customer_size) {
                const device uchar* row = (const device uchar*)(o_comment + (i * comment_len));
                int effective_len = q13_effective_len_fixed_100(row);
                if (effective_len > 0) {
                    bool skip = q13_has_special_requests(row, effective_len);
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
                int effective_len = q13_effective_len_fixed_100(row);
                if (effective_len > 0) {
                    bool skip = q13_has_special_requests(row, effective_len);
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
                int effective_len = q13_effective_len_fixed_100(row);
                if (effective_len > 0) {
                    bool skip = q13_has_special_requests(row, effective_len);
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
                int effective_len = q13_effective_len_fixed_100(row);
                if (effective_len > 0) {
                    bool skip = q13_has_special_requests(row, effective_len);
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
// Q3 Fused Probe + Direct Aggregation Kernel
// Eliminates intermediate append buffer; aggregates directly
// into a global hash table with atomic CAS + fetch_add.
// =============================================================
kernel void q3_probe_and_aggregate_direct_kernel(
    const device int* l_orderkey        [[buffer(0)]],
    const device int* l_shipdate        [[buffer(1)]],
    const device float* l_extendedprice [[buffer(2)]],
    const device float* l_discount      [[buffer(3)]],
    const device uint* customer_bitmap  [[buffer(4)]],
    const device int* orders_map        [[buffer(5)]],
    const device int* o_custkey         [[buffer(6)]],
    const device int* o_orderdate       [[buffer(7)]],
    const device int* o_shippriority    [[buffer(8)]],
    device Q3Aggregates* final_ht       [[buffer(9)]],
    constant uint& lineitem_size        [[buffer(10)]],
    constant int& cutoff_date           [[buffer(11)]],
    constant uint& final_ht_size        [[buffer(12)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint grid_size [[threads_per_grid]])
{
    uint global_id = (group_id * threads_per_group) + thread_id_in_group;
    const int BATCH_SIZE = 4;

    for (uint i = global_id; i < lineitem_size; i += grid_size * BATCH_SIZE) {
        uint idx[BATCH_SIZE];
        bool active[BATCH_SIZE];

        for (int k = 0; k < BATCH_SIZE; k++) {
            idx[k] = i + k * grid_size;
            active[k] = (idx[k] < lineitem_size);
        }

        int l_shipdate_val[BATCH_SIZE];
        int l_orderkey_val[BATCH_SIZE];

        for (int k = 0; k < BATCH_SIZE; k++) {
            if (active[k]) {
                l_shipdate_val[k] = l_shipdate[idx[k]];
                l_orderkey_val[k] = l_orderkey[idx[k]];
            }
        }

        // Filter 1: l_shipdate > cutoff_date
        bool pass_date[BATCH_SIZE];
        for (int k = 0; k < BATCH_SIZE; k++) {
            pass_date[k] = active[k] && (l_shipdate_val[k] > cutoff_date);
        }

        // Probe Orders Direct Map
        int orders_idx[BATCH_SIZE];
        for (int k = 0; k < BATCH_SIZE; k++) {
            orders_idx[k] = pass_date[k] ? orders_map[l_orderkey_val[k]] : -1;
        }

        bool pass_order[BATCH_SIZE];
        for (int k = 0; k < BATCH_SIZE; k++) {
            pass_order[k] = pass_date[k] && (orders_idx[k] != -1);
        }

        // Probe Customer Bitmap
        int custkey[BATCH_SIZE];
        for (int k = 0; k < BATCH_SIZE; k++) {
            if (pass_order[k]) custkey[k] = o_custkey[orders_idx[k]];
        }

        bool pass_customer[BATCH_SIZE];
        for (int k = 0; k < BATCH_SIZE; k++) {
            if (pass_order[k]) {
                uint word_idx = custkey[k] / 32;
                uint bit_idx = custkey[k] % 32;
                pass_customer[k] = (customer_bitmap[word_idx] & (1u << bit_idx)) != 0;
            } else {
                pass_customer[k] = false;
            }
        }

        // Direct aggregation into final hash table
        for (int k = 0; k < BATCH_SIZE; k++) {
            if (pass_customer[k]) {
                float revenue = l_extendedprice[idx[k]] * (1.0f - l_discount[idx[k]]);
                int key = l_orderkey_val[k];
                uint hash = (uint)key % final_ht_size;

                for (uint p = 0; p < final_ht_size; p++) {
                    uint probe_idx = (hash + p) % final_ht_size;
                    int expected = 0;
                    if (atomic_compare_exchange_weak_explicit(
                            &final_ht[probe_idx].key, &expected, key,
                            memory_order_relaxed, memory_order_relaxed)) {
                        // Claimed new slot — set metadata (same for all lineitems of this order)
                        atomic_store_explicit(&final_ht[probe_idx].orderdate,
                                             (uint)o_orderdate[orders_idx[k]], memory_order_relaxed);
                        atomic_store_explicit(&final_ht[probe_idx].shippriority,
                                             (uint)o_shippriority[orders_idx[k]], memory_order_relaxed);
                    }
                    int current_key = atomic_load_explicit(&final_ht[probe_idx].key, memory_order_relaxed);
                    if (current_key == key) {
                        atomic_fetch_add_explicit(&final_ht[probe_idx].revenue, revenue, memory_order_relaxed);
                        break;
                    }
                }
            }
        }
    }
}

// Non-atomic struct matching Q3Aggregates layout for post-aggregation compaction
struct Q3CompactResult {
    int key;
    float revenue;
    uint orderdate;
    uint shippriority;
};

// GPU compaction: extract non-empty entries from sparse hash table into dense output
kernel void q3_compact_results_kernel(
    device Q3CompactResult* hash_table [[buffer(0)]],
    device Q3CompactResult* output     [[buffer(1)]],
    device atomic_uint& result_count   [[buffer(2)]],
    constant uint& ht_size             [[buffer(3)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= ht_size) return;
    if (hash_table[index].key > 0) {
        uint pos = atomic_fetch_add_explicit(&result_count, 1, memory_order_relaxed);
        output[pos] = hash_table[index];
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
