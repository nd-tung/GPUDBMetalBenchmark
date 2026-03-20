#include "common.h"

// ===================================================================
// SF100 RADIX PARTITIONED HASH JOIN KERNELS
// ===================================================================

constant uint SF100_RADIX_BITS = 10;
constant uint SF100_NUM_PARTITIONS = 1024; // 1 << 10

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
