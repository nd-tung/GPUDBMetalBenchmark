#include "common.h"

// ===================================================================
// TPC-H Q9 KERNELS — Product Type Profit Measure
// ===================================================================
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
// Uses SWAR to scan 4 bytes at a time for the discriminant byte 'g', then
// verifies the remaining "reen" at candidate positions.
kernel void q9_build_part_ht_kernel(
    const device int* p_partkey [[buffer(0)]],
    const device char* p_name [[buffer(1)]],
    device atomic_uint* part_bitmap [[buffer(2)]],
    constant uint& part_size [[buffer(3)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    uint index = group_id * threads_per_group + thread_id_in_group;
    if (index >= part_size) return;

    const device uchar* name = (const device uchar*)(p_name + index * 55);
    bool match = false;
    const uint g_spread = 0x67676767u; // 'g' = 0x67 broadcast to all 4 bytes

    // SWAR: process 13 words covering bytes 0-51 (positions up to 50 can start "green")
    for (int w = 0; w < 13 && !match; ++w) {
        // Build a uint from 4 consecutive bytes (unaligned-safe on Metal device memory)
        uint word = (uint)name[w * 4]
                  | ((uint)name[w * 4 + 1] << 8)
                  | ((uint)name[w * 4 + 2] << 16)
                  | ((uint)name[w * 4 + 3] << 24);
        // SWAR zero-byte detection: any byte == 'g' ?
        uint xor_val = word ^ g_spread;
        if (((xor_val - 0x01010101u) & ~xor_val & 0x80808080u) == 0) continue;

        // At least one byte matches 'g'; verify each position
        int base = w * 4;
        for (int b = 0; b < 4 && base + b <= 50; ++b) {
            if (name[base + b] == 'g' &&
                name[base + b + 1] == 'r' &&
                name[base + b + 2] == 'e' &&
                name[base + b + 3] == 'e' &&
                name[base + b + 4] == 'n') {
                match = true;
                break;
            }
        }
    }

    if (match) {
        bitmap_set(part_bitmap, p_partkey[index]);
    }
}

// KERNEL 2: Build Direct Map on SUPPLIER, storing nationkey as the value.
kernel void q9_build_supplier_ht_kernel(
    const device int* s_suppkey [[buffer(0)]],
    const device int* s_nationkey [[buffer(1)]],
    device int* supplier_nation_map [[buffer(2)]], // Direct map: index is suppkey
    constant uint& supplier_size [[buffer(3)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    uint index = group_id * threads_per_group + thread_id_in_group;
    if (index >= supplier_size) return;
    int key = s_suppkey[index];
    supplier_nation_map[key] = s_nationkey[index];
}

// KERNEL 3: Build HT on PARTSUPP, storing supplycost index as value
// Uses part_bitmap as a Bloom-style pre-filter: only inserts entries whose
// partkey passed the '%green%' filter, making the HT ~25x smaller.
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
    const device uint* part_bitmap [[buffer(5)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_id_in_group [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    uint index = group_id * threads_per_group + thread_id_in_group;
    if (index >= partsupp_size) return;
    int pk = ps_partkey[index];

    // Bloom / bitmap pre-filter: skip entries whose partkey didn't pass '%green%'
    if (!bitmap_test(part_bitmap, pk)) return;

    int sk = ps_suppkey[index];
    int val = (int)index;
    // Combined hash of (partkey, suppkey) to reduce probe lengths
    uint mix = (uint)pk * 0x9E3779B1u ^ (uint)sk * 0x85EBCA77u;
    uint ps_mask = partsupp_ht_size - 1;
    uint hash_index = mix & ps_mask;
    for (uint i = 0; i <= ps_mask; ++i) {
        uint probe_index = (hash_index + i) & ps_mask;
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
    uint ord_mask = orders_ht_size - 1;
    uint hash_index = sf100_hash_key(key) & ord_mask;
    for (uint i = 0; i <= ord_mask; ++i) {
        uint probe_index = (hash_index + i) & ord_mask;
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

    const uint global_tid = (group_id * threads_per_group) + thread_id_in_group;
    const uint BATCH = 4;
    const uint stride = grid_size * BATCH;

    for (uint base = global_tid * BATCH; base < lineitem_size; base += stride) {
        for(int k=0; k<BATCH; ++k) {
            uint i = base + k;
            if (i >= lineitem_size) break;

            int partkey = l_partkey[i];

            // 1. Check Bitmap
            if (!bitmap_test(part_bitmap, partkey)) continue;

            int suppkey = l_suppkey[i];

            // 2. Direct Lookup
            int nationkey = supplier_nation_map[suppkey];
            if (nationkey == -1) continue;


            // 3. Probe partsupp_ht to get supply cost index (use combined hash of (partkey,suppkey))
            int ps_idx = -1;
            uint ps_mask = partsupp_ht_size - 1;
            uint ps_hash = ((uint)partkey * 0x9E3779B1u ^ (uint)suppkey * 0x85EBCA77u) & ps_mask;
            for (uint j = 0; j <= ps_mask; ++j) {
                uint probe_idx = (ps_hash + j) & ps_mask;
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
            uint ord_mask = orders_ht_size - 1;
            uint ord_hash = sf100_hash_key(orderkey) & ord_mask;
            for (uint j = 0; j <= ord_mask; ++j) {
                uint probe_idx = (ord_hash + j) & ord_mask;
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
    if (index >= intermediate_size) return;
    Q9Aggregates_Local local_result = intermediate_results[index];
    if (local_result.key == 0) return;

    uint final_mask = final_hashtable_size - 1;
    uint hash_index = local_result.key & final_mask;
    for (uint i = 0; i <= final_mask; ++i) {
        uint probe_index = (hash_index + i) & final_mask;
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
