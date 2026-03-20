#include "common.h"

// ===================================================================
// TPC-H Q3 KERNELS — Shipping Priority Query
// ===================================================================
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

// Packed hash table entry for orders join (used when direct map too large).
// key == -1 indicates an empty slot.
struct Q3OrdersHTEntry {
    int key;          // orderkey, -1 = empty
    int custkey;
    uint orderdate;
    uint shippriority;
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
        bitmap_set(customer_bitmap, c_custkey[index]);
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
    const device int* o_custkey [[buffer(5)]],
    const device uint* customer_bitmap [[buffer(6)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= orders_size) return;
    if (o_orderdate[index] >= cutoff_date) return;

    // Bitmap pre-filter: only insert orders whose customer is in 'BUILDING' segment
    int ck = o_custkey[index];
    if (!bitmap_test(customer_bitmap, ck)) return;

    int key = o_orderkey[index];
    orders_map[key] = (int)index;
}

// KERNEL 2b: Build a HASH TABLE on the ORDERS table.
// Used when direct-map array is too large (e.g. SF100 where max_orderkey ~600M).
// Open-addressing with linear probing. Power-of-2 capacity.
kernel void q3_build_orders_ht_kernel(
    const device int* o_orderkey       [[buffer(0)]],
    const device int* o_custkey        [[buffer(1)]],
    const device int* o_orderdate      [[buffer(2)]],
    const device int* o_shippriority   [[buffer(3)]],
    device Q3OrdersHTEntry* ht         [[buffer(4)]],
    constant uint& orders_size         [[buffer(5)]],
    constant int& cutoff_date          [[buffer(6)]],
    constant uint& ht_capacity         [[buffer(7)]],
    const device uint* customer_bitmap [[buffer(8)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= orders_size) return;
    if (o_orderdate[index] >= cutoff_date) return;

    // Bitmap pre-filter: skip orders whose customer is not in 'BUILDING' segment
    int ck = o_custkey[index];
    if (!bitmap_test(customer_bitmap, ck)) return;

    int key = o_orderkey[index];
    uint h = sf100_hash_key(key);

    for (uint i = 0; i < ht_capacity; i++) {
        uint slot = (h + i) & (ht_capacity - 1);
        int expected = -1;
        if (atomic_compare_exchange_weak_explicit(
                (device atomic_int*)&ht[slot].key, &expected, key,
                memory_order_relaxed, memory_order_relaxed)) {
            ht[slot].custkey = o_custkey[index];
            ht[slot].orderdate = (uint)o_orderdate[index];
            ht[slot].shippriority = (uint)o_shippriority[index];
            return;
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
    const device int* orders_map        [[buffer(4)]],
    const device int* o_custkey         [[buffer(5)]],
    const device int* o_orderdate       [[buffer(6)]],
    const device int* o_shippriority    [[buffer(7)]],
    device Q3Aggregates* final_ht       [[buffer(8)]],
    constant uint& lineitem_size        [[buffer(9)]],
    constant int& cutoff_date           [[buffer(10)]],
    constant uint& final_ht_size        [[buffer(11)]],
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

        // Customer bitmap check skipped — orders map already filtered at build time

        // Direct aggregation into final hash table
        for (int k = 0; k < BATCH_SIZE; k++) {
            if (pass_order[k]) {
                float revenue = l_extendedprice[idx[k]] * (1.0f - l_discount[idx[k]]);
                int key = l_orderkey_val[k];
                uint ht_mask = final_ht_size - 1;
                uint hash = (uint)key & ht_mask;

                for (uint p = 0; p <= ht_mask; p++) {
                    uint probe_idx = (hash + p) & ht_mask;
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

// =============================================================
// Fused probe + aggregate kernel using HASH TABLE for orders lookup.
// Used when direct-map array is too large. Same batched aggregation
// logic as q3_probe_and_aggregate_direct_kernel but probes a compact
// open-addressing hash table instead of dense orders_map[].
// =============================================================
kernel void q3_probe_and_aggregate_ht_kernel(
    const device int* l_orderkey        [[buffer(0)]],
    const device int* l_shipdate        [[buffer(1)]],
    const device float* l_extendedprice [[buffer(2)]],
    const device float* l_discount      [[buffer(3)]],
    const device Q3OrdersHTEntry* orders_ht [[buffer(4)]],
    device Q3Aggregates* final_ht       [[buffer(5)]],
    constant uint& lineitem_size        [[buffer(6)]],
    constant int& cutoff_date           [[buffer(7)]],
    constant uint& ht_capacity          [[buffer(8)]],
    constant uint& final_ht_size        [[buffer(9)]],
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

        // Probe Orders Hash Table
        uint orderdate_val[BATCH_SIZE];
        uint shippriority_val[BATCH_SIZE];
        bool pass_order[BATCH_SIZE];

        for (int k = 0; k < BATCH_SIZE; k++) {
            pass_order[k] = false;
            if (pass_date[k]) {
                uint h = sf100_hash_key(l_orderkey_val[k]);
                for (uint j = 0; j < ht_capacity; j++) {
                    uint slot = (h + j) & (ht_capacity - 1);
                    int sk = orders_ht[slot].key;
                    if (sk == -1) break;
                    if (sk == l_orderkey_val[k]) {
                        orderdate_val[k] = orders_ht[slot].orderdate;
                        shippriority_val[k] = orders_ht[slot].shippriority;
                        pass_order[k] = true;
                        break;
                    }
                }
            }
        }

        // Customer bitmap check skipped — orders HT already filtered at build time

        // Direct aggregation into final hash table
        for (int k = 0; k < BATCH_SIZE; k++) {
            if (pass_order[k]) {
                float revenue = l_extendedprice[idx[k]] * (1.0f - l_discount[idx[k]]);
                int key = l_orderkey_val[k];
                uint ht_mask = final_ht_size - 1;
                uint hash = (uint)key & ht_mask;

                for (uint p = 0; p <= ht_mask; p++) {
                    uint probe_idx = (hash + p) & ht_mask;
                    int expected = 0;
                    if (atomic_compare_exchange_weak_explicit(
                            &final_ht[probe_idx].key, &expected, key,
                            memory_order_relaxed, memory_order_relaxed)) {
                        atomic_store_explicit(&final_ht[probe_idx].orderdate,
                                             orderdate_val[k], memory_order_relaxed);
                        atomic_store_explicit(&final_ht[probe_idx].shippriority,
                                             shippriority_val[k], memory_order_relaxed);
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
