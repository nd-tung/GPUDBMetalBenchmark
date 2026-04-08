#!/bin/bash
# Run all 22 TPC-H queries with DuckDB for given scale factors
# Output: one line per query with timing in ms
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

run_sf() {
    local sf="$1"
    local data_path="$PROJECT_ROOT/data/$sf"
    local db_file="/tmp/duckdb_bench_${sf}.duckdb"

    rm -f "$db_file"

    echo "=== DuckDB $sf ===" >&2

    # Load data
    duckdb "$db_file" <<EOF
PRAGMA threads=$(sysctl -n hw.ncpu 2>/dev/null || echo 8);
PRAGMA memory_limit='8GB';

CREATE TABLE nation   (n_nationkey INTEGER, n_name VARCHAR(25), n_regionkey INTEGER, n_comment VARCHAR(152));
CREATE TABLE region   (r_regionkey INTEGER, r_name VARCHAR(25), r_comment VARCHAR(152));
CREATE TABLE part     (p_partkey INTEGER, p_name VARCHAR(55), p_mfgr VARCHAR(25), p_brand VARCHAR(10), p_type VARCHAR(25), p_size INTEGER, p_container VARCHAR(10), p_retailprice DECIMAL(15,2), p_comment VARCHAR(23));
CREATE TABLE supplier (s_suppkey INTEGER, s_name VARCHAR(25), s_address VARCHAR(40), s_nationkey INTEGER, s_phone VARCHAR(15), s_acctbal DECIMAL(15,2), s_comment VARCHAR(101));
CREATE TABLE partsupp (ps_partkey INTEGER, ps_suppkey INTEGER, ps_availqty INTEGER, ps_supplycost DECIMAL(15,2), ps_comment VARCHAR(199));
CREATE TABLE customer (c_custkey INTEGER, c_name VARCHAR(25), c_address VARCHAR(40), c_nationkey INTEGER, c_phone VARCHAR(15), c_acctbal DECIMAL(15,2), c_mktsegment VARCHAR(10), c_comment VARCHAR(117));
CREATE TABLE orders   (o_orderkey INTEGER, o_custkey INTEGER, o_orderstatus VARCHAR(1), o_totalprice DECIMAL(15,2), o_orderdate DATE, o_orderpriority VARCHAR(15), o_clerk VARCHAR(15), o_shippriority INTEGER, o_comment VARCHAR(79));
CREATE TABLE lineitem (l_orderkey INTEGER, l_partkey INTEGER, l_suppkey INTEGER, l_linenumber INTEGER, l_quantity DECIMAL(15,2), l_extendedprice DECIMAL(15,2), l_discount DECIMAL(15,2), l_tax DECIMAL(15,2), l_returnflag VARCHAR(1), l_linestatus VARCHAR(1), l_shipdate DATE, l_commitdate DATE, l_receiptdate DATE, l_shipinstruct VARCHAR(25), l_shipmode VARCHAR(10), l_comment VARCHAR(44));

COPY nation   FROM '${data_path}/nation.tbl'   (DELIMITER '|');
COPY region   FROM '${data_path}/region.tbl'   (DELIMITER '|');
COPY part     FROM '${data_path}/part.tbl'     (DELIMITER '|');
COPY supplier FROM '${data_path}/supplier.tbl' (DELIMITER '|');
COPY partsupp FROM '${data_path}/partsupp.tbl' (DELIMITER '|');
COPY customer FROM '${data_path}/customer.tbl' (DELIMITER '|');
COPY orders   FROM '${data_path}/orders.tbl'   (DELIMITER '|');
COPY lineitem FROM '${data_path}/lineitem.tbl' (DELIMITER '|');
EOF
    echo "Data loaded for $sf" >&2

    # Warmup
    duckdb "$db_file" -c "SELECT COUNT(*) FROM lineitem; SELECT COUNT(*) FROM orders;" >/dev/null 2>&1

    # Run each query
    for q_num in $(seq 1 22); do
        local sql_file="$PROJECT_ROOT/sql/q${q_num}.sql"
        if [[ ! -f "$sql_file" ]]; then
            printf "Q%-2d: N/A\n" "$q_num"
            continue
        fi
        local query_sql
        query_sql=$(grep -v '^--' "$sql_file" | tr '\n' ' ')

        local prof_file="/tmp/duckdb_prof_${sf}_q${q_num}.json"
        duckdb "$db_file" <<EOF >/dev/null 2>&1
PRAGMA enable_profiling='json';
PRAGMA profiling_output='${prof_file}';
${query_sql}
EOF
        local exec_ms=""
        if [[ -s "$prof_file" ]]; then
            local secs
            secs=$(python3 -c "import json; d=json.load(open('${prof_file}')); print(d.get('latency',''))" 2>/dev/null)
            if [[ -n "$secs" && "$secs" != "" ]]; then
                exec_ms=$(python3 -c "print(f'{float(\"$secs\")*1000:.2f}')")
            fi
        fi
        rm -f "$prof_file" 2>/dev/null
        printf "Q%-2d: %s\n" "$q_num" "${exec_ms:-N/A}"
    done

    rm -f "$db_file"
}

for sf in "$@"; do
    run_sf "$sf"
done
