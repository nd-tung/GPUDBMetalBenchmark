#!/bin/bash
# DuckDB Verification Script — runs all 22 TPC-H queries and prints results
# for comparison against GPU benchmark output.
# Usage: ./scripts/verify_duckdb.sh [SF-1|SF-10]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
SQL_DIR="$PROJECT_ROOT/sql"

SF="${1:-SF-1}"
DATA_PATH="$DATA_DIR/$SF"

if [[ ! -d "$DATA_PATH" ]]; then
    echo "Error: Data directory $DATA_PATH not found"
    exit 1
fi

DB="/tmp/verify_duckdb_$$.duckdb"
trap "rm -f '$DB'" EXIT

echo "=== DuckDB TPC-H Verification ($SF) ==="
echo ""

# Load schema + data
duckdb "$DB" <<EOF
PRAGMA threads=8;
PRAGMA memory_limit='8GB';

CREATE TABLE nation   (n_nationkey INT, n_name VARCHAR(25), n_regionkey INT, n_comment VARCHAR(152));
CREATE TABLE region   (r_regionkey INT, r_name VARCHAR(25), r_comment VARCHAR(152));
CREATE TABLE part     (p_partkey INT, p_name VARCHAR(55), p_mfgr VARCHAR(25), p_brand VARCHAR(10), p_type VARCHAR(25), p_size INT, p_container VARCHAR(10), p_retailprice DECIMAL(15,2), p_comment VARCHAR(23));
CREATE TABLE supplier (s_suppkey INT, s_name VARCHAR(25), s_address VARCHAR(40), s_nationkey INT, s_phone VARCHAR(15), s_acctbal DECIMAL(15,2), s_comment VARCHAR(101));
CREATE TABLE partsupp (ps_partkey INT, ps_suppkey INT, ps_availqty INT, ps_supplycost DECIMAL(15,2), ps_comment VARCHAR(199));
CREATE TABLE customer (c_custkey INT, c_name VARCHAR(25), c_address VARCHAR(40), c_nationkey INT, c_phone VARCHAR(15), c_acctbal DECIMAL(15,2), c_mktsegment VARCHAR(10), c_comment VARCHAR(117));
CREATE TABLE orders   (o_orderkey INT, o_custkey INT, o_orderstatus VARCHAR(1), o_totalprice DECIMAL(15,2), o_orderdate DATE, o_orderpriority VARCHAR(15), o_clerk VARCHAR(15), o_shippriority INT, o_comment VARCHAR(79));
CREATE TABLE lineitem (l_orderkey INT, l_partkey INT, l_suppkey INT, l_linenumber INT, l_quantity DECIMAL(15,2), l_extendedprice DECIMAL(15,2), l_discount DECIMAL(15,2), l_tax DECIMAL(15,2), l_returnflag VARCHAR(1), l_linestatus VARCHAR(1), l_shipdate DATE, l_commitdate DATE, l_receiptdate DATE, l_shipinstruct VARCHAR(25), l_shipmode VARCHAR(10), l_comment VARCHAR(44));

COPY nation   FROM '$DATA_PATH/nation.tbl'   (DELIMITER '|');
COPY region   FROM '$DATA_PATH/region.tbl'   (DELIMITER '|');
COPY part     FROM '$DATA_PATH/part.tbl'     (DELIMITER '|');
COPY supplier FROM '$DATA_PATH/supplier.tbl' (DELIMITER '|');
COPY partsupp FROM '$DATA_PATH/partsupp.tbl' (DELIMITER '|');
COPY customer FROM '$DATA_PATH/customer.tbl' (DELIMITER '|');
COPY orders   FROM '$DATA_PATH/orders.tbl'   (DELIMITER '|');
COPY lineitem FROM '$DATA_PATH/lineitem.tbl' (DELIMITER '|');

SELECT 'Loaded ' || COUNT(*) || ' lineitem rows' FROM lineitem;
EOF

echo ""

# Run each query
for q_num in $(seq 1 22); do
    sql_file="$SQL_DIR/q${q_num}.sql"
    if [[ ! -f "$sql_file" ]]; then
        echo "--- Q${q_num}: SQL file not found, skipping ---"
        continue
    fi
    echo -n "Q${q_num}: "
    # Strip comment lines
    query_sql=$(grep -v '^--' "$sql_file" | tr '\n' ' ')
    result=$(duckdb "$DB" -c ".mode csv" -c "$query_sql" 2>/dev/null)
    row_count=$(echo "$result" | tail -n +2 | wc -l | tr -d ' ')
    echo "${row_count} rows"
    echo "$result" > "/tmp/duckdb_verify_q${q_num}.csv"
done

echo "=== Verification complete ==="
