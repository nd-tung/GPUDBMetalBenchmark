#!/bin/bash
# DuckDB TPC-H Benchmark Script
# Runs Q1, Q3, Q6, Q9, Q13 and records execution times
# Usage: benchmark_duckdb.sh [--query-results] [SF-1|SF-10|SF-100|all]
#   --query-results: save per-query result logs

set -e

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
parse_common_args "$@"

RESULTS_FILE="$RESULTS_DIR/duckdb_results.csv"
LOG_DIR="$RESULTS_DIR/duckdb_logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() { echo -e "${YELLOW}--- $1 ---${NC}"; }

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------
check_prerequisites() {
    if ! command -v duckdb &> /dev/null; then
        echo -e "${RED}Error: DuckDB is not installed.${NC}"
        echo "Install with: brew install duckdb"
        exit 1
    fi
    echo -e "${GREEN}DuckDB found: $(duckdb --version)${NC}"

    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python3 is required for timing.${NC}"
        exit 1
    fi

    if ! command -v jq &> /dev/null; then
        echo -e "${RED}Error: 'jq' is required for DuckDB profiling.${NC}"
        echo "Install with: brew install jq"
        exit 1
    fi
    echo -e "${GREEN}jq found for profiling JSON parsing${NC}"
}

check_data_files() {
    local data_path="$DATA_DIR/$1"
    if [[ ! -d "$data_path" ]]; then
        echo -e "${RED}Error: Data directory not found: $data_path${NC}"
        exit 1
    fi
    for file in lineitem.tbl orders.tbl customer.tbl part.tbl supplier.tbl partsupp.tbl nation.tbl region.tbl; do
        if [[ ! -f "$data_path/$file" ]]; then
            echo -e "${RED}Error: Required data file not found: $data_path/$file${NC}"
            exit 1
        fi
    done
    echo -e "${GREEN}Data files found for $1${NC}"
}

# ---------------------------------------------------------------------------
# Run a query with DuckDB profiler timing
# ---------------------------------------------------------------------------
execute_sql() {
    local query_name="$1"
    local query_sql="$2"
    local scale_factor="$3"
    local db_file="$4"

    echo -e "${GREEN}Running ${query_name}...${NC}"

    local prof_file="/tmp/duckdb_profile_${TIMESTAMP}_${query_name}.json"

    local cmd_output
    cmd_output=$(duckdb "$db_file" << EOF 2>/dev/null
PRAGMA enable_profiling='json';
PRAGMA profiling_output='$prof_file';
$query_sql;
EOF
)

    # Extract timing from profiler JSON
    local exec_ms=""
    if [[ -s "$prof_file" ]]; then
        local secs
        secs=$(jq -r '.latency // empty' "$prof_file")
        if [[ -n "$secs" && "$secs" != "null" ]]; then
            exec_ms=$(python3 -c "print(f'{float(\"$secs\")*1000:.2f}')")
        fi
    fi
    rm -f "$prof_file" >/dev/null 2>&1 || true

    if [[ -z "$exec_ms" ]]; then
        echo -e "${RED}Error: Could not extract timing from DuckDB profiler${NC}"
        exit 1
    fi

    echo "  ✓ Execution time: ${exec_ms} ms"

    # Record to CSV
    echo "$TIMESTAMP,$scale_factor,$query_name,$exec_ms,$GPU_NAME,$MEMORY_GB" >> "$RESULTS_FILE"

    # Save per-query log if requested
    if [[ "$SHOW_QUERY_RESULTS" -eq 1 ]]; then
        local log_file="${LOG_DIR}/${TIMESTAMP}/${scale_factor}_${query_name}.log"
        {
            echo "=== DuckDB ${query_name} Results ==="
            echo "Timestamp: ${TIMESTAMP}"
            echo "Scale Factor: ${scale_factor}"
            echo ""
            echo "${cmd_output}"
            echo ""
            echo "Execution time: ${exec_ms} ms"
        } > "$log_file"
        echo "  ✓ Results saved to: ${log_file}"
    fi
}

# ---------------------------------------------------------------------------
# Load data into DuckDB
# ---------------------------------------------------------------------------
load_data() {
    local scale_factor="$1"
    local db_file="$2"
    local data_path="$DATA_DIR/$scale_factor"

    print_header "Loading $scale_factor Data"

    duckdb "$db_file" << EOF
PRAGMA threads=8;
PRAGMA memory_limit='8GB';
PRAGMA temp_directory='/tmp';
EOF

    local tables=(lineitem orders customer part supplier partsupp nation)
    local schemas=(
        "l_orderkey INTEGER, l_partkey INTEGER, l_suppkey INTEGER, l_linenumber INTEGER, l_quantity DECIMAL(15,2), l_extendedprice DECIMAL(15,2), l_discount DECIMAL(15,2), l_tax DECIMAL(15,2), l_returnflag VARCHAR(1), l_linestatus VARCHAR(1), l_shipdate DATE, l_commitdate DATE, l_receiptdate DATE, l_shipinstruct VARCHAR(25), l_shipmode VARCHAR(10), l_comment VARCHAR(44)"
        "o_orderkey INTEGER, o_custkey INTEGER, o_orderstatus VARCHAR(1), o_totalprice DECIMAL(15,2), o_orderdate DATE, o_orderpriority VARCHAR(15), o_clerk VARCHAR(15), o_shippriority INTEGER, o_comment VARCHAR(79)"
        "c_custkey INTEGER, c_name VARCHAR(25), c_address VARCHAR(40), c_nationkey INTEGER, c_phone VARCHAR(15), c_acctbal DECIMAL(15,2), c_mktsegment VARCHAR(10), c_comment VARCHAR(117)"
        "p_partkey INTEGER, p_name VARCHAR(55), p_mfgr VARCHAR(25), p_brand VARCHAR(10), p_type VARCHAR(25), p_size INTEGER, p_container VARCHAR(10), p_retailprice DECIMAL(15,2), p_comment VARCHAR(23)"
        "s_suppkey INTEGER, s_name VARCHAR(25), s_address VARCHAR(40), s_nationkey INTEGER, s_phone VARCHAR(15), s_acctbal DECIMAL(15,2), s_comment VARCHAR(101)"
        "ps_partkey INTEGER, ps_suppkey INTEGER, ps_availqty INTEGER, ps_supplycost DECIMAL(15,2), ps_comment VARCHAR(199)"
        "n_nationkey INTEGER, n_name VARCHAR(25), n_regionkey INTEGER, n_comment VARCHAR(152)"
    )

    for i in "${!tables[@]}"; do
        local tbl="${tables[$i]}"
        local schema="${schemas[$i]}"
        local start_time end_time load_time
        start_time=$(python3 -c "import time; print(time.time())")
        duckdb "$db_file" << EOF
DROP TABLE IF EXISTS ${tbl};
CREATE TABLE ${tbl} (${schema});
COPY ${tbl} FROM '${data_path}/${tbl}.tbl' (DELIMITER '|', HEADER false);
EOF
        end_time=$(python3 -c "import time; print(time.time())")
        load_time=$(python3 -c "print(round(($end_time - $start_time) * 1000, 2))")
        local row_count
        row_count=$(duckdb "$db_file" -c "SELECT COUNT(*) FROM ${tbl}" 2>/dev/null | tail -1)
        echo "  ${tbl}: $(echo $row_count | tr -d ' ') rows (${load_time} ms)"
    done

    echo -e "${GREEN}Data loaded successfully${NC}"
    echo ""
}

# ---------------------------------------------------------------------------
# Run all queries for a scale factor
# ---------------------------------------------------------------------------
run_benchmarks() {
    local scale_factor="$1"
    local db_file="/tmp/benchmark_${TIMESTAMP}.duckdb"

    check_data_files "$scale_factor"
    load_data "$scale_factor" "$db_file"

    echo "Warming up cache..."
    duckdb "$db_file" -c "SELECT COUNT(*) FROM lineitem; SELECT COUNT(*) FROM orders; SELECT COUNT(*) FROM customer;" >/dev/null 2>&1
    echo ""

    print_header "TPC-H Query 1 ($scale_factor)"
    execute_sql "Q1" "SELECT l_returnflag, l_linestatus, SUM(l_quantity) AS sum_qty, SUM(l_extendedprice) AS sum_base_price, SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price, SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge, AVG(l_quantity) AS avg_qty, AVG(l_extendedprice) AS avg_price, AVG(l_discount) AS avg_disc, COUNT(*) AS count_order FROM lineitem WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90 days' GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus;" "$scale_factor" "$db_file"

    print_header "TPC-H Query 3 ($scale_factor)"
    execute_sql "Q3" "SELECT l_orderkey, SUM(l_extendedprice * (1 - l_discount)) AS revenue, o_orderdate, o_shippriority FROM customer, orders, lineitem WHERE c_mktsegment = 'BUILDING' AND c_custkey = o_custkey AND l_orderkey = o_orderkey AND o_orderdate < DATE '1995-03-15' AND l_shipdate > DATE '1995-03-15' GROUP BY l_orderkey, o_orderdate, o_shippriority ORDER BY revenue DESC, o_orderdate LIMIT 10;" "$scale_factor" "$db_file"

    print_header "TPC-H Query 6 ($scale_factor)"
    execute_sql "Q6" "SELECT SUM(l_extendedprice * l_discount) AS revenue FROM lineitem WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01' AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24;" "$scale_factor" "$db_file"

    print_header "TPC-H Query 9 ($scale_factor)"
    execute_sql "Q9" "SELECT n.n_name AS nation, EXTRACT(YEAR FROM o.o_orderdate) AS o_year, SUM(l.l_extendedprice * (1 - l.l_discount) - ps.ps_supplycost * l.l_quantity) AS sum_profit FROM part p, supplier s, lineitem l, partsupp ps, orders o, nation n WHERE s.s_suppkey = l.l_suppkey AND ps.ps_suppkey = l.l_suppkey AND ps.ps_partkey = l.l_partkey AND p.p_partkey = l.l_partkey AND o.o_orderkey = l.l_orderkey AND s.s_nationkey = n.n_nationkey AND p.p_name LIKE '%green%' GROUP BY nation, o_year ORDER BY nation, o_year DESC;" "$scale_factor" "$db_file"

    print_header "TPC-H Query 13 ($scale_factor)"
    execute_sql "Q13" "SELECT c_count, COUNT(*) AS custdist FROM (SELECT c.c_custkey, COUNT(o.o_orderkey) AS c_count FROM customer c LEFT OUTER JOIN orders o ON c.c_custkey = o.o_custkey AND o.o_comment NOT LIKE '%special%requests%' GROUP BY c.c_custkey) AS c_orders GROUP BY c_count ORDER BY custdist DESC, c_count DESC;" "$scale_factor" "$db_file"

    rm -f "$db_file"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   DuckDB TPC-H Benchmark${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

check_prerequisites
ensure_csv "$RESULTS_FILE" "timestamp,scale_factor,query,execution_time_ms,gpu_name,memory_gb"
[[ "$SHOW_QUERY_RESULTS" -eq 1 ]] && mkdir -p "${LOG_DIR}/${TIMESTAMP}"

MODE="${POSITIONAL_ARGS[0]:-all}"

case "$MODE" in
    sf1|SF-1)     run_benchmarks "SF-1"   ;;
    sf10|SF-10)   run_benchmarks "SF-10"  ;;
    sf100|SF-100) run_benchmarks "SF-100" ;;
    all)
        for sf in SF-1 SF-10 SF-100; do
            [[ -d "$DATA_DIR/$sf" ]] && run_benchmarks "$sf" || true
        done
        ;;
    *)
        echo "Usage: $0 [--query-results] [sf1|sf10|sf100|all]"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Benchmark completed!${NC}"
echo -e "${BLUE}Results saved in: $RESULTS_FILE${NC}"
echo ""
echo "Latest results (${TIMESTAMP}):"
grep "^${TIMESTAMP}" "$RESULTS_FILE" | column -t -s,

if [[ "$SHOW_QUERY_RESULTS" -eq 1 ]]; then
    echo ""
    echo "Log files:"
    ls -lh "${LOG_DIR}/${TIMESTAMP}/"
fi
