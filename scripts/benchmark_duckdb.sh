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

    local tables=(nation region part supplier partsupp customer orders lineitem)
    local schemas=(
        "n_nationkey INTEGER, n_name VARCHAR(25), n_regionkey INTEGER, n_comment VARCHAR(152)"
        "r_regionkey INTEGER, r_name VARCHAR(25), r_comment VARCHAR(152)"
        "p_partkey INTEGER, p_name VARCHAR(55), p_mfgr VARCHAR(25), p_brand VARCHAR(10), p_type VARCHAR(25), p_size INTEGER, p_container VARCHAR(10), p_retailprice DECIMAL(15,2), p_comment VARCHAR(23)"
        "s_suppkey INTEGER, s_name VARCHAR(25), s_address VARCHAR(40), s_nationkey INTEGER, s_phone VARCHAR(15), s_acctbal DECIMAL(15,2), s_comment VARCHAR(101)"
        "ps_partkey INTEGER, ps_suppkey INTEGER, ps_availqty INTEGER, ps_supplycost DECIMAL(15,2), ps_comment VARCHAR(199)"
        "c_custkey INTEGER, c_name VARCHAR(25), c_address VARCHAR(40), c_nationkey INTEGER, c_phone VARCHAR(15), c_acctbal DECIMAL(15,2), c_mktsegment VARCHAR(10), c_comment VARCHAR(117)"
        "o_orderkey INTEGER, o_custkey INTEGER, o_orderstatus VARCHAR(1), o_totalprice DECIMAL(15,2), o_orderdate DATE, o_orderpriority VARCHAR(15), o_clerk VARCHAR(15), o_shippriority INTEGER, o_comment VARCHAR(79)"
        "l_orderkey INTEGER, l_partkey INTEGER, l_suppkey INTEGER, l_linenumber INTEGER, l_quantity DECIMAL(15,2), l_extendedprice DECIMAL(15,2), l_discount DECIMAL(15,2), l_tax DECIMAL(15,2), l_returnflag VARCHAR(1), l_linestatus VARCHAR(1), l_shipdate DATE, l_commitdate DATE, l_receiptdate DATE, l_shipinstruct VARCHAR(25), l_shipmode VARCHAR(10), l_comment VARCHAR(44)"
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

    # Run all 22 TPC-H queries from sql/ directory
    for q_num in $(seq 1 22); do
        local sql_file="$PROJECT_ROOT/sql/q${q_num}.sql"
        if [[ ! -f "$sql_file" ]]; then
            echo -e "${RED}Warning: $sql_file not found, skipping Q${q_num}${NC}"
            continue
        fi
        local query_sql
        query_sql=$(grep -v '^--' "$sql_file" | tr '\n' ' ')
        print_header "TPC-H Query ${q_num} ($scale_factor)"
        execute_sql "Q${q_num}" "$query_sql" "$scale_factor" "$db_file"
    done

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
