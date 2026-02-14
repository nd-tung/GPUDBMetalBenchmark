#!/bin/bash

# DuckDB Benchmark Script - FIXED VERSION
# Proper performance comparison against GPU implementation
# Fixes: Pre-loads data, optimizes settings, measures only query execution

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"
RESULTS_DIR="$PROJECT_ROOT/results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DB_FILE="/tmp/benchmark_${TIMESTAMP}.duckdb"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}   DuckDB vs GPU Database Benchmark${NC}"
echo -e "${BLUE}============================================${NC}"
echo

# Function to print section headers
print_header() {
    echo -e "${YELLOW}--- $1 ---${NC}"
}

# Function to execute DuckDB command with DuckDB internal profiler timing only
execute_sql() {
    local description="$1"
    local sql_command="$2"
    local scale_factor="$3"

    echo -e "${GREEN}$description${NC}"

    # Use DuckDB JSON profiling to capture exec-only timing
    local PROF_FILE="/tmp/duckdb_profile_${TIMESTAMP}_$$.json"

    # Run query with profiling enabled; results printed to stdout; profiling JSON to file
    local cmd_output
    cmd_output=$(duckdb "$DB_FILE" << EOF 2>/dev/null
PRAGMA enable_profiling='json';
PRAGMA profiling_output='$PROF_FILE';
$sql_command;
EOF
)

    # Extract exec-only timing from JSON (seconds -> ms)
    local exec_ms=""
    if command -v jq >/dev/null 2>&1 && [[ -s "$PROF_FILE" ]]; then
        # Extract latency from DuckDB profiler JSON (total execution time)
        local secs
        secs=$(jq -r '.latency // empty' "$PROF_FILE")
        if [[ -n "$secs" && "$secs" != "null" ]]; then
            exec_ms=$(python3 - <<PY
secs = float("$secs")
print(f"{secs*1000:.2f}")
PY
)
        fi
    fi
    rm -f "$PROF_FILE" >/dev/null 2>&1 || true
    
    if [[ -z "$exec_ms" ]]; then
        echo -e "${RED}Error: Could not extract timing from DuckDB profiler${NC}"
        echo -e "${RED}Make sure 'jq' is installed: brew install jq${NC}"
        exit 1
    fi

    echo "  ✓ Execution time: ${exec_ms} ms"

    # Log results: timestamp,scale_factor,benchmark,exec_ms
    echo "$TIMESTAMP,$scale_factor,$description,$exec_ms" >> "$RESULTS_DIR/duckdb_results.csv"
}

# Check if DuckDB is installed
check_duckdb() {
    if ! command -v duckdb &> /dev/null; then
        echo -e "${RED}Error: DuckDB is not installed.${NC}"
        exit 1
    fi
    echo -e "${GREEN}DuckDB found: $(duckdb --version)${NC}"
}

# Check if Python3 is available for high-precision timing
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python3 is required for high-precision timing.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Python3 found for timing${NC}"
}

# Check for jq for JSON parsing (required for DuckDB profiler)
check_jq() {
    if ! command -v jq &> /dev/null; then
        echo -e "${RED}Error: 'jq' is required for DuckDB profiling.${NC}"
        echo -e "${RED}Install with: brew install jq${NC}"
        exit 1
    else
        echo -e "${GREEN}jq found for profiling JSON parsing${NC}"
    fi
}

# Check if data files exist
check_data_files() {
    local scale_factor="$1"
    local data_path="$DATA_DIR/$scale_factor"
    
    if [[ ! -d "$data_path" ]]; then
        echo -e "${RED}Error: Data directory not found: $data_path${NC}"
        exit 1
    fi
    
    local required_files=("lineitem.tbl" "orders.tbl" "customer.tbl" "part.tbl" "supplier.tbl" "partsupp.tbl" "nation.tbl" "region.tbl")
    for file in "${required_files[@]}"; do
        if [[ ! -f "$data_path/$file" ]]; then
            echo -e "${RED}Error: Required data file not found: $data_path/$file${NC}"
            exit 1
        fi
    done
    
    echo -e "${GREEN}Data files found for $scale_factor${NC}"
}

# Create results directory
setup_results() {
    mkdir -p "$RESULTS_DIR"
    if [[ ! -f "$RESULTS_DIR/duckdb_results.csv" ]]; then
        echo "timestamp,scale_factor,benchmark,exec_ms,result" > "$RESULTS_DIR/duckdb_results.csv"
    fi
}

# Initialize DuckDB with optimizations
init_duckdb() {
    echo -e "${BLUE}Initializing DuckDB with optimizations...${NC}"
    
    duckdb "$DB_FILE" << EOF
-- Optimize DuckDB settings for performance (using v1.4.1 compatible settings)
PRAGMA threads=8;
PRAGMA memory_limit='8GB';
PRAGMA temp_directory='/tmp';
EOF
    
    echo -e "${GREEN}DuckDB optimized and ready${NC}"
}

# Load data into memory tables (one-time cost)
load_data() {
    local scale_factor="$1"
    local data_path="$DATA_DIR/$scale_factor"
    
    print_header "Loading $scale_factor Data into Memory"
    
    echo "Loading lineitem table..."
    start_time=$(python3 -c "import time; print(time.time())")
    
    duckdb "$DB_FILE" << EOF
DROP TABLE IF EXISTS lineitem;
CREATE TABLE lineitem AS 
SELECT * FROM read_csv_auto('$data_path/lineitem.tbl', 
    delim='|', 
    header=false,
    columns = {
        'l_orderkey': 'INTEGER',
        'l_partkey': 'INTEGER', 
        'l_suppkey': 'INTEGER',
        'l_linenumber': 'INTEGER',
        'l_quantity': 'DECIMAL(10,2)',
        'l_extendedprice': 'DECIMAL(10,2)',
        'l_discount': 'DECIMAL(10,2)',
        'l_tax': 'DECIMAL(10,2)',
        'l_returnflag': 'CHAR(1)',
        'l_linestatus': 'CHAR(1)',
        'l_shipdate': 'DATE',
        'l_commitdate': 'DATE',
        'l_receiptdate': 'DATE',
        'l_shipinstruct': 'VARCHAR(25)',
        'l_shipmode': 'VARCHAR(10)',
        'l_comment': 'VARCHAR(44)'
    }
);
EOF
    
    end_time=$(python3 -c "import time; print(time.time())")
    lineitem_load_time=$(python3 -c "print(round(($end_time - $start_time) * 1000, 2))")
    
    echo "Loading orders table..."
    start_time=$(python3 -c "import time; print(time.time())")
    
    duckdb "$DB_FILE" << EOF
DROP TABLE IF EXISTS orders;
CREATE TABLE orders AS 
SELECT * FROM read_csv_auto('$data_path/orders.tbl', 
    delim='|', 
    header=false,
    columns = {
        'o_orderkey': 'INTEGER',
        'o_custkey': 'INTEGER',
        'o_orderstatus': 'CHAR(1)',
        'o_totalprice': 'DECIMAL(10,2)',
        'o_orderdate': 'DATE',
        'o_orderpriority': 'VARCHAR(15)',
        'o_clerk': 'VARCHAR(15)',
        'o_shippriority': 'INTEGER',
        'o_comment': 'VARCHAR(79)'
    }
);
EOF
    
    end_time=$(python3 -c "import time; print(time.time())")
    orders_load_time=$(python3 -c "print(round(($end_time - $start_time) * 1000, 2))")
    
    echo "Loading customer table..."
    start_time=$(python3 -c "import time; print(time.time())")
    
    duckdb "$DB_FILE" << EOF
DROP TABLE IF EXISTS customer;
CREATE TABLE customer AS 
SELECT * FROM read_csv_auto('$data_path/customer.tbl', 
    delim='|', 
    header=false,
    columns = {
        'c_custkey': 'INTEGER',
        'c_name': 'VARCHAR(25)',
        'c_address': 'VARCHAR(40)',
        'c_nationkey': 'INTEGER',
        'c_phone': 'VARCHAR(15)',
        'c_acctbal': 'DECIMAL(10,2)',
        'c_mktsegment': 'VARCHAR(10)',
        'c_comment': 'VARCHAR(117)'
    }
);
EOF
    
    end_time=$(python3 -c "import time; print(time.time())")
    customer_load_time=$(python3 -c "print(round(($end_time - $start_time) * 1000, 2))")
    
    echo "Loading part table..."
    start_time=$(python3 -c "import time; print(time.time())")
    
    duckdb "$DB_FILE" << EOF
DROP TABLE IF EXISTS part;
CREATE TABLE part AS 
SELECT * FROM read_csv_auto('$data_path/part.tbl', 
    delim='|', 
    header=false,
    columns = {
        'p_partkey': 'INTEGER',
        'p_name': 'VARCHAR(55)',
        'p_mfgr': 'VARCHAR(25)',
        'p_brand': 'VARCHAR(10)',
        'p_type': 'VARCHAR(25)',
        'p_size': 'INTEGER',
        'p_container': 'VARCHAR(10)',
        'p_retailprice': 'DECIMAL(10,2)',
        'p_comment': 'VARCHAR(23)'
    }
);
EOF
    
    end_time=$(python3 -c "import time; print(time.time())")
    part_load_time=$(python3 -c "print(round(($end_time - $start_time) * 1000, 2))")
    
    echo "Loading supplier table..."
    start_time=$(python3 -c "import time; print(time.time())")
    
    duckdb "$DB_FILE" << EOF
DROP TABLE IF EXISTS supplier;
CREATE TABLE supplier AS 
SELECT * FROM read_csv_auto('$data_path/supplier.tbl', 
    delim='|', 
    header=false,
    columns = {
        's_suppkey': 'INTEGER',
        's_name': 'VARCHAR(25)',
        's_address': 'VARCHAR(40)',
        's_nationkey': 'INTEGER',
        's_phone': 'VARCHAR(15)',
        's_acctbal': 'DECIMAL(10,2)',
        's_comment': 'VARCHAR(101)'
    }
);
EOF
    
    end_time=$(python3 -c "import time; print(time.time())")
    supplier_load_time=$(python3 -c "print(round(($end_time - $start_time) * 1000, 2))")

    echo "Loading partsupp table..."
    start_time=$(python3 -c "import time; print(time.time())")
    
    duckdb "$DB_FILE" << EOF
DROP TABLE IF EXISTS partsupp;
CREATE TABLE partsupp AS 
SELECT * FROM read_csv_auto('$data_path/partsupp.tbl', 
    delim='|', 
    header=false,
    columns = {
        'ps_partkey': 'INTEGER',
        'ps_suppkey': 'INTEGER',
        'ps_availqty': 'INTEGER',
        'ps_supplycost': 'DECIMAL(10,2)',
        'ps_comment': 'VARCHAR(199)'
    }
);
EOF
    
    end_time=$(python3 -c "import time; print(time.time())")
    partsupp_load_time=$(python3 -c "print(round(($end_time - $start_time) * 1000, 2))")

    echo "Loading nation table..."
    start_time=$(python3 -c "import time; print(time.time())")
    
    duckdb "$DB_FILE" << EOF
DROP TABLE IF EXISTS nation;
CREATE TABLE nation AS 
SELECT * FROM read_csv_auto('$data_path/nation.tbl', 
    delim='|', 
    header=false,
    columns = {
        'n_nationkey': 'INTEGER',
        'n_name': 'VARCHAR(25)',
        'n_regionkey': 'INTEGER',
        'n_comment': 'VARCHAR(152)'
    }
);
EOF
    
    end_time=$(python3 -c "import time; print(time.time())")
    nation_load_time=$(python3 -c "print(round(($end_time - $start_time) * 1000, 2))")
    
    # Get statistics
    lineitem_count=$(duckdb "$DB_FILE" -c "SELECT COUNT(*) FROM lineitem" 2>/dev/null | tail -1)
    orders_count=$(duckdb "$DB_FILE" -c "SELECT COUNT(*) FROM orders" 2>/dev/null | tail -1)
    customer_count=$(duckdb "$DB_FILE" -c "SELECT COUNT(*) FROM customer" 2>/dev/null | tail -1)
    part_count=$(duckdb "$DB_FILE" -c "SELECT COUNT(*) FROM part" 2>/dev/null | tail -1)
    supplier_count=$(duckdb "$DB_FILE" -c "SELECT COUNT(*) FROM supplier" 2>/dev/null | tail -1)
    
    echo -e "${GREEN}Data loaded successfully${NC}"
    echo "  Lineitem: $lineitem_count rows (loaded in ${lineitem_load_time}ms)"
    echo "  Orders: $orders_count rows (loaded in ${orders_load_time}ms)"
    echo "  Customer: $customer_count rows (loaded in ${customer_load_time}ms)"
    echo "  Part: $part_count rows (loaded in ${part_load_time}ms)"
    echo "  Supplier: $supplier_count rows (loaded in ${supplier_load_time}ms)"
    echo "  Partsupp: $(duckdb "$DB_FILE" -c "SELECT COUNT(*) FROM partsupp" 2>/dev/null | tail -1) rows (loaded in ${partsupp_load_time}ms)"
    echo "  Nation: $(duckdb "$DB_FILE" -c "SELECT COUNT(*) FROM nation" 2>/dev/null | tail -1) rows (loaded in ${nation_load_time}ms)"
    echo
}

# Run benchmarks for a specific scale factor
run_benchmarks() {
    local scale_factor="$1"
    
    print_header "Running $scale_factor Benchmarks (Query Execution Only)"
    
    check_data_files "$scale_factor"
    load_data "$scale_factor"
    
    # 4. TPC-H Query 1 Benchmark
    print_header "TPC-H Query 1 Benchmark ($scale_factor)"
    
    execute_sql "Q1" \
        "SELECT
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
        ORDER BY l_returnflag, l_linestatus;" \
        "$scale_factor"
    
    # 5. TPC-H Query 3 Benchmark (Shipping Priority Query)
    print_header "TPC-H Query 3 Benchmark ($scale_factor)"
    
    execute_sql "Q3" \
        "SELECT
            l.l_orderkey,
            SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
            o.o_orderdate,
            o.o_shippriority
        FROM customer c
        JOIN orders o ON c.c_custkey = o.o_custkey
        JOIN lineitem l ON l.l_orderkey = o.o_orderkey
        WHERE c.c_mktsegment = 'BUILDING'
          AND o.o_orderdate < DATE '1995-03-15'
          AND l.l_shipdate > DATE '1995-03-15'
        GROUP BY l.l_orderkey, o.o_orderdate, o.o_shippriority
        ORDER BY revenue DESC, o.o_orderdate
        LIMIT 10;" \
        "$scale_factor"
    
    # 6. TPC-H Query 6 Benchmark (Forecasting Revenue Change Query)
    print_header "TPC-H Query 6 Benchmark ($scale_factor)"
    
    execute_sql "Q6" \
        "SELECT
            SUM(l_extendedprice * l_discount) AS revenue
        FROM lineitem
        WHERE l_shipdate >= DATE '1994-01-01'
          AND l_shipdate < DATE '1995-01-01'
          AND l_discount BETWEEN 0.05 AND 0.07
          AND l_quantity < 24;" \
        "$scale_factor"
    
    # 7. TPC-H Query 9 Benchmark - Product Type Profit Measure
    print_header "TPC-H Query 9 Benchmark ($scale_factor)"
    
    execute_sql "Q9" \
        "SELECT 
            n.n_name AS nation,
            EXTRACT(YEAR FROM o.o_orderdate) AS o_year,
            SUM(l.l_extendedprice * (1 - l.l_discount) - ps.ps_supplycost * l.l_quantity) AS sum_profit
        FROM part p, supplier s, lineitem l, partsupp ps, orders o, nation n
        WHERE s.s_suppkey = l.l_suppkey
          AND ps.ps_suppkey = l.l_suppkey
          AND ps.ps_partkey = l.l_partkey
          AND p.p_partkey = l.l_partkey
          AND o.o_orderkey = l.l_orderkey
          AND s.s_nationkey = n.n_nationkey
          AND p.p_name LIKE '%green%'
        GROUP BY nation, o_year
        ORDER BY nation, o_year DESC;" \
        "$scale_factor"
    
    # 8. TPC-H Query 13 Benchmark (Customer Distribution Query)
    print_header "TPC-H Query 13 Benchmark ($scale_factor)"
    
    execute_sql "Q13" \
        "SELECT
            c_count,
            COUNT(*) AS custdist
        FROM (
            SELECT
                c.c_custkey,
                COUNT(o.o_orderkey) AS c_count
            FROM customer c
            LEFT OUTER JOIN orders o
              ON c.c_custkey = o.o_custkey
             AND o.o_comment NOT LIKE '%special%requests%'
            GROUP BY c.c_custkey
        ) AS c_orders
        GROUP BY c_count
        ORDER BY custdist DESC, c_count DESC;" \
        "$scale_factor"
}

# Cleanup function
cleanup() {
    rm -f "$DB_FILE"
}

# Main execution
main() {
    local scale_factors=("SF-1" "SF-10" "SF-100")
    
    # Handle command line arguments
    if [[ $# -gt 0 ]]; then
        scale_factors=("$@")
    fi
    
    echo -e "${BLUE}Starting DuckDB benchmark...${NC}"
    echo "Scale factors to test: ${scale_factors[*]}"
    echo
    
    check_duckdb
    check_python
    check_jq
    setup_results
    init_duckdb
    
    # Run benchmarks for each scale factor
    for sf in "${scale_factors[@]}"; do
        if [[ -d "$DATA_DIR/$sf" ]]; then
            run_benchmarks "$sf"
        else
            echo -e "${YELLOW}Warning: Skipping $sf (directory not found)${NC}"
        fi
    done
    
    cleanup
    
    echo -e "${GREEN}Benchmark completed!${NC}"
    echo -e "${BLUE}Results saved in: $RESULTS_DIR/duckdb_results.csv${NC}"
}

# Set trap for cleanup
trap cleanup EXIT

# Run main function
main "$@"