#!/bin/bash
# DuckDB Benchmark Script with Results Export
# Runs TPC-H queries Q1, Q3, Q6, Q9, Q13 and records both execution times and results

set -e

# Parse command line arguments
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SCALE_FACTOR=${1:-SF-1}
DATA_DIR="$PROJECT_ROOT/data/${SCALE_FACTOR}"
RESULTS_FILE="$PROJECT_ROOT/results/duckdb_results.csv"
LOG_DIR="$PROJECT_ROOT/results/duckdb_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DB_FILE="/tmp/duckdb_benchmark_${TIMESTAMP}.duckdb"

# Detect GPU name and system memory
GPU_NAME=$(system_profiler SPDisplaysDataType 2>/dev/null | awk -F': ' '/Chipset Model/{print $2}' | head -1 | xargs)
MEMORY_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))

echo "=== DuckDB TPC-H Benchmark with Results Export ==="
echo "Scale Factor: ${SCALE_FACTOR}"
echo "Data Directory: ${DATA_DIR}"
echo "Results CSV: ${RESULTS_FILE}"
echo "Results Logs: ${LOG_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo "Database: ${DB_FILE}"
echo ""

# Create results and log directories if they don't exist
mkdir -p benchmark_results
mkdir -p "${LOG_DIR}/${TIMESTAMP}"

# Initialize CSV file with header if it doesn't exist
if [ ! -f "${RESULTS_FILE}" ]; then
    echo "timestamp,scale_factor,query,execution_time_ms,gpu_name,memory_gb" > "${RESULTS_FILE}"
fi

# Check if DuckDB is installed
if ! command -v duckdb &> /dev/null; then
    echo "Error: DuckDB is not installed."
    echo "Install with: brew install duckdb"
    exit 1
fi

# Check if jq is installed (required for DuckDB profiler)
if ! command -v jq &> /dev/null; then
    echo "Error: 'jq' is required for DuckDB profiling."
    echo "Install with: brew install jq"
    exit 1
fi

echo "DuckDB found: $(duckdb --version)"
echo "jq found for profiling JSON parsing"
echo ""

# Function to run a query, extract DuckDB internal timing, and save results
run_query_with_results() {
    local query_name=$1
    local query_sql=$2
    local log_file="${LOG_DIR}/${TIMESTAMP}/${SCALE_FACTOR}_${query_name}.log"
    local prof_file="/tmp/duckdb_profile_${TIMESTAMP}_${query_name}.json"
    
    echo "Running ${query_name}..."
    
    # Run query with DuckDB profiling enabled
    local output
    output=$(duckdb "$DB_FILE" << EOF 2>&1
PRAGMA enable_profiling='json';
PRAGMA profiling_output='$prof_file';
$query_sql;
EOF
)
    
    # Extract timing from DuckDB profiler JSON
    local exec_time=""
    if command -v jq >/dev/null 2>&1 && [[ -s "$prof_file" ]]; then
        local secs
        secs=$(jq -r '.latency // empty' "$prof_file")
        if [[ -n "$secs" && "$secs" != "null" ]]; then
            exec_time=$(python3 -c "print(int(float('$secs') * 1000))")
        fi
    fi
    rm -f "$prof_file" >/dev/null 2>&1 || true
    
    if [[ -z "$exec_time" ]]; then
        echo "  ⚠️  Warning: Could not extract DuckDB profiler timing for ${query_name}"
        echo "  Make sure 'jq' is installed: brew install jq"
        exec_time="0"
    fi
    
    # Save full output to log file
    echo "=== DuckDB ${query_name} Results ===" > "${log_file}"
    echo "Timestamp: ${TIMESTAMP}" >> "${log_file}"
    echo "Scale Factor: ${SCALE_FACTOR}" >> "${log_file}"
    echo "Query: ${query_name}" >> "${log_file}"
    echo "" >> "${log_file}"
    echo "${output}" >> "${log_file}"
    echo "" >> "${log_file}"
    echo "Execution time: ${exec_time} ms" >> "${log_file}"
    
    echo "  ✓ Execution time: ${exec_time} ms"
    echo "  ✓ Results saved to: ${log_file}"
    
    # Append timing to CSV
    echo "${TIMESTAMP},${SCALE_FACTOR},${query_name},${exec_time},${GPU_NAME},${MEMORY_GB}" >> "${RESULTS_FILE}"
}

# Load data into DuckDB
echo "Loading TPC-H data for ${SCALE_FACTOR}..."

duckdb "$DB_FILE" << EOF
-- Create tables
CREATE TABLE lineitem (
    l_orderkey INTEGER,
    l_partkey INTEGER,
    l_suppkey INTEGER,
    l_linenumber INTEGER,
    l_quantity DECIMAL(15,2),
    l_extendedprice DECIMAL(15,2),
    l_discount DECIMAL(15,2),
    l_tax DECIMAL(15,2),
    l_returnflag VARCHAR(1),
    l_linestatus VARCHAR(1),
    l_shipdate DATE,
    l_commitdate DATE,
    l_receiptdate DATE,
    l_shipinstruct VARCHAR(25),
    l_shipmode VARCHAR(10),
    l_comment VARCHAR(44)
);

CREATE TABLE orders (
    o_orderkey INTEGER,
    o_custkey INTEGER,
    o_orderstatus VARCHAR(1),
    o_totalprice DECIMAL(15,2),
    o_orderdate DATE,
    o_orderpriority VARCHAR(15),
    o_clerk VARCHAR(15),
    o_shippriority INTEGER,
    o_comment VARCHAR(79)
);

CREATE TABLE customer (
    c_custkey INTEGER,
    c_name VARCHAR(25),
    c_address VARCHAR(40),
    c_nationkey INTEGER,
    c_phone VARCHAR(15),
    c_acctbal DECIMAL(15,2),
    c_mktsegment VARCHAR(10),
    c_comment VARCHAR(117)
);

CREATE TABLE part (
    p_partkey INTEGER,
    p_name VARCHAR(55),
    p_mfgr VARCHAR(25),
    p_brand VARCHAR(10),
    p_type VARCHAR(25),
    p_size INTEGER,
    p_container VARCHAR(10),
    p_retailprice DECIMAL(15,2),
    p_comment VARCHAR(23)
);

CREATE TABLE supplier (
    s_suppkey INTEGER,
    s_name VARCHAR(25),
    s_address VARCHAR(40),
    s_nationkey INTEGER,
    s_phone VARCHAR(15),
    s_acctbal DECIMAL(15,2),
    s_comment VARCHAR(101)
);

CREATE TABLE partsupp (
    ps_partkey INTEGER,
    ps_suppkey INTEGER,
    ps_availqty INTEGER,
    ps_supplycost DECIMAL(15,2),
    ps_comment VARCHAR(199)
);

CREATE TABLE nation (
    n_nationkey INTEGER,
    n_name VARCHAR(25),
    n_regionkey INTEGER,
    n_comment VARCHAR(152)
);

-- Load data
COPY lineitem FROM '${DATA_DIR}/lineitem.tbl' (DELIMITER '|', HEADER false);
COPY orders FROM '${DATA_DIR}/orders.tbl' (DELIMITER '|', HEADER false);
COPY customer FROM '${DATA_DIR}/customer.tbl' (DELIMITER '|', HEADER false);
COPY part FROM '${DATA_DIR}/part.tbl' (DELIMITER '|', HEADER false);
COPY supplier FROM '${DATA_DIR}/supplier.tbl' (DELIMITER '|', HEADER false);
COPY partsupp FROM '${DATA_DIR}/partsupp.tbl' (DELIMITER '|', HEADER false);
COPY nation FROM '${DATA_DIR}/nation.tbl' (DELIMITER '|', HEADER false);
EOF

echo "Data loaded successfully"
echo ""

echo "Warming up cache (like GPU and CedarDB approach)..."
duckdb "$DB_FILE" -c "SELECT COUNT(*) FROM lineitem; SELECT COUNT(*) FROM orders; SELECT COUNT(*) FROM customer;" >/dev/null 2>&1
echo "Cache warmed up"
echo ""

echo "Starting benchmark runs with results export..."
echo ""

# Q1: Pricing Summary Report
Q1_SQL="SELECT 
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
WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90 days'
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus"

run_query_with_results "Q1" "$Q1_SQL"

# Q3: Shipping Priority
Q3_SQL="SELECT 
    l_orderkey,
    SUM(l_extendedprice * (1 - l_discount)) AS revenue,
    o_orderdate,
    o_shippriority
FROM customer, orders, lineitem
WHERE c_mktsegment = 'BUILDING'
  AND c_custkey = o_custkey
  AND l_orderkey = o_orderkey
  AND o_orderdate < DATE '1995-03-15'
  AND l_shipdate > DATE '1995-03-15'
GROUP BY l_orderkey, o_orderdate, o_shippriority
ORDER BY revenue DESC, o_orderdate
LIMIT 10"

run_query_with_results "Q3" "$Q3_SQL"

# Q6: Forecasting Revenue Change
Q6_SQL="SELECT SUM(l_extendedprice * l_discount) AS revenue
FROM lineitem
WHERE l_shipdate >= DATE '1994-01-01'
  AND l_shipdate < DATE '1995-01-01'
  AND l_discount BETWEEN 0.05 AND 0.07
  AND l_quantity < 24"

run_query_with_results "Q6" "$Q6_SQL"

# Q9: Product Type Profit Measure
Q9_SQL="SELECT 
    nation,
    o_year,
    SUM(amount) AS sum_profit
FROM (
    SELECT
        n_name AS nation,
        EXTRACT(year FROM o_orderdate) AS o_year,
        l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount
    FROM
        part,
        supplier,
        lineitem,
        partsupp,
        orders,
        nation
    WHERE
        s_suppkey = l_suppkey
        AND ps_suppkey = l_suppkey
        AND ps_partkey = l_partkey
        AND p_partkey = l_partkey
        AND o_orderkey = l_orderkey
        AND s_nationkey = n_nationkey
        AND p_name LIKE '%green%'
) AS profit
GROUP BY
    nation,
    o_year
ORDER BY
    nation,
    o_year DESC"

run_query_with_results "Q9" "$Q9_SQL"

# Q13: Customer Distribution
Q13_SQL="SELECT
    c_count,
    COUNT(*) AS custdist
FROM (
    SELECT
        c_custkey,
        COUNT(o_orderkey) AS c_count
    FROM customer
    LEFT OUTER JOIN orders ON c_custkey = o_custkey
        AND o_comment NOT LIKE '%special%requests%'
    GROUP BY c_custkey
) AS c_orders
GROUP BY c_count
ORDER BY custdist DESC, c_count DESC"

run_query_with_results "Q13" "$Q13_SQL"

echo ""
echo "=== Benchmark Complete ==="
echo "Results CSV: ${RESULTS_FILE}"
echo "Results Logs: ${LOG_DIR}/${TIMESTAMP}/"
echo ""

# Display results for this run
echo "Latest results (${TIMESTAMP}):"
grep "^${TIMESTAMP}" "${RESULTS_FILE}" | column -t -s,

echo ""
echo "Log files created:"
ls -lh "${LOG_DIR}/${TIMESTAMP}/"

# Cleanup database file
rm -f "$DB_FILE" 2>/dev/null || true
echo ""
echo "Database file cleaned up: ${DB_FILE}"
