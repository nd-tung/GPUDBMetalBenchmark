#!/bin/bash
# CedarDB Benchmark Script with Results Export
# Runs TPC-H queries Q1, Q3, Q6, Q9, Q13 and records both execution times and results

set -e

# Parse command line arguments
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SCHEMA_LOCAL_PATH="$PROJECT_ROOT/sql/schema.sql"
SCALE_FACTOR=${1:-SF-1}
DATA_DIR="$PROJECT_ROOT/data/${SCALE_FACTOR}"
RESULTS_FILE="$PROJECT_ROOT/results/cedardb_results.csv"
LOG_DIR="$PROJECT_ROOT/results/cedardb_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# CedarDB connection details
CEDAR_HOST="localhost"
CEDAR_PORT="5432"
CEDAR_USER="postgres"
CEDAR_PASSWORD="cedar"
CEDAR_DB="tpch"

echo "=== CedarDB TPC-H Benchmark with Results Export ==="
echo "Scale Factor: ${SCALE_FACTOR}"
echo "Data Directory: ${DATA_DIR}"
echo "Results CSV: ${RESULTS_FILE}"
echo "Results Logs: ${LOG_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo ""

# Create results and log directories if they don't exist
mkdir -p benchmark_results
mkdir -p "${LOG_DIR}/${TIMESTAMP}"

# Initialize CSV file with header if it doesn't exist
if [ ! -f "${RESULTS_FILE}" ]; then
    echo "timestamp,scale_factor,query,execution_time_ms" > "${RESULTS_FILE}"
fi

round_ms() {
    local raw="$1"
    raw="$(echo "$raw" | head -n 1 | tr -d '[:space:],')"
    if [ -z "$raw" ]; then
        echo "0"
        return 0
    fi
    awk -v t="$raw" 'BEGIN{ if(t=="" || t==".") {print 0} else {printf "%.0f", t} }'
}

schema_table_count() {
    docker exec -e PGPASSWORD="${CEDAR_PASSWORD}" cedardb \
        psql -h localhost -p 5432 -U "${CEDAR_USER}" -d "${CEDAR_DB}" -tAc \
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public' AND table_name IN ('nation','region','part','supplier','partsupp','customer','orders','lineitem');" \
        2>/dev/null | tr -d '[:space:]' || echo "0"
}

ensure_schema() {
    if [ ! -f "${SCHEMA_LOCAL_PATH}" ]; then
        echo "Error: schema file not found at ${SCHEMA_LOCAL_PATH}"
        exit 1
    fi

    local table_count
    table_count="$(schema_table_count)"
    if [ "$table_count" -ne 8 ]; then
        echo "  Creating schema (found ${table_count}/8 tables)..."
        docker exec -e PGPASSWORD="${CEDAR_PASSWORD}" cedardb \
            psql -h localhost -p 5432 -U "${CEDAR_USER}" -d "${CEDAR_DB}" \
            -v ON_ERROR_STOP=1 \
            -c "DROP TABLE IF EXISTS lineitem, orders, customer, partsupp, supplier, part, region, nation CASCADE;" >/dev/null

        docker cp "${SCHEMA_LOCAL_PATH}" cedardb:/tmp/schema.sql
        docker exec -e PGPASSWORD="${CEDAR_PASSWORD}" cedardb \
            psql -h localhost -p 5432 -U "${CEDAR_USER}" -d "${CEDAR_DB}" \
            -v ON_ERROR_STOP=1 -f /tmp/schema.sql >/dev/null
    fi
}

# Function to run a query, extract timing, and save results
run_query_with_results() {
    local query_name=$1
    local query_sql=$2
    local exec_time
    local log_file="${LOG_DIR}/${TIMESTAMP}/${SCALE_FACTOR}_${query_name}.log"
    
    echo "Running ${query_name}..."
    
    # Run query with \timing to get execution time
    local output=$(docker exec -e PGPASSWORD="${CEDAR_PASSWORD}" cedardb psql -h localhost -p 5432 -U "${CEDAR_USER}" -d "${CEDAR_DB}" -c "\timing on" -c "${query_sql}" 2>&1)
    
    # Save full output to log file
    echo "=== CedarDB ${query_name} Results ===" > "${log_file}"
    echo "Timestamp: ${TIMESTAMP}" >> "${log_file}"
    echo "Scale Factor: ${SCALE_FACTOR}" >> "${log_file}"
    echo "Query: ${query_name}" >> "${log_file}"
    echo "" >> "${log_file}"
    echo "${output}" >> "${log_file}"
    
    # Check for errors
    if echo "$output" | grep -q "ERROR:"; then
        echo "  ❌ ERROR: Query failed!"
        echo "$output" | grep "ERROR:" | head -3
        echo "  Log saved to: ${log_file}"
        exec_time="0"
    else
        # Extract execution time from "Time: X.XXX ms" format
        exec_time=$(echo "$output" | grep -oE 'Time: [0-9.]+' | grep -oE '[0-9.]+' | head -1)
        
        if [ -z "$exec_time" ]; then
            echo "  Warning: Could not extract execution time for ${query_name}"
            echo "  Log saved to: ${log_file}"
            exec_time="0"
        else
            # Round to integer milliseconds
            exec_time=$(round_ms "$exec_time")
            echo "  ✓ Execution time: ${exec_time} ms"
            echo "  ✓ Results saved to: ${log_file}"
        fi
    fi
    
    # Append timing to CSV
    echo "${TIMESTAMP},${SCALE_FACTOR},${query_name},${exec_time}" >> "${RESULTS_FILE}"
}

# Check if CedarDB is accessible
if ! docker exec -e PGPASSWORD="${CEDAR_PASSWORD}" cedardb psql -h localhost -p 5432 -U "${CEDAR_USER}" -d postgres -c "SELECT 1" &>/dev/null; then
    echo "Error: Cannot connect to CedarDB at ${CEDAR_HOST}:${CEDAR_PORT}"
    echo "Make sure CedarDB is running and accessible"
    exit 1
fi

# Check if database exists, create and load data if needed
DB_EXISTS=$(docker exec -e PGPASSWORD="${CEDAR_PASSWORD}" cedardb psql -h localhost -p 5432 -U "${CEDAR_USER}" -d postgres -c "\l" | grep -w "${CEDAR_DB}" | wc -l)

# Check if we need to reload data (different scale factor or no data)
NEED_RELOAD=0
if [ "$DB_EXISTS" -eq 0 ]; then
    NEED_RELOAD=1
else
    # Check if lineitem table exists and has data
    LINEITEM_COUNT=$(docker exec -e PGPASSWORD="${CEDAR_PASSWORD}" cedardb psql -h localhost -p 5432 -U "${CEDAR_USER}" -d "${CEDAR_DB}" -t -c "SELECT COUNT(*) FROM lineitem;" 2>/dev/null || echo "0")
    LINEITEM_COUNT=$(echo "$LINEITEM_COUNT" | tr -d ' ')
    
    # SF-1 has ~6M rows, SF-10 has ~60M rows, SF-100 has ~600M rows
    if [ "${SCALE_FACTOR}" == "SF-1" ] && [ "$LINEITEM_COUNT" -ne 6001215 ]; then
        NEED_RELOAD=1
    elif [ "${SCALE_FACTOR}" == "SF-10" ] && [ "$LINEITEM_COUNT" -ne 59986052 ]; then
        NEED_RELOAD=1
    elif [ "${SCALE_FACTOR}" == "SF-100" ] && [ "$LINEITEM_COUNT" -ne 600037902 ]; then
        NEED_RELOAD=1
    fi
fi

if [ "$NEED_RELOAD" -eq 1 ]; then
    if [ "$DB_EXISTS" -eq 1 ]; then
        ensure_schema

        echo "Truncating existing tables to reload with ${SCALE_FACTOR}..."
        docker exec -e PGPASSWORD="${CEDAR_PASSWORD}" cedardb \
            psql -h localhost -p 5432 -U "${CEDAR_USER}" -d "${CEDAR_DB}" \
            -c "TRUNCATE TABLE lineitem, orders, customer, partsupp, supplier, part, region, nation CASCADE;" 2>/dev/null || true
    else
        echo "Creating TPC-H database..."
        docker exec -e PGPASSWORD="${CEDAR_PASSWORD}" cedardb psql -h localhost -p 5432 -U "${CEDAR_USER}" -d postgres -c "CREATE DATABASE ${CEDAR_DB};"

        ensure_schema
    fi
    
    echo "Loading TPC-H data for ${SCALE_FACTOR}..."

    # Copy data files to container first
    echo "  Copying data files to container..."
    docker cp "${DATA_DIR}/." cedardb:/tmp/tpch_data/
    
    # Load data from .tbl files with progress indication
    for table in nation region part supplier partsupp customer orders lineitem; do
        if [ -f "${DATA_DIR}/${table}.tbl" ]; then
            # Get file size for progress indication
            file_size=$(ls -lh "${DATA_DIR}/${table}.tbl" | awk '{print $5}')
            echo "  Loading ${table} (${file_size})..."
            
            # Run COPY in background and show a progress indicator
            docker exec -e PGPASSWORD="${CEDAR_PASSWORD}" cedardb psql -h localhost -p 5432 -U "${CEDAR_USER}" -d "${CEDAR_DB}" -c "\COPY ${table} FROM '/tmp/tpch_data/${table}.tbl' DELIMITER '|' CSV;" &
            copy_pid=$!
            
            # Show progress while COPY is running
            spinner=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
            i=0
            while kill -0 $copy_pid 2>/dev/null; do
                printf "\r    ${spinner[$i]} Loading ${table}... "
                i=$(( (i+1) % 10 ))
                sleep 0.1
            done
            
            # Wait for the process to complete and get result
            wait $copy_pid
            copy_result=$?
            
            if [ $copy_result -eq 0 ]; then
                # Get row count
                row_count=$(docker exec -e PGPASSWORD="${CEDAR_PASSWORD}" cedardb psql -h localhost -p 5432 -U "${CEDAR_USER}" -d "${CEDAR_DB}" -t -c "SELECT COUNT(*) FROM ${table};")
                printf "\r    ✓ Loaded ${table}: $(echo $row_count | tr -d ' ') rows (${file_size})\n"
            else
                printf "\r    ✗ Failed to load ${table}\n"
            fi
        fi
    done
    
    echo "Data loaded successfully"
fi

echo ""
echo "Warming up cache (like DuckDB and GPU approach)..."
docker exec -e PGPASSWORD="${CEDAR_PASSWORD}" cedardb psql -h localhost -p 5432 -U "${CEDAR_USER}" -d "${CEDAR_DB}" -c "SELECT COUNT(*) FROM lineitem; SELECT COUNT(*) FROM orders; SELECT COUNT(*) FROM customer;" &>/dev/null
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
ORDER BY l_returnflag, l_linestatus;"

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
LIMIT 10;"

run_query_with_results "Q3" "$Q3_SQL"

# Q6: Forecasting Revenue Change
Q6_SQL="SELECT SUM(l_extendedprice * l_discount) AS revenue
FROM lineitem
WHERE l_shipdate >= DATE '1994-01-01'
  AND l_shipdate < DATE '1995-01-01'
  AND l_discount BETWEEN 0.05 AND 0.07
  AND l_quantity < 24;"

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
    o_year DESC;"

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
ORDER BY custdist DESC, c_count DESC;"

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
