#!/bin/bash
# CedarDB TPC-H Benchmark Script
# Runs Q1, Q3, Q6, Q9, Q13 and records execution times
# Usage: benchmark_cedardb.sh [--query-results] [SF-1|SF-10|SF-100]
#   --query-results: save per-query result logs

set -e

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
parse_common_args "$@"

SCHEMA_LOCAL_PATH="$PROJECT_ROOT/sql/schema.sql"
RESULTS_FILE="$RESULTS_DIR/cedardb_results.csv"
LOG_DIR="$RESULTS_DIR/cedardb_logs"

# CedarDB connection — override via environment variables
CEDAR_HOST="${CEDAR_HOST:-localhost}"
CEDAR_PORT="${CEDAR_PORT:-5432}"
CEDAR_USER="${CEDAR_USER:-postgres}"
CEDAR_PASSWORD="${CEDAR_PASSWORD:-cedar}"
CEDAR_DB="${CEDAR_DB:-tpch}"

SCALE_FACTOR="$(normalize_sf "${POSITIONAL_ARGS[0]:-SF-1}")"

echo "=== CedarDB TPC-H Benchmark ==="
echo "Scale Factor: ${SCALE_FACTOR}"
echo "Data Directory: ${DATA_DIR}/${SCALE_FACTOR}"
echo "Results File: ${RESULTS_FILE}"
echo "Timestamp: ${TIMESTAMP}"
echo ""

ensure_csv "$RESULTS_FILE" "timestamp,scale_factor,query,execution_time_ms,gpu_name,memory_gb"
[[ "$SHOW_QUERY_RESULTS" -eq 1 ]] && mkdir -p "${LOG_DIR}/${TIMESTAMP}"

# ---------------------------------------------------------------------------
# CedarDB helpers
# ---------------------------------------------------------------------------
cedar_psql() {
    docker exec -e PGPASSWORD="${CEDAR_PASSWORD}" cedardb \
        psql -h localhost -p 5432 -U "${CEDAR_USER}" "$@"
}

schema_table_count() {
    cedar_psql -d "${CEDAR_DB}" -tAc \
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
        cedar_psql -d "${CEDAR_DB}" -v ON_ERROR_STOP=1 \
            -c "DROP TABLE IF EXISTS lineitem, orders, customer, partsupp, supplier, part, region, nation CASCADE;" >/dev/null
        docker cp "${SCHEMA_LOCAL_PATH}" cedardb:/tmp/schema.sql
        cedar_psql -d "${CEDAR_DB}" -v ON_ERROR_STOP=1 -f /tmp/schema.sql >/dev/null
    fi
}

# ---------------------------------------------------------------------------
# Run a query and extract timing
# ---------------------------------------------------------------------------
run_query() {
    local query_name=$1
    local query_sql=$2
    local exec_time

    echo "Running ${query_name}..."

    local output
    output=$(cedar_psql -d "${CEDAR_DB}" -c "\timing on" -c "${query_sql}" 2>&1)

    if echo "$output" | grep -q "ERROR:"; then
        echo "  ❌ ERROR: Query failed!"
        echo "$output" | grep "ERROR:" | head -3
        exec_time="0"
    else
        exec_time=$(echo "$output" | grep -oE 'Time: [0-9.]+' | grep -oE '[0-9.]+' | head -1)
        if [ -z "$exec_time" ]; then
            echo "  Warning: Could not extract execution time for ${query_name}"
            exec_time="0"
        else
            exec_time=$(round_ms "$exec_time")
            echo "  ✓ Execution time: ${exec_time} ms"
        fi
    fi

    echo "${TIMESTAMP},${SCALE_FACTOR},${query_name},${exec_time},${GPU_NAME},${MEMORY_GB}" >> "$RESULTS_FILE"

    if [[ "$SHOW_QUERY_RESULTS" -eq 1 ]]; then
        local log_file="${LOG_DIR}/${TIMESTAMP}/${SCALE_FACTOR}_${query_name}.log"
        {
            echo "=== CedarDB ${query_name} Results ==="
            echo "Timestamp: ${TIMESTAMP}"
            echo "Scale Factor: ${SCALE_FACTOR}"
            echo "Query: ${query_name}"
            echo ""
            echo "${output}"
        } > "$log_file"
        echo "  ✓ Results saved to: ${log_file}"
    fi
}

# ---------------------------------------------------------------------------
# Check connectivity and load data
# ---------------------------------------------------------------------------
if ! cedar_psql -d postgres -c "SELECT 1" &>/dev/null; then
    echo "Error: Cannot connect to CedarDB at ${CEDAR_HOST}:${CEDAR_PORT}"
    echo "Make sure CedarDB is running and accessible"
    exit 1
fi

DB_EXISTS=$(cedar_psql -d postgres -c "\l" | grep -w "${CEDAR_DB}" | wc -l)
DATA_PATH="$DATA_DIR/$SCALE_FACTOR"

NEED_RELOAD=0
if [ "$DB_EXISTS" -eq 0 ]; then
    NEED_RELOAD=1
else
    LINEITEM_COUNT=$(cedar_psql -d "${CEDAR_DB}" -t -c "SELECT COUNT(*) FROM lineitem;" 2>/dev/null || echo "0")
    LINEITEM_COUNT=$(echo "$LINEITEM_COUNT" | tr -d ' ')
    if [ "${SCALE_FACTOR}" == "SF-1" ] && [ "$LINEITEM_COUNT" -ne 6001215 ]; then NEED_RELOAD=1
    elif [ "${SCALE_FACTOR}" == "SF-10" ] && [ "$LINEITEM_COUNT" -ne 59986052 ]; then NEED_RELOAD=1
    elif [ "${SCALE_FACTOR}" == "SF-100" ] && [ "$LINEITEM_COUNT" -ne 600037902 ]; then NEED_RELOAD=1
    fi
fi

if [ "$NEED_RELOAD" -eq 1 ]; then
    if [ "$DB_EXISTS" -eq 1 ]; then
        ensure_schema
        echo "Truncating existing tables to reload with ${SCALE_FACTOR}..."
        cedar_psql -d "${CEDAR_DB}" \
            -c "TRUNCATE TABLE lineitem, orders, customer, partsupp, supplier, part, region, nation CASCADE;" 2>/dev/null || true
    else
        echo "Creating TPC-H database..."
        cedar_psql -d postgres -c "CREATE DATABASE ${CEDAR_DB};"
        ensure_schema
    fi

    echo "Loading TPC-H data for ${SCALE_FACTOR}..."
    echo "  Copying data files to container..."
    docker cp "${DATA_PATH}/." cedardb:/tmp/tpch_data/

    spinner=('⠋' '⠙' '⠹' '⠸' '⠼' '⠴' '⠦' '⠧' '⠇' '⠏')
    for table in nation region part supplier partsupp customer orders lineitem; do
        if [ -f "${DATA_PATH}/${table}.tbl" ]; then
            file_size=$(ls -lh "${DATA_PATH}/${table}.tbl" | awk '{print $5}')
            echo "  Loading ${table} (${file_size})..."

            cedar_psql -d "${CEDAR_DB}" -c "\COPY ${table} FROM '/tmp/tpch_data/${table}.tbl' DELIMITER '|' CSV;" &
            copy_pid=$!

            i=0
            while kill -0 $copy_pid 2>/dev/null; do
                printf "\r    ${spinner[$i]} Loading ${table}... "
                i=$(( (i+1) % 10 ))
                sleep 0.1
            done

            wait $copy_pid
            copy_result=$?
            if [ $copy_result -eq 0 ]; then
                row_count=$(cedar_psql -d "${CEDAR_DB}" -t -c "SELECT COUNT(*) FROM ${table};")
                printf "\r    ✓ Loaded ${table}: $(echo $row_count | tr -d ' ') rows (${file_size})\n"
            else
                printf "\r    ✗ Failed to load ${table}\n"
            fi
        fi
    done
    echo "Data loaded successfully"
fi

echo ""
echo "Warming up cache..."
cedar_psql -d "${CEDAR_DB}" -c "SELECT COUNT(*) FROM lineitem; SELECT COUNT(*) FROM orders; SELECT COUNT(*) FROM customer;" &>/dev/null
echo "Cache warmed up"

echo ""
echo "Starting benchmark runs..."
echo ""

# Run all 22 TPC-H queries from sql/ directory
for q_num in $(seq 1 22); do
    sql_file="$PROJECT_ROOT/sql/q${q_num}.sql"
    if [[ ! -f "$sql_file" ]]; then
        echo "Warning: $sql_file not found, skipping Q${q_num}"
        continue
    fi
    query_sql=$(grep -v '^--' "$sql_file" | tr '\n' ' ')
    run_query "Q${q_num}" "$query_sql"
done

echo ""
echo "=== Benchmark Complete ==="
echo "Results written to: ${RESULTS_FILE}"
echo ""
echo "Latest results (${TIMESTAMP}):"
grep "^${TIMESTAMP}" "${RESULTS_FILE}" | column -t -s,

if [[ "$SHOW_QUERY_RESULTS" -eq 1 ]]; then
    echo ""
    echo "Log files:"
    ls -lh "${LOG_DIR}/${TIMESTAMP}/"
fi
