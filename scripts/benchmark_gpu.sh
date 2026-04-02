#!/bin/bash
# GPU Metal TPC-H Benchmark Script
# Runs Q1, Q3, Q6, Q9, Q13 and records timing metrics to CSV
# Usage: benchmark_gpu.sh [--query-results] [sf1|sf10|sf100|all]
#   --query-results: show live GPU output and save per-query logs

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
parse_common_args "$@"

BUILD_BIN="$PROJECT_ROOT/build/bin/GPUDBMetalBenchmark"
LOG_DIR="$RESULTS_DIR/gpu_logs"
GPU_CSV="$RESULTS_DIR/gpu_results.csv"

QUERIES="Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 Q10 Q11 Q12 Q13 Q14 Q15 Q16 Q17 Q18 Q19 Q20 Q21 Q22"

if [[ "$SHOW_QUERY_RESULTS" -eq 1 ]]; then
    echo "=== GPU Metal TPC-H Benchmark with Results Export ==="
    echo "Binary: ${BUILD_BIN}"
    echo "Results CSV: ${GPU_CSV}"
    echo "Timestamp: ${TIMESTAMP}"
    echo ""
fi

mkdir -p "$RESULTS_DIR" "${LOG_DIR}/${TIMESTAMP}"

if [[ ! -x "$BUILD_BIN" ]]; then
  echo "Error: GPUDBMetalBenchmark binary not found at $BUILD_BIN" >&2
  echo "Run 'make build' first."
  exit 1
fi

# CSV header
ensure_csv "$GPU_CSV" "timestamp,scale_factor,query,gpu_exec_ms,cpu_post_ms,total_exec_ms,gpu_name,memory_gb"

# ---------------------------------------------------------------------------
# Timing extraction
# ---------------------------------------------------------------------------
extract_metric() {
  local block="$1"
  local label="$2"
  echo "$block" | grep -F "$label:" | head -1 \
    | grep -oE '[0-9]+\.[0-9]+[[:space:]]+ms' | head -1 | awk '{print $1}'
}

# ---------------------------------------------------------------------------
# Parse benchmark output → CSV rows
# ---------------------------------------------------------------------------
parse_and_record() {
  local sf_label="$1"
  local out_file="$2"

  for q in $QUERIES; do
    local header_line
    header_line=$(grep -E "^(SF100 )?${q} \|" "$out_file" || true)
    [[ -z "$header_line" ]] && continue

    local block
    block=$(awk "/^(SF100 )?${q} \|/,/Total Execution:/" "$out_file")

    local gpu_exec cpu_post total_exec
    gpu_exec=$(extract_metric "$block" "GPU Execution")
    cpu_post=$(extract_metric "$block" "CPU Post Process")
    total_exec=$(extract_metric "$block" "Total Execution")

    if [[ -n "$gpu_exec" ]]; then
      echo "$TIMESTAMP,$sf_label,$q,${gpu_exec},${cpu_post:-0},${total_exec:-0},${GPU_NAME},${MEMORY_GB}" >> "$GPU_CSV"
      printf "  %-4s gpu=%-10s cpu_post=%-10s total=%s ms\n" \
        "$q" "${gpu_exec} ms" "${cpu_post:-0} ms" "${total_exec:-0}"
    fi

    # Save per-query log
    local q_log="${LOG_DIR}/${TIMESTAMP}/${sf_label}_${q}.log"
    local q_num="${q#Q}"
    {
      echo "=== GPU Metal ${q} Results ==="
      echo "Timestamp: ${TIMESTAMP}"
      echo "Scale Factor: ${sf_label}"
      echo ""
      awk "/Running.*Query ${q_num}/,/Total Execution:/" "$out_file" 2>/dev/null ||
        awk "/Running.*${q}/,/Total Execution:/" "$out_file" 2>/dev/null || true
    } > "$q_log"
    [[ "$SHOW_QUERY_RESULTS" -eq 1 ]] && [[ -s "$q_log" ]] && echo "    → ${q_log}"
  done
}

# ---------------------------------------------------------------------------
# Run benchmark for one scale factor
# ---------------------------------------------------------------------------
run_benchmark() {
  local sf_arg="$1"
  local sf_label="$2"

  echo "=== GPU benchmark: ${sf_label} ==="
  local out_file="${LOG_DIR}/${TIMESTAMP}/${sf_label}_full.log"

  if [[ "$SHOW_QUERY_RESULTS" -eq 1 ]]; then
    echo "Running GPU benchmarks for ${sf_label}..."
    (cd "$PROJECT_ROOT" && "$BUILD_BIN" "$sf_arg" all) 2>&1 | tee "$out_file"
  else
    (cd "$PROJECT_ROOT" && "$BUILD_BIN" "$sf_arg" all) > "$out_file" 2>&1 || true
  fi

  echo ""
  [[ "$SHOW_QUERY_RESULTS" -eq 1 ]] && echo "Extracting results for ${sf_label}..."
  parse_and_record "$sf_label" "$out_file"
  echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
MODE="${POSITIONAL_ARGS[0]:-all}"

case "$MODE" in
  sf1|SF-1)     run_benchmark sf1  SF-1   ;;
  sf10|SF-10)   run_benchmark sf10 SF-10  ;;
  sf100|SF-100) run_benchmark sf100 SF-100 ;;
  all)
    [[ -d "$PROJECT_ROOT/data/SF-1" ]]   && run_benchmark sf1  SF-1
    [[ -d "$PROJECT_ROOT/data/SF-10" ]]  && run_benchmark sf10 SF-10  || true
    [[ -d "$PROJECT_ROOT/data/SF-100" ]] && run_benchmark sf100 SF-100 || true
    ;;
  *)
    echo "Usage: $0 [--query-results] [sf1|sf10|sf100|all]"
    exit 1
    ;;
esac

echo "Results saved to: $GPU_CSV"
echo "Logs: ${LOG_DIR}/${TIMESTAMP}/"
echo ""
echo "Latest results (${TIMESTAMP}):"
grep "^${TIMESTAMP}" "$GPU_CSV" | column -t -s,

if [[ "$SHOW_QUERY_RESULTS" -eq 1 ]]; then
  echo ""
  echo "Log files:"
  ls -lh "${LOG_DIR}/${TIMESTAMP}/"
fi
