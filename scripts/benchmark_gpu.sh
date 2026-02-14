#!/bin/bash
# GPU Metal TPC-H Benchmark Script
# Runs Q1, Q3, Q6, Q9, Q13 and records timing metrics to CSV
# Usage: benchmark_gpu.sh [sf1|sf10|sf100|all]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_BIN="$PROJECT_ROOT/build/bin/GPUDBMetalBenchmark"
RESULTS_DIR="$PROJECT_ROOT/results"
LOG_DIR="$RESULTS_DIR/gpu_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
GPU_CSV="$RESULTS_DIR/gpu_results.csv"

QUERIES="Q1 Q3 Q6 Q9 Q13"

mkdir -p "$RESULTS_DIR" "${LOG_DIR}/${TIMESTAMP}"

# CSV header
CSV_HEADER="timestamp,scale_factor,query,gpu_exec_ms,cpu_post_ms,total_exec_ms"
if [[ ! -f "$GPU_CSV" ]]; then
  echo "$CSV_HEADER" > "$GPU_CSV"
fi

# ---------------------------------------------------------------------------
# Timing extraction
# ---------------------------------------------------------------------------
# Extract value from "  GPU Execution:      3.61 ms" or
# "  Total Execution:    3.61 ms  (GPU + CPU post)"
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
    block=$(awk "/^(SF100 )?${q} \|/,/Wall Clock:/" "$out_file")

    local gpu_exec cpu_post total_exec
    gpu_exec=$(extract_metric "$block" "GPU Execution")
    cpu_post=$(extract_metric "$block" "CPU Post Process")
    total_exec=$(extract_metric "$block" "Total Execution")

    if [[ -n "$gpu_exec" ]]; then
      echo "$TIMESTAMP,$sf_label,$q,${gpu_exec},${cpu_post:-0},${total_exec:-0}" >> "$GPU_CSV"
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
      awk "/Running.*Query ${q_num}/,/Wall Clock:/" "$out_file" 2>/dev/null ||
        awk "/Running.*${q}/,/Wall Clock:/" "$out_file" 2>/dev/null || true
    } > "$q_log"
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
  (cd "$PROJECT_ROOT" && "$BUILD_BIN" "$sf_arg" all) > "$out_file" 2>&1 || true
  parse_and_record "$sf_label" "$out_file"
  echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
MODE="${1:-all}"

case "$MODE" in
  sf1)    run_benchmark sf1  SF-1   ;;
  sf10)   run_benchmark sf10 SF-10  ;;
  sf100)  run_benchmark sf100 SF-100 ;;
  all)
    [[ -d "$PROJECT_ROOT/data/SF-1" ]]   && run_benchmark sf1  SF-1
    [[ -d "$PROJECT_ROOT/data/SF-10" ]]  && run_benchmark sf10 SF-10  || true
    [[ -d "$PROJECT_ROOT/data/SF-100" ]] && run_benchmark sf100 SF-100 || true
    ;;
  *)
    echo "Usage: $0 [sf1|sf10|sf100|all]"
    exit 1
    ;;
esac

echo "Results saved to: $GPU_CSV"
echo "Logs: ${LOG_DIR}/${TIMESTAMP}/"
echo ""
echo "Latest results (${TIMESTAMP}):"
grep "^${TIMESTAMP}" "$GPU_CSV" | column -t -s,
