# GPU Database Benchmark

GPU-accelerated database operations using Apple Metal vs DuckDB vs CedarDB comparison.

## Latest Benchmark Results

**Timestamp**: 2025-12-25  
**Methodology**: Warm cache (data pre-loaded), execution time only

<img width="3600" height="1800" alt="sf1_comparison" src="https://github.com/user-attachments/assets/cc16778e-95b4-49ae-9237-21c6de63154b" />
<img width="3600" height="1800" alt="sf10_comparison" src="https://github.com/user-attachments/assets/657ab6d8-ad19-4894-8bac-8c2055f86292" />

### GPU Metal Results (Timestamp: 20251225_190157)

#### SF-1 Dataset (6M lineitem rows)
| Query | GPU Time (ms) | CPU Time (ms) | Wall Clock (ms) |
|-------|--------------|----------------|----------------|
| Q1    | 26.83        | 0.00           | 26.84          |
| Q3    | 1.65         | 1.25           | 2.90           |
| Q6    | 1.65         | 0.00           | 1.65           |
| Q9    | 23.97        | 0.03           | 24.00          |
| Q13   | 21.55        | 2.14           | 23.69          |

#### SF-10 Dataset (60M lineitem rows)
| Query | GPU Time (ms) | CPU Time (ms) | Wall Clock (ms) |
|-------|--------------|----------------|----------------|
| Q1    | 157.01       | 0.01           | 157.02         |
| Q3    | 17.35        | 20.73          | 38.08          |
| Q6    | 9.66         | 0.00           | 9.67           |
| Q9    | 364.58       | 0.13           | 364.71         |
| Q13   | 215.71       | 33.94          | 249.64         |

### DuckDB Results (Timestamp: 20251225_191603)

#### SF-1 Dataset
| Query | Wall Clock (ms) |
|-------|-----------------|
| Q1    | 64              |
| Q3    | 28              |
| Q6    | 12              |
| Q9    | 76              |
| Q13   | 71              |

#### SF-10 Dataset
| Query | Wall Clock (ms) |
|-------|-----------------|
| Q1    | 574             |
| Q3    | 276             |
| Q6    | 102             |
| Q9    | 1461            |
| Q13   | 805             |

### CedarDB Results (Timestamp: 20251225_191707)

#### SF-1 Dataset
| Query | Wall Clock (ms) |
|-------|-----------------|
| Q1    | 91              |
| Q3    | 117             |
| Q6    | 8               |
| Q9    | 489             |
| Q13   | 166             |

#### SF-10 Dataset
| Query | Wall Clock (ms) |
|-------|-----------------|
| Q1    | 493             |
| Q3    | 2248            |
| Q6    | 37              |
| Q9    | 1666            |
| Q13   | 2365            |

## How to Run Benchmarks

### Prerequisites
```bash
# Install DuckDB
brew install duckdb

# Install CedarDB via Docker
docker pull cedardb/cedardb
```

### Quick Start
```bash
# 1. Build GPU benchmark
make

# 2. Generate test data (if not already generated)
./scripts/create_tpch_data.sh

# 3. Run all benchmarks
./scripts/benchmark_gpu.sh sf1
./scripts/benchmark_duckdb.sh SF-1
./scripts/benchmark_cedardb.sh SF-1  # Requires Docker

# 4. View results
cat results/gpu_results.csv
cat results/duckdb_results.csv
cat results/cedardb_results.csv
```

### Manual Execution
```bash
# Run individual queries manually
./build/bin/GPUDBMetalBenchmark sf1 q1
./build/bin/GPUDBMetalBenchmark sf10 q13
```

## Benchmark Scripts

The project includes automated benchmark scripts for running comprehensive performance tests:

### Data Generation
```bash
./scripts/create_tpch_data.sh
```
Generates TPC-H benchmark data at different scale factors (SF-1, SF-10). Downloads and compiles the TPC-H dbgen tool, then generates `.tbl` files in `data/`.

### GPU Benchmarks
```bash
./scripts/benchmark_gpu.sh sf1
```
Runs all TPC-H queries (Q1, Q3, Q6, Q9, Q13) on GPU Metal implementation and saves timing results to `results/gpu_results.csv`.

```bash
./scripts/benchmark_gpu_with_query_results.sh sf1
```
Extended version that also saves complete query results to `results/gpu_logs/` for verification.

### DuckDB Benchmarks
```bash
./scripts/benchmark_duckdb.sh SF-1
```
Runs all queries on DuckDB and saves results to `results/duckdb_results.csv`.

```bash
./scripts/benchmark_duckdb_with_query_results.sh SF-1
```
Extended version with query result export to `results/duckdb_logs/`.

### CedarDB Benchmarks
```bash
# Start CedarDB container first
docker run -d --name cedardb -p 5432:5432 -e CEDAR_PASSWORD=cedar --memory=12g cedardb/cedardb

# Run benchmarks
./scripts/benchmark_cedardb.sh SF-1
```
Runs queries on CedarDB via Docker and saves to `results/cedardb_results.csv`.

```bash
./scripts/benchmark_cedardb_with_query_results.sh SF-1
```
Extended version with query result logging.

## Benchmark Details

- **TPC-H Queries**: Q1 (Pricing Summary), Q3 (Shipping Priority), Q6 (Revenue Forecasting), Q9 (Product Profit), Q13 (Customer Distribution)
- **Data Format**: TPC-H standard `.tbl` files
- **Cache Strategy**: Warm cache (data pre-loaded, queries run on hot cache)
- **Timing Method**: Execution time only (excludes I/O and data loading)



