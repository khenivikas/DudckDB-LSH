# DuckDB-LSH: Locality-Sensitive Hashing for Vector Similarity Search in DuckDB

This project extends DuckDB's [Vector Similarity Search (VSS)](https://github.com/duckdb/duckdb-vss) extension with a **Locality-Sensitive Hashing (LSH)** index, implemented as a native DuckDB index type alongside the existing HNSW index. The LSH index uses random hyperplane hashing to enable approximate nearest neighbor (ANN) search directly within DuckDB.

## Overview

The extension registers two index types:
- **HNSW** — Graph-based approximate nearest neighbor index (from the original VSS extension, powered by [usearch](https://github.com/unum-cloud/usearch))
- **LSH** — Hash-based approximate nearest neighbor index using random hyperplane projections

LSH partitions the vector space using multiple hash tables, each built from random hyperplane projections. Vectors that hash to the same bucket are candidate neighbors. By combining results across multiple independent hash tables, the index achieves high recall while avoiding exhaustive distance computation.

### LSH Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lsh_tables` | Number of independent hash tables | 8 |
| `lsh_bits` | Number of hash bits (hyperplanes) per table | 16 |

- **More tables** → higher recall, higher latency and build time
- **More bits** → lower recall, lower latency (more selective hashing)

## Prerequisites

- C++17 compiler (Clang or GCC)
- CMake 2.8.12+
- Make
- Python 3.8+ with `duckdb`, `numpy`, `matplotlib` (for benchmarks)
- `h5py` (for GloVe/Deep dataset conversion)

## Building

### 1. Clone the repository

```bash
git clone --recursive https://github.com/khenivikas/DudckDB-LSH.git
cd DudckDB-LSH
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### 2. Tag the DuckDB version

The DuckDB submodule must be tagged as `v1.4.0` so the built extension matches the version of the `duckdb` Python package. If the tag doesn't already exist locally:

```bash
cd duckdb
git tag v1.4.0
cd ..
```

You can verify with:

```bash
cd duckdb && git describe --tags
# Should output: v1.4.0
```

### 3. Build the extension

```bash
make release
```

If you encounter a CMake generator mismatch error, clear the cache first:

```bash
rm build/release/CMakeCache.txt
rm -rf build/release/CMakeFiles
make release
```

This produces:
- `./build/release/duckdb` — DuckDB shell with the extension loaded
- `./build/release/extension/vss/vss.duckdb_extension` — Loadable extension binary

### 4. Verify the build

```bash
./build/release/duckdb --version
# Should output: v1.4.0 ...
```

## Usage

### Creating an LSH Index

```sql
CREATE TABLE vectors (id INTEGER, embedding FLOAT[128]);

-- Insert data
INSERT INTO vectors SELECT i, array_value(...) FROM range(100000) t(i);

-- Create LSH index with custom parameters
CREATE INDEX lsh_idx ON vectors USING LSH (embedding)
    WITH (lsh_tables=16, lsh_bits=8);
```

### Querying

The index accelerates `ORDER BY ... LIMIT` queries using distance functions:

```sql
SELECT id FROM vectors
ORDER BY array_distance(embedding, [1.0, 2.0, ...]::FLOAT[128])
LIMIT 10;
```

### Creating an HNSW Index (for comparison)

```sql
CREATE INDEX hnsw_idx ON vectors USING HNSW (embedding);
```

## Benchmarks

The `eval/` directory contains a benchmarking framework that compares Full Scan, HNSW, and LSH across multiple datasets and parameter configurations.

### Benchmark Datasets

Datasets are **not included in the repository** due to their size. Download them manually and place them in the correct directories:

#### SIFT-1M (128d, Euclidean)

- **Source**: [ANN Benchmarks SIFT](http://corpus-texmex.irisa.fr/) — download `sift.tar.gz`
- **Place files in**: `eval/data/sift/`
- **Required files**: `sift_base.fvecs`, `sift_query.fvecs`, `sift_groundtruth.ivecs`
- **Size**: ~500MB
- **Vectors**: 1M base, 10K queries, 128 dimensions

#### GloVe-200 (200d, Angular)

- **Download**: [http://ann-benchmarks.com/glove-200-angular.hdf5](http://ann-benchmarks.com/glove-200-angular.hdf5)
- **Place HDF5 in**: `eval/data/glove/`
- **Convert to fvecs**:
  ```bash
  pip install h5py
  python eval/download_glove.py
  ```
- **Size**: ~1.1GB (HDF5)
- **Vectors**: 1.18M base, 10K queries, 200 dimensions

### Running Benchmarks

```bash
# Install Python dependencies
pip install duckdb numpy matplotlib h5py

# Run individual dataset benchmarks
python eval/benchmark.py --dataset sift
python eval/benchmark.py --dataset glove

# Generate cross-dataset comparison charts
python eval/compare.py
```

Each benchmark evaluates:
- **Full Scan** (exact brute-force baseline)
- **HNSW** (graph-based ANN index)
- **LSH** with a grid of parameter combinations: `lsh_tables` ∈ {2, 4, 8, 16, 32} × `lsh_bits` ∈ {8, 12, 16, 24, 32}

Metrics collected: Recall@10, mean/median/p99 latency, throughput (queries/sec), and index build time.

### Benchmark Output Structure

```
eval/
├── sift/
│   ├── results/benchmark_results.csv
│   └── plots/                          # SIFT-specific charts
├── glove/
│   ├── results/benchmark_results.csv
│   └── plots/                          # GloVe-specific charts
└── comparison/                         # Cross-dataset comparison charts
```

Generated plots include:
- Recall vs. latency scatter plots (per dataset size)
- LSH parameter sensitivity heatmaps (recall and latency)
- Index build time vs. dataset size
- Query latency scalability
- Throughput bar charts
- Summary comparison across methods

## Results Summary

### SIFT-128d (HNSW-favorable)

On SIFT, HNSW dominates the recall-latency tradeoff. At 1M vectors, HNSW achieves 0.96 recall at 1.9ms latency. LSH achieves higher recall (0.999 with t=32, b=8) but at significantly higher latency (469ms).

### GloVe-200d (LSH-favorable)

On GloVe, the results invert dramatically:

| Method | Recall@10 | Mean Latency | Build Time |
|--------|-----------|-------------|------------|
| Full Scan | 1.000 | 128.9ms | 0s |
| HNSW | **0.654** | 14.2ms | **431.1s** |
| LSH (t=32, b=8) | **0.999** | 517.0ms | 14.3s |
| LSH (t=16, b=8) | **0.998** | 463.2ms | **7.9s** |

At 1M vectors on GloVe:
- HNSW recall collapses to 0.654 — effectively unusable
- LSH maintains 0.999 recall with t=32, b=8
- LSH builds 54x faster than HNSW (7.9s vs 431.1s) while achieving vastly superior recall

### Key Findings

- **HNSW's greedy graph traversal degrades on high-dimensional, angular-distance data**, where the navigable small-world property becomes less effective
- **LSH's randomized hashing degrades gracefully** across dimensionality and distance structure, making it the superior choice for dense embedding vectors from NLP and deep learning models
- **LSH builds significantly faster**, which matters for workloads with frequent index rebuilds
- **LSH's main weakness is query latency** — high-recall configurations scan large candidate sets

## Project Structure

```
DudckDB-LSH/
├── src/
│   ├── hnsw/                    # HNSW index implementation
│   ├── lsh/                     # LSH index implementation
│   │   ├── lsh_index.cpp        # Core LSH logic (hashing, search)
│   │   ├── lsh_index_scan.cpp   # Index scan operator
│   │   ├── lsh_index_plan.cpp   # Query plan integration
│   │   ├── lsh_optimize_scan.cpp # Optimizer rules
│   │   └── lsh_index_physical_create.cpp
│   ├── include/                 # Header files
│   └── vss_extension.cpp        # Extension entry point
├── eval/                        # Benchmarking framework
│   ├── benchmark.py             # Main benchmark script
│   ├── compare.py               # Cross-dataset comparison
│   ├── download_glove.py        # GloVe HDF5 → fvecs converter
├── test/sql/                    # SQL-based tests
├── duckdb/                      # DuckDB submodule
└── extension-ci-tools/          # CI tooling submodule
```

## Running Tests

```bash
make test
```

## License

See [LICENSE](LICENSE).
