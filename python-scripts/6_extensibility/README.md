# Benchmark 6: Extensibility (Image Embedding Backends Comparison)

## Purpose

This benchmark compares the performance of two different backends within `pg_gembed` for generating image embeddings:

- **EmbedAnything**: The default backend using high-level Python-based libraries.
- **ORT (ONNX Runtime)**: A high-performance inference engine for ONNX models.

The benchmark tracks throughput (images per second) and resource utilization across multiple embedding models.

## Methods Tested

| Method           | Provider      | Embedding Model |
|------------------|---------------|-----------------|
| `embed_anything` | EmbedAnything | Various (CLIP)  |
| `ort`            | ONNX Runtime  | Various (CLIP)  |

## Key Features

- **Fixed Size Throughput**: Focuses on a specific dataset size (default 1024 images) to compare constant overhead and
  sustained performance.
- **Multiple Models**: Tests different model sizes (Base, Large, etc.) to see how backends scale with model complexity.
- **In-DB Execution**: Both methods execute entirely within the PostgreSQL process using `pg_gembed`.

## Prerequisites

- **PostgreSQL** with `pg_gembed` extension installed.
- **TPCx-AI Data**: Specifically `CUSTOMER_IMAGES` in the `data/` directory.
- **Python Dependencies**: `pandas`, `matplotlib`, `psycopg2`.

## Running the Benchmark

1. Ensure PostgreSQL is running.
2. Run the benchmark script:
   ```bash
   PYTHONPATH=.:proto python3.13 6_extensibility/benchmark.py --sizes 1024 --runs 3
   ```
3. Generate the specific comparison plot:
   ```bash
   python3.13 6_extensibility/plot_backends.py --csv 6_extensibility/output/benchmark_YYYYMMDD_HHMMSS_run.csv
   ```

## Output

Results are saved in `output` with CSVs and plots.

