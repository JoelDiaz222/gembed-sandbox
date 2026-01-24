# Benchmark 1: Internal vs External Embedding Generation

## Purpose

This benchmark compares different methods for generating embeddings before storing them in PostgreSQL with pgvector:

1. **Internal generation** using pg_gembed extension (embedding happens inside PostgreSQL)
2. **External generation** using EmbedAnything library (embedding happens outside PostgreSQL)

## Methods Tested

| Method        | Embedding Location     | Transport | Description                              |
|---------------|------------------------|-----------|------------------------------------------|
| PG Local      | PostgreSQL (pg_gembed) | Direct    | Internal embedding via SQL function      |
| PG gRPC       | PostgreSQL (pg_gembed) | Direct    | Same as above with gRPC warmup           |
| In-Process    | Python (EmbedAnything) | Direct    | ONNX model in Python process (no server) |
| External gRPC | Python (EmbedAnything) | gRPC      | Embeddings via gRPC server               |
| External HTTP | Python (EmbedAnything) | HTTP      | Embeddings via HTTP server               |

## Key Features

- **Fresh connection per method**: Each method gets a new database connection to ensure fair comparison
- **Warmup phase**: Stabilize performance before measurement
- **Resource monitoring**: Tracks CPU, memory (process and system-wide)
- **All-at-once processing**: Measures latency of fully processing a subset of the dataset for each size

## Running

1. Start the gRPC server (if testing gRPC methods):

```bash
PYTHONPATH=.:proto python3.13 servers/grpc_server.py
```

2. Start the HTTP server (if testing HTTP methods):

```bash
PYTHONPATH=.:proto python3.13 servers/http_server.py
```

3. Run the benchmark:

```bash
PYTHONPATH=.:proto python3.13 1_internal-vs-external-gen/benchmark.py
```

## Output

Console summary with performance comparison table across different test sizes.
