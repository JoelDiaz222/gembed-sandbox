# Benchmark 2: PG+pgvector vs ChromaDB Generation + Storage

## Purpose

This benchmark compares embedding generation and storage approaches:

1. **PostgreSQL + pgvector + pg_gembed**: Internal embedding generation inside the database
2. **ChromaDB + EmbedAnything**: Embedding generation in the Python process (no server)

This tests both where embeddings are generated (internal vs external) and where they are stored (relational DB vs purpose-built vector DB).

## Methods Tested

| Method | Storage | Embedding Generation |
|--------|---------|---------------------|
| PG Local | PostgreSQL + pgvector | Inside PostgreSQL (pg_gembed) |
| PG gRPC | PostgreSQL + pgvector | Inside PostgreSQL (pg_gembed) |
| Chroma | ChromaDB | In Python process (EmbedAnything) |

## Key Differences from Benchmark 1

- **Embedding location matters**: pg_gembed generates embeddings inside PostgreSQL; ChromaDB uses EmbedAnything in Python process
- **Storage comparison**: Measures how different storage backends handle embeddings
- **ChromaDB integration**: Tests ChromaDB's persistent client with HNSW indexing

## Running

1. Start the gRPC server (required for pg_gembed embedding):
```bash
PYTHONPATH=.:proto python servers/grpc_server.py
```

2. Run the benchmark:
```bash
PYTHONPATH=.:proto python 2_pg_gembed-vs-chromadb/benchmark.py
```

## ChromaDB Configuration

- **Collection**: `bench_collection`
- **Index**: HNSW (Hierarchical Navigable Small World)
- **Distance**: Cosine similarity (default)
- **Persistence**: Local directory `chroma_bench_{timestamp}`

## pgvector Configuration

The HNSW index is configured to match ChromaDB's defaults for fair comparison:
- **Index**: `USING hnsw (embedding vector_cosine_ops)`
- **Parameters**: `m = 16, ef_construction = 100`

## Output

- Console summary with performance comparison table
- ChromaDB data directory (can be deleted after benchmark)

## Notes

- ChromaDB data directories are created fresh for each run
- PostgreSQL table is recreated for each method
- Memory measurements include both process-specific and system-wide metrics
