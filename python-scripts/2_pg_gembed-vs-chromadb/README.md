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
PYTHONPATH=.:proto python3.13 servers/grpc_server.py
```

2. Run the benchmark (both modes):
```bash
PYTHONPATH=.:proto python3.13 2_pg_gembed-vs-chromadb/benchmark.py
```

3. Or run specific mode:
```bash
# With indexing (HNSW enabled)
PYTHONPATH=.:proto python3.13 2_pg_gembed-vs-chromadb/benchmark.py --with-index

# Without indexing (minimal HNSW for ChromaDB, no index for pgvector)
PYTHONPATH=.:proto python3.13 2_pg_gembed-vs-chromadb/benchmark.py --without-index
```

## Index Modes

The benchmark runs in two modes to compare indexing overhead.

### With Index (`output/with_index/`)
- **pgvector**: HNSW index with `m = 16, ef_construction = 100`
- **ChromaDB**: HNSW index configured with `max_neighbors = 16, ef_construction = 100` (persisted on the collection)

### Without Index (`output/without_index/`)
- **pgvector**: No index (sequential scan)
- **ChromaDB**: Minimal HNSW index configured with `max_neighbors = 2, ef_construction = 0` to reduce indexing overhead while avoiding unsafe parameter combinations
  - Note: ChromaDB persists an HNSW index configuration on collections; extremely small HNSW parameter combinations (e.g. `max_neighbors = 0`) can cause runtime failures, so we use a safe minimal setting above.
## Output

Results are saved to separate directories:
- `output/with_index/` - Results with full indexing enabled
- `output/without_index/` - Results with minimal/no indexing

## Notes

- ChromaDB data directories are created fresh for each run
- PostgreSQL table is recreated for each method
- Memory measurements include both process-specific and system-wide metrics
