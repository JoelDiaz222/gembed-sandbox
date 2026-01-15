# Benchmark 2: PG+pgvector vs Vector DBs (ChromaDB, Qdrant)

## Purpose

This benchmark compares embedding generation and storage approaches:

1. **PostgreSQL + pgvector + pg_gembed**: Internal embedding generation inside the database
2. **Vector Databases (ChromaDB, Qdrant) + EmbedAnything**: Embedding generation in the Python process (no server)

This tests both where embeddings are generated (internal vs external) and where they are stored (relational DB vs
purpose-built vector DB).

## Methods Tested

| Method   | Storage               | Embedding Generation              |
|----------|-----------------------|-----------------------------------|
| PG Local | PostgreSQL + pgvector | Inside PostgreSQL (pg_gembed)     |
| PG gRPC  | PostgreSQL + pgvector | External with a gRPC server       |
| Chroma   | ChromaDB              | In Python process (EmbedAnything) |
| Qdrant   | Qdrant                | In Python process (EmbedAnything) |

## Key Differences from Benchmark 1

- **Embedding location matters**: pg_gembed generates embeddings inside PostgreSQL; ChromaDB/Qdrant use EmbedAnything in
  Python process
- **Storage comparison**: Measures how different storage backends (PostgreSQL, ChromaDB, Qdrant) handle embeddings
- **Vector DB integration**: Tests persistent clients with HNSW indexing

## Prerequisites

1. **PostgreSQL**: Running locally with pgvector and pg_gembed installed.
2. **Qdrant**: Running in a Docker container.
   ```bash
   docker pull qdrant/qdrant
   docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant
   ```

## Running

1. Start the gRPC server (required for pg_gembed embedding):

```bash
PYTHONPATH=.:proto python3.13 servers/grpc_server.py
```

2. Run the benchmark (both modes):

```bash
PYTHONPATH=.:proto python3.13 2_pg_gembed-vs-vectordbs/benchmark.py
```

3. Or run specific mode:

```bash
# With indexing (HNSW enabled)
PYTHONPATH=.:proto python3.13 2_pg_gembed-vs-vectordbs/benchmark.py --with-index

# Without indexing (minimal HNSW/deactivated index)
PYTHONPATH=.:proto python3.13 2_pg_gembed-vs-vectordbs/benchmark.py --without-index
```

## Index Modes

The benchmark runs in two modes to compare indexing overhead.

### With Index (`output/with_index/`)

- **pgvector**: HNSW index with `m = 16, ef_construction = 100`
- **ChromaDB**: HNSW index configured with `max_neighbors = 16, ef_construction = 100`
- **Qdrant**: HNSW index configured with `m = 16, ef_construct = 100`

### Without Index (`output/without_index/`)

- **pgvector**: No index (sequential scan)
- **ChromaDB**: Minimal HNSW index (`max_neighbors = 2, ef_construction = 0`)
- **Qdrant**: HNSW index disabled (`m = 0`)

## Output

Results are saved to separate directories:

- `output/with_index/` - Results with full indexing enabled
- `output/without_index/` - Results with minimal/no indexing

## Notes

- PostgreSQL table is recreated for each method
- ChromaDB data directories are created fresh for each run
- Qdrant collection is recreated for each run
- Memory measurements include both process-specific and system-wide metrics