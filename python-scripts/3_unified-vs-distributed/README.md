# Benchmark 3: Unified vs Distributed Architecture

## Purpose

This benchmark compares two architectural approaches for managing embeddings at scale:

1. **Unified Architecture**: All data and embeddings in PostgreSQL with pg_gembed
2. **Distributed Architecture**: Metadata in PostgreSQL + embeddings in ChromaDB

## Scenarios

### Scenario 1: Cold Start (Insert + Embed)

Simulates initial data loading or migration:
- Fresh insert of all data (timed)
- Data insertion + embedding generation measured together
- Common use case: Initial system setup, data migration, batch loading

**What's measured:**
- Total time for insertion + embedding generation
- CPU and memory usage for complete operation

### Scenario 2: Pre-existing Data (Update Only)

Simulates updating embeddings for existing records:
- Data is pre-inserted (setup phase, not timed)
- Only embedding generation is measured
- Common use case: Updating embeddings after model change or adding new embedding column

**What's measured:**
- Time to generate embeddings for existing data
- CPU and memory usage during embedding generation

## Methods Tested

| Method | Data Storage | Embedding Storage | Embedding Generation |
|--------|-------------|-------------------|---------------------|
| PG Unified | PostgreSQL | PostgreSQL (pgvector) | pg_gembed (internal) |
| PG Distributed | PostgreSQL | ChromaDB | EmbedAnything (in-process) |

## Test Data

Realistic product data with multiple text fields:
- Product names, descriptions, categories, tags
- Simulated reviews
- Concatenated into single text for embedding

## Running

```bash
PYTHONPATH=.:proto python 3_unified-vs-distributed/benchmark.py
```

## Output

Results are grouped by scenario:

### Scenario 1 Results
- Focus on total system throughput
- Includes I/O overhead from insertions
- Reflects cold-start performance

### Scenario 2 Results
- Focus on embedding generation performance
- Lower times indicate more efficient embedding generation

## Interpretation

- **Scenario 1** shows practical end-to-end performance for data loading
- **Scenario 2** helps understand raw embedding performance
- Compare methods within each scenario (not across scenarios)

## Notes

- Each scenario sets up its own data independently
- Warmup batches are not counted in final metrics
- ChromaDB directories are cleaned up between runs
- pgvector HNSW index uses `m = 16, ef_construction = 100` to match ChromaDB's defaults
