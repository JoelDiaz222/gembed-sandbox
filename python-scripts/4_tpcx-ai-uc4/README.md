# Benchmark 4: TPCx-AI Use Case 4 - Spam Classification (Serving)

## Purpose

This benchmark evaluates and compares the performance of a "Unified" database approach against specialized Vector Databases for a serving-heavy workload, modeled after TPCx-AI Use Case 4 (Spam Classification).

The core task involves:
1.  **Ingestion Phase**: Storing and indexing a dataset of reviews with known labels (Spam/Ham).
2.  **Serving Phase**: Processing a stream of new reviews by generating their embeddings and performing a k-Nearest Neighbors (k-NN) search against the ingested set to determine their label.

## Methods Tested

| Method | Component | Embedding Location | Search Engine | Description |
|--------|-----------|--------------------|---------------|-------------|
| **PG Unified** | PostgreSQL | In-DB (`pg_gembed`) | `pgvector` (HNSW) | All-in-one approach. Text is sent to DB; DB generates embedding and searches in the same transaction. |
| **Chroma** | ChromaDB | Application (Python) | Chroma (HNSW) | Traditional App-side embedding. Embeddings generated in Python, then queried against Chroma. |
| **Qdrant** | Qdrant | Application (Python) | Qdrant (HNSW) | Traditional App-side embedding. Embeddings generated in Python, then queried against Qdrant container. |

## Key Features

-   **Realistic Serving Simulation**: Simulates an application querying for classification.
-   **In-DB Embedding**: Tests the capability of PostgreSQL to handle the compute-intensive embedding generation alongside the search.
-   **Resource Monitoring**: Tracks CPU and Memory usage for Python (App), PostgreSQL, and Docker containers (Qdrant).

## Prerequisites

-   **Python 3.10+**
-   **PostgreSQL** with extensions:
    -   `vector` (pgvector)
    -   `pg_gembed`
-   **Docker** (for Qdrant) running a container named `qdrant` (exposed on port 6333).
-   **Dependencies**: Install via `pip install -r ../requirements.txt` (ensure `psycopg2`, `embed-anything`, `chromadb`, `qdrant-client`, `docker`, `psutil` are included).

## Running the Benchmark

1.  Ensure PostgreSQL is running and accessible.
2.  Ensure Qdrant is running:
    ```bash
    docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
    ```
3.  Run the benchmark script:
    ```bash
    # Optionally set DB credentials
    export PG_DBNAME=joeldiaz
    
    python 4_tpcx-ai-uc4/benchmark.py
    ```

## Output

-   **Console**: Real-time progress and summary statistics.
-   **Files** (in `output/`):
    -   CSV report containing throughput and resource metrics for all batch sizes.
    -   Plots comparing throughput across the three methods.
