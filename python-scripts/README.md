# Embedding Benchmarks

This repository contains benchmarks comparing different approaches for generating and storing embeddings with PostgreSQL, pgvector, pg_gembed and ChromaDB.

## Project Structure

```
python-scripts/
├── proto/                              # gRPC protocol definitions
│   ├── tei.proto                       # TEI protocol definition
│   ├── tei_pb2.py                      # Generated protobuf classes
│   └── tei_pb2_grpc.py                 # Generated gRPC stubs
├── servers/                            # Shared embedding servers
│   ├── grpc_server.py                  # gRPC embedding server
│   └── http_server.py                  # HTTP embedding server
├── 1_internal-vs-external-gen/         # Benchmark 1: Embedding generation methods
├── 2_pg_gembed-vs-chromadb/            # Benchmark 2: Generation + storage comparison
└── 3_unified-vs-distributed/           # Benchmark 3: Architecture comparison
```

## Setup

1. Create and activate a virtual environment:

```bash
python3.13 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure PostgreSQL is running with pgvector and pg_gembed extensions installed.

## Proto Regeneration (if needed)

Only regenerate if `tei.proto` changes:

```bash
cd proto
python3.13 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. tei.proto
```

## Running Servers

### gRPC Server
```bash
PYTHONPATH=.:proto python3.13 servers/grpc_server.py
```

### HTTP Server
```bash
PYTHONPATH=.:proto python3.13 servers/http_server.py
```

## Benchmarks

### Benchmark 1: Internal vs External Embedding Generation

**Directory:** `1_internal-vs-external-gen/`

Compares embedding generation methods:
- **PG Local**: PostgreSQL with pg_gembed (internal embedding generation)
- **PG gRPC**: PostgreSQL with pg_gembed (internal) + gRPC warmup
- **In-Process**: EmbedAnything in Python process (no server)
- **External gRPC**: gRPC server → PostgreSQL storage
- **External HTTP**: HTTP server → PostgreSQL storage

```bash
PYTHONPATH=.:proto python3.13 1_internal-vs-external-gen/benchmark.py
```

### Benchmark 2: PG+pgvector vs ChromaDB Generation + Storage

**Directory:** `2_pg_gembed-vs-chromadb/`

Compares both embedding generation location and storage systems:
- **PG Local**: pg_gembed generates embeddings inside PostgreSQL
- **PG gRPC**: Same as above with gRPC warmup
- **Chroma**: EmbedAnything in Python process (no server) → stored in ChromaDB

```bash
PYTHONPATH=.:proto python3.13 2_pg_gembed-vs-chromadb/benchmark.py
```

### Benchmark 3: Unified vs Distributed Architecture

**Directory:** `3_unified-vs-distributed/`

Compares architectural patterns with two scenarios:

**Scenario 1: Pre-existing Data (Update Only)**
- Data already exists in both systems
- Only measures embedding generation time
- Simulates updating embeddings for existing records

**Scenario 2: Cold Start (Insert + Embed)**
- Fresh insert of all data
- Measures total time including data insertion
- Simulates initial data loading

Each scenario compares:
- **Unified**: All data + embeddings in PostgreSQL with pg_gembed
- **Distributed**: Metadata in PostgreSQL + embeddings in ChromaDB

```bash
PYTHONPATH=.:proto python3.13 3_unified-vs-distributed/benchmark.py
```
