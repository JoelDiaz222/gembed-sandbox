# Embedding Benchmarks

This repository contains benchmarks comparing different approaches for generating and storing embeddings with PostgreSQL, pgvector, pg_gembed and ChromaDB.

## Project Structure

```
python-scripts/
├── data/                               # Benchmark data and loader
│   └── loader.py                       # Data loading utilities (with synthetic fallback)
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

**Scenario 1: Cold Start (Insert + Embed)**
- Fresh insert of all data
- Measures total time including data insertion
- Simulates initial data loading or migration

**Scenario 2: Pre-existing Data (Update Only)**
- Data already exists in both systems
- Only measures embedding generation time
- Simulates updating embeddings after model change

Each scenario compares:
- **Unified**: All data + embeddings in PostgreSQL with pg_gembed
- **Distributed**: Metadata in PostgreSQL + embeddings in ChromaDB

```bash
PYTHONPATH=.:proto python3.13 3_unified-vs-distributed/benchmark.py
```

## Benchmark Data

The benchmarks can use **real product review data** from the [TPCx-AI](http://tpc.org/tpcx-ai/default5.asp) benchmark suite. If the data file is not present, benchmarks will automatically fall back to synthetic data.

### Quick Start (Synthetic Data)

Benchmarks work out of the box with synthetic data—no additional setup required.

### Using TPCx-AI Data (Recommended)

For more realistic benchmarks, obtain the TPCx-AI review data:

1. **Download TPCx-AI toolkit** from [tpc.org](http://tpc.org/tpcx-ai/default5.asp) (requires free registration)

2. **Generate the data** using Docker:
   ```bash
   cd /path/to/tpcx-ai-v2.0.0
   docker run --platform linux/amd64 --rm \
     -v "$(pwd)":/tpcx-ai \
     -w /tpcx-ai eclipse-temurin:8-jdk bash -c \
     "apt-get update -qq && \
      apt-get install -y -qq libx11-6 libxext6 libxrender1 libxtst6 libxi6 \
      libgl1 libxrandr2 libxcursor1 libxinerama1 libxfixes3 xvfb 2>/dev/null && \
      printf '\nYES\n' | xvfb-run java -jar lib/pdgf/pdgf.jar \
      -l data-gen/config/tpcxai-schema-noplugins.xml \
      -l data-gen/config/tpcxai-generation.xml \
      -ns -sf 1 -s Review"
   ```

3. **Copy the Review table** to the data directory:
   ```bash
   cp /path/to/tpcx-ai-v2.0.0/output/Review.psv data/
   ```

The Review.psv file contains 200,000 product reviews (~135MB) with realistic text and spam labels.

### Data Loading

All benchmarks use `data/loader.py`:

```python
from data.loader import get_review_texts, is_using_synthetic

# Check data source
if is_using_synthetic():
    print("Using synthetic data")

# Get 1000 legitimate reviews
texts = get_review_texts(1000)
```
