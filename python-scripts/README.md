# Embedding Benchmarks

This repository contains benchmarks comparing different approaches for generating and storing embeddings with
PostgreSQL, pgvector, pg_gembed and ChromaDB.

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
├── 2_pg_gembed-vs-vectordbs/           # Benchmark 2: Generation + storage comparison
├── 3_mono-store-vs-poly-store/         # Benchmark 3: Architecture comparison
├── 4_tpcx-ai-uc4/                      # Benchmark 4: TPC-x AI Use Case 4
├── 5_tpcx-ai-uc9/                      # Benchmark 5: TPC-x AI Use Case 9
├── 6_extensibility/                    # Benchmark 6: Image embedding backend comparison
├── 7_portability/                      # Benchmark 7: Multi-adapter portability
└── 8_measure_gembed_overhead/          # Benchmark 8: Gembed stack overhead analysis
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

### Benchmark 2: PG+pgvector vs Vector DBs (ChromaDB, Qdrant)

**Directory:** `2_pg_gembed-vs-vectordbs/`

Compares both embedding generation location and storage systems:

- **PG Local**: pg_gembed generates embeddings inside PostgreSQL
- **PG gRPC**: Same as above with gRPC warmup
- **Chroma**: EmbedAnything in Python process (no server) → stored in ChromaDB
- **Qdrant**: EmbedAnything in Python process (no server) → stored in Qdrant (requires Docker)

```bash
PYTHONPATH=.:proto python3.13 2_pg_gembed-vs-vectordbs/benchmark.py
```

### Benchmark 3: Mono-Store vs Poly-Store Architecture

**Directory:** `3_mono-store-vs-poly-store/`

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

- **Mono-Store (PG)**: All data + embeddings in PostgreSQL with pg_gembed
- **Poly-Store (PG, Chroma)**: Metadata in PostgreSQL + embeddings in ChromaDB
- **Poly-Store (PG, Qdrant)**: Metadata in PostgreSQL + embeddings in Qdrant
  
  All systems use an HNSW index with matching parameters for fair comparison: `m = 16`, `ef_construction = 100`.

```bash
PYTHONPATH=.:proto python3.13 3_mono-store-vs-poly-store/benchmark.py
```

### Benchmark 8: Gembed Stack Overhead Analysis

**Directory:** `8_measure_gembed_overhead/`

> **Requires:** `pg_gembed` compiled from the `feature/telemetry` branch.

Quantifies the latency overhead introduced by the full Gembed stack
(C extension → C→Rust FFI → EmbedAnything backend) by reading internal
timestamp checkpoints written to `/tmp/gembed_telemetry_log`.

Measured overhead components (all in µs):

| Component | What it covers |
|---|---|
| `validate_backend_us` | `validate_backend()` FFI call |
| `validate_model_us` | `validate_model()` FFI call |
| `pre_ffi_overhead_us` | C processing before the FFI boundary |
| `ffi_roundtrip_us` | Full C→Rust→C crossing |
| `rs_dispatch_us` | Rust backend dispatch before EmbedAnything |
| `pure_embedding_us` | Actual EmbedAnything / Candle inference |
| `rs_to_c_return_us` | Rust→C return path |
| `post_ffi_overhead_us` | C processing after FFI returns |
| **Stack overhead %** | `100 × (wall − inference) / wall` |

Produces a stacked-bar chart (PDF + PNG) and a LaTeX table with
mean ± std over 10 orchestrated runs.

```bash
# Single run
PYTHONPATH=. venv/bin/python 8_measure_gembed_overhead/benchmark.py \
    --sizes 1 8 64 512 4096

# Full orchestrated run (10 × each size, generates plot + LaTeX table)
./orchestrator.sh 8_measure_gembed_overhead
```

See [`8_measure_gembed_overhead/README.md`](8_measure_gembed_overhead/README.md)
for full setup instructions.

## Benchmark Data

The benchmarks can use **real product review data** from the [TPCx-AI](http://tpc.org/tpcx-ai/default5.asp) benchmark
suite. If the data file is not present, benchmarks will automatically fall back to synthetic data.

### Quick Start (Synthetic Data)

Benchmarks work out of the box with synthetic data—no additional setup required.

### Using TPCx-AI Data (Recommended)

For more realistic benchmarks, obtain the TPCx-AI review and image data:

1. **Download TPCx-AI toolkit** from [tpc.org](http://tpc.org/tpcx-ai/default5.asp) (requires free registration)

2. **Fix dictionary dependency** for image generation:
   ```bash
   cd /path/to/tpcx-ai-v2.0.0
   ln -s lib/pdgf/dicts .
   ```

3. **Generate the data** using Docker:

   **Review Data (Use Case 4):**
   ```bash
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

   **Image Data (Use Case 9):**
   ```bash
   docker run --platform linux/amd64 --rm \
     -v "$(pwd)":/tpcx-ai \
     -w /tpcx-ai eclipse-temurin:8-jdk bash -c \
     "apt-get update -qq && \
      apt-get install -y -qq libx11-6 libxext6 libxrender1 libxtst6 libxi6 \
      libgl1 libxrandr2 libxcursor1 libxinerama1 libxfixes3 xvfb 2>/dev/null && \
      printf '\nYES\n' | xvfb-run java -jar lib/pdgf/pdgf.jar \
      -l data-gen/config/tpcxai-schema-noplugins.xml \
      -l data-gen/config/tpcxai-generation.xml \
      -ns -sf 1 -s CUSTOMER_IMAGES"
   ```

4. **Copy the data** to the data directory:
   ```bash
   cp /path/to/tpcx-ai-v2.0.0/output/Review.psv data/
   cp -r /path/to/tpcx-ai-v2.0.0/output/CUSTOMER_IMAGES data/
   ```

The `Review.psv` file contains 200,000 product reviews (~135MB). The `CUSTOMER_IMAGES` directory contains facial images organized by customer name.

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

```shell
docker pull qdrant/qdrant

docker run --name qdrant -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
```