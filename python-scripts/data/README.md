# Benchmark Data

This directory contains the data loader for embedding benchmarks. The loader supports both **TPCx-AI real data** (recommended) and **synthetic fallback data**.

## Quick Start

Benchmarks work immediately with synthetic data—no setup required. For more realistic results, add the TPCx-AI review data as described below.

## Obtaining TPCx-AI Data (Optional)

The TPCx-AI benchmark suite provides realistic product review data for AI/ML workloads.

### Steps

1. **Download TPCx-AI toolkit** from [tpc.org/tpcx-ai](http://tpc.org/tpcx-ai/default5.asp)
   - Free registration required
   - Download version 2.0.0 or later

2. **Generate data using Docker** (PDGF requires Java 8 + X11 libs):
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

3. **Copy Review.psv to this directory**:
   ```bash
   cp /path/to/tpcx-ai-v2.0.0/output/Review.psv /path/to/python-scripts/data/
   ```

### Review.psv Format

| Column | Type | Description |
|--------|------|-------------|
| ID | int | Unique review identifier |
| spam | int | Spam label (0=legitimate, 1=spam) |
| text | str | Review text content |

**Statistics** (SF=1):
- Total reviews: 200,000
- Legitimate reviews (~70%): 140,048  
- Spam reviews (~30%): 59,952
- File size: ~135MB

## loader.py

Python module for loading benchmark data with automatic fallback.

### Functions

```python
from data.loader import get_review_texts, is_using_synthetic, make_inputs

# Check if using synthetic or real data
is_using_synthetic() -> bool

# Get n review texts
get_review_texts(n, shuffle=True, legitimate_only=True) -> List[str]

# Get reviews with spam labels
get_reviews_with_labels(n, shuffle=True, legitimate_only=True) -> List[Tuple[str, int]]

# Drop-in replacement for benchmarks
make_inputs(n) -> List[str]
```

### Usage

```python
from data.loader import get_review_texts, is_using_synthetic

# Will print a note if falling back to synthetic data
texts = get_review_texts(1000)

# Check data source
if is_using_synthetic():
    print("Using synthetic data - add Review.psv for realistic benchmarks")
```
