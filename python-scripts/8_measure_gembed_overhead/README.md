# Benchmark 8: Gembed Overhead Analysis

Measures the internal overhead of the **Gembed stack** by instrumenting
`pg_gembed` with high-resolution timestamps at key execution boundaries
and correlating them with Python-side wall-clock timing.

> **Prerequisite:** `pg_gembed` must be compiled from the
> `feature/telemetry` branch, which writes checkpoints to
> `/tmp/gembed_telemetry_log` automatically.

---

## What Is Being Measured

Every call to `embed_texts()` passes through several layers, each of which
contributes latency:

```
SQL → C extension (pg_gembed.c)
       ├─ validate_backend()      (C → Rust FFI)
       ├─ validate_model()        (C → Rust FFI)
       └─ embed_batch_text()
            ├─ generate_embeddings()    ← C → Rust FFI boundary
            │    └─ backend.embed()
            │         └─ runtime.block_on(backend.embed())
            │               ← EmbedAnything / Candle inference
            └─ construct_vector_array()
```

The benchmark captures these **telemetry checkpoints** (written to
`/tmp/gembed_telemetry_log` as tab-separated `<epoch_µs>\t<label>\t<n>`
lines):

### C Layer (`pg_gembed.c` / `internal.c`)

| Checkpoint | Description |
|---|---|
| `c_ext_entry_embed_texts` | SQL function `embed_texts()` entered |
| `c_validate_backend_start` / `_done` | `validate_backend()` FFI call |
| `c_validate_model_start` / `_done` | `validate_model()` FFI call |
| `c_embed_batch_text_entry` | `embed_batch_text()` entered |
| `c_embed_batch_text_pre_embed` | `InputData` built, about to cross FFI |
| `c_pre_ffi` / `c_post_ffi` | `generate_embeddings()` C→Rust boundary |
| `c_embed_batch_text_done` | Vector array built, returning |
| `c_ext_exit_embed_texts` | SQL function returning |

### Rust Layer (`lib.rs` / `embed_anything.rs`)

| Checkpoint | Description |
|---|---|
| `rs_pre_embed` / `rs_post_embed` | `backend.embed()` wrapper |
| `rs_ea_model_cache_hit` | Model found in thread-local cache |
| `rs_ea_model_load_start` / `_done` | Model load (first call only) |
| `rs_ea_embed_texts_start` / `_done` | `runtime.block_on(backend.embed())` |

---

## Derived Intervals

From the checkpoint timestamps the benchmark derives:

| Interval | Formula |
|---|---|
| `total_c_ext_us` | `ext_exit - ext_entry` |
| `validation_total_us` | `validate_model_done - validate_backend_start` |
| `validate_backend_us` | `validate_backend_done - validate_backend_start` |
| `validate_model_us` | `validate_model_done - validate_model_start` |
| `pre_ffi_overhead_us` | `c_pre_ffi - ext_entry` |
| `ffi_roundtrip_us` | `c_post_ffi - c_pre_ffi` |
| `rs_dispatch_us` | `rs_ea_embed_texts_start - rs_pre_embed` |
| `pure_embedding_us` | `rs_ea_embed_texts_done - rs_ea_embed_texts_start` |
| `rs_to_c_return_us` | `c_post_ffi - rs_post_embed` |
| `post_ffi_overhead_us` | `ext_exit - c_post_ffi` |
| `wall_time_us` | Python `time.perf_counter()` |

**Stack overhead %** = `100 × (wall_time - pure_embedding) / wall_time`

---

## Setup

### 1. Build pg_gembed from the telemetry branch

```bash
cd pg_gembed
git checkout feature/telemetry
make && sudo make install
sudo systemctl restart postgresql
```

The build always defines `GEMBED_TELEMETRY` and the Rust `telemetry`
module; no extra flags are required.

### 2. Verify the log file is being written

```bash
psql -c "SELECT embed_texts('embed_anything', \
         'sentence-transformers/all-MiniLM-L6-v2', \
         ARRAY['hello world']::text[]);" my_db

cat /tmp/gembed_telemetry_log
```

You should see lines like:

```
1744543200123456	c_ext_entry_embed_texts	0
1744543200123512	c_validate_backend_start	0
...
```

---

## Running

### Single run (manual)

```bash
cd python-scripts
PYTHONPATH=. venv/bin/python 8_measure_gembed_overhead/benchmark.py \
    --sizes 1 8 64 512 4096
```

### Via orchestrator (recommended — 10 runs, mean ± std)

```bash
cd python-scripts
./orchestrator.sh 8_measure_gembed_overhead
```

---

## Output

| File | Description |
|---|---|
| `output/benchmark_<run_id>_run.csv` | Single-run raw CSV (concatenated by orchestrator) |
| `output/benchmark_<timestamp>.csv` | Concatenated multi-run CSV |
| `output/overhead_breakdown_<timestamp>.pdf/png` | Stacked-bar overhead breakdown chart |
| `output/overhead_table_<timestamp>.tex` | LaTeX table (mean ± std, overhead %) |

### Plot

A stacked bar chart showing each overhead component as a fraction of total
wall-clock time, across batch sizes. Error bars (±1σ) are drawn on the
total wall-clock height.

### LaTeX Table

A `booktabs`-style table suitable for direct inclusion in a thesis or paper:

```latex
\input{8_measure_gembed_overhead/output/overhead_table_<timestamp>.tex}
```

Rows: overhead checkpoints + total wall time + **stack overhead %**.
Columns: batch sizes.

---

## Notes

- Measurements use the **warm-model-cache** path (`rs_ea_model_cache_hit`
  events). Cold-start model loading is excluded by the per-size pre-run.
- Only the `embed_texts` (batch) SQL function is instrumented;
  `embed_text` (single) is untouched.
- Backend: `embed_anything` / Model: `sentence-transformers/all-MiniLM-L6-v2`.
