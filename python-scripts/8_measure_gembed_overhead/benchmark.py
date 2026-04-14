#!/usr/bin/env python3
"""
Benchmark 8: Gembed Overhead Analysis
======================================

Measures the per-request overhead of the Gembed stack (C extension +
C→Rust FFI + EmbedAnything Rust backend) by correlating:

  - Python-side wall-clock time for each embed_texts() call
  - Internal timestamp checkpoints written by pg_gembed to
    /dev/shm/gembed_telemetry_log  (populated only when pg_gembed is compiled
    from the feature/telemetry branch)

The benchmark follows the same single-run pattern as benchmarks 1–7.
Multiple statistical runs are handled by the orchestrator, which invokes
this script N times and concatenates the resulting CSV files.

Telemetry checkpoints (embed_texts batch path only):

  C layer  (pg_gembed.c / internal.c):
    c_ext_entry_embed_texts          – SQL function entered
    c_validate_backend_start/done    – validate_backend() FFI call
    c_validate_model_start/done      – validate_model() FFI call
    c_embed_batch_text_entry         – embed_batch_text() entered
    c_embed_batch_text_pre_embed     – InputData ready, crossing FFI
    c_pre_ffi / c_post_ffi           – generate_embeddings() boundary
    c_embed_batch_text_done          – vector array built, returning
    c_ext_exit_embed_texts           – SQL function returning

  Rust layer  (lib.rs / embed_anything.rs):
    rs_pre_embed / rs_post_embed     – backend.embed() wrapper
    rs_ea_model_cache_hit            – model in thread-local cache
    rs_ea_model_load_start/done      – model load (first call only)
    rs_ea_embed_texts_start/done     – runtime.block_on(backend.embed())

Usage:
  python benchmark.py --sizes 1 8 64 512 4096 [--run-id <id>]
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from data.loader import get_review_texts
from utils.benchmark_utils import (
    connect_and_get_pid,
    warmup_pg_connection,
    EMBED_ANYTHING_MODEL,
)

OUTPUT_DIR = Path(__file__).parent / "output"

BACKEND = "embed_anything"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TLOG = "/dev/shm/gembed_telemetry_log"

# Overhead interval columns saved to CSV (all in µs)
INTERVAL_KEYS = [
    "wall_time_us",
    "total_c_ext_us",
    "validation_total_us",
    "validate_backend_us",
    "validate_model_us",
    "pre_ffi_overhead_us",
    "ffi_roundtrip_us",
    "rs_dispatch_us",
    "pure_embedding_us",
    "rs_to_c_return_us",
    "post_ffi_overhead_us",
]

# =============================================================================
# Telemetry helpers
# =============================================================================

import subprocess


def truncate_log() -> None:
    try:
        subprocess.run(
            ["psql", "-d", "postgres", "-c", "COPY (SELECT 1) TO PROGRAM '> /dev/shm/gembed_telemetry_log';"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )
    except Exception:
        pass


def read_log() -> Dict[str, int]:
    """Return {label: most-recent-epoch-µs} from the log."""
    # First, postgres owns the file, so we chmod it using postgres
    subprocess.run(
        ["psql", "-d", "postgres", "-c", "COPY (SELECT 1) TO PROGRAM 'chmod 666 /dev/shm/gembed_telemetry_log';"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False
    )

    result: Dict[str, int] = {}
    try:
        with open(TLOG) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    try:
                        result[parts[1]] = int(parts[0])
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    except PermissionError:
        pass
    return result


def compute_intervals(ts: Dict[str, int], size: int, wall_us: float) -> Dict:
    def gap(a: str, b: str) -> Optional[float]:
        return ts[b] - ts[a] if a in ts and b in ts else None

    return {
        "size": size,
        "wall_time_us": wall_us,
        "total_c_ext_us": gap("c_ext_entry_embed_texts", "c_ext_exit_embed_texts"),
        "validate_backend_us": gap("c_validate_backend_start", "c_validate_backend_done"),
        "validate_model_us": gap("c_validate_model_start", "c_validate_model_done"),
        "validation_total_us": gap("c_validate_backend_start", "c_validate_model_done"),
        "pre_ffi_overhead_us": gap("c_ext_entry_embed_texts", "c_pre_ffi"),
        "ffi_roundtrip_us": gap("c_pre_ffi", "c_post_ffi"),
        "rs_dispatch_us": gap("rs_pre_embed", "rs_ea_embed_texts_start"),
        "pure_embedding_us": gap("rs_ea_embed_texts_start", "rs_ea_embed_texts_done"),
        "rs_to_c_return_us": gap("rs_post_embed", "c_post_ffi"),
        "post_ffi_overhead_us": gap("c_post_ffi", "c_ext_exit_embed_texts"),
    }


# =============================================================================
# PostgreSQL helpers
# =============================================================================

def run_embed_texts(conn, texts: List[str]) -> None:
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM embed_texts(%s, %s, %s::text[]) LIMIT 1",
        (BACKEND, MODEL, texts),
    )
    cur.fetchall()
    cur.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark 8: Gembed overhead analysis via telemetry"
    )
    parser.add_argument("--sizes", type=int, nargs="+", required=True,
                        help="Batch sizes (number of texts) to benchmark")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run identifier for output file naming")
    args = parser.parse_args()

    test_sizes = args.sizes
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nBenchmark 8 — Gembed Overhead Analysis")
    print(f"Run ID : {run_id}")
    print(f"Backend: {BACKEND}  Model: {MODEL}")
    print(f"Sizes  : {test_sizes}")
    print(f"Log    : {TLOG}")

    texts_by_size = {s: get_review_texts(s, shuffle=True) for s in test_sizes}

    # ------------------------------------------------------------------
    # Warm-up: two passes so the model is in the thread-local cache
    # ------------------------------------------------------------------
    print("\nWarming up...")
    conn, _ = connect_and_get_pid()
    warmup_pg_connection(conn, backend=BACKEND, model_name=MODEL)
    run_embed_texts(conn, get_review_texts(1, shuffle=True))
    conn.commit()
    truncate_log()

    # ------------------------------------------------------------------
    # Measurement loop — one timed call per size
    # ------------------------------------------------------------------
    rows = []

    for size in test_sizes:
        texts = texts_by_size[size]
        print(f"\nSize: {size}", flush=True)

        try:
            t0 = time.perf_counter()
            run_embed_texts(conn, texts)
            wall_us = (time.perf_counter() - t0) * 1_000_000
            conn.commit()

            time.sleep(0.05)
            ts = read_log()
            truncate_log()

            if not ts:
                print(
                    "  [WARN] No telemetry events — is pg_gembed compiled from "
                    "the feature/telemetry branch?",
                    file=sys.stderr,
                )
                continue

            row = compute_intervals(ts, size, wall_us)
            rows.append(row)

            def fmt(k):
                v = row.get(k)
                return f"{v:>9.1f} µs" if v is not None else "        N/A"

            pct_overhead = None
            if row.get("pure_embedding_us") and wall_us > 0:
                pct_overhead = 100.0 * (1.0 - row["pure_embedding_us"] / wall_us)

            print(f"  Wall (Python)       : {wall_us:>9.1f} µs")
            print(f"  Total C ext         : {fmt('total_c_ext_us')}")
            print(f"  Validation (B+M)    : {fmt('validation_total_us')}")
            print(f"    validate_backend  : {fmt('validate_backend_us')}")
            print(f"    validate_model    : {fmt('validate_model_us')}")
            print(f"  Pre-FFI overhead    : {fmt('pre_ffi_overhead_us')}")
            print(f"  FFI roundtrip (C↔Rs): {fmt('ffi_roundtrip_us')}")
            print(f"    Rust dispatch     : {fmt('rs_dispatch_us')}")
            print(f"    EA embed_texts    : {fmt('pure_embedding_us')}")
            print(f"    Rs→C return       : {fmt('rs_to_c_return_us')}")
            print(f"  Post-FFI overhead   : {fmt('post_ffi_overhead_us')}")
            if pct_overhead is not None:
                print(f"  Stack overhead      : {pct_overhead:>8.2f}%  (non-inference time)")

        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            import traceback;
            traceback.print_exc()
            conn.close()

    conn.close()

    # ------------------------------------------------------------------
    # Save run CSV  (compatible with orchestrator concatenation)
    # ------------------------------------------------------------------
    if not rows:
        print("\nNo results to save.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / f"benchmark_{run_id}_run.csv"

    fieldnames = ["size"] + INTERVAL_KEYS

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nRun CSV saved to: {out_csv}")
    print("Run completed!")


if __name__ == "__main__":
    main()
