import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List

from utils.benchmark_utils import (
    connect_and_get_pid,
    temp_image_dir, ResourceMonitor, BenchmarkResult,
)
from utils.plot_utils import save_single_run_csv, generate_plots_b6

OUTPUT_DIR = Path(__file__).parent / "output"
DATA_DIR = Path(__file__).parent.parent / "data" / "CUSTOMER_IMAGES"

BACKENDS = ["embed_anything", "ort"]
DEFAULT_MODELS = [
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-large-patch14",
    "google/siglip-large-patch16-384"
]


# =============================================================================
# Data Helpers
# =============================================================================

def get_all_image_paths(limit: int, shuffle: bool = True) -> List[str]:
    """Gather image paths from DATA_DIR, recycling if necessary."""
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        imgs.extend(list(DATA_DIR.rglob(ext)))

    if not imgs:
        raise ValueError(f"No images found in {DATA_DIR}. Please ensure TPCx-AI data is present.")

    imgs = [str(p.absolute()) for p in imgs]
    
    if shuffle:
        random.shuffle(imgs)

    if len(imgs) < limit:
        mult = (limit // len(imgs)) + 1
        imgs = (imgs * mult)[:limit]
    else:
        imgs = imgs[:limit]

    return imgs


# =============================================================================
# PostgreSQL Functions
# =============================================================================

def warmup_pg_connection_images(conn, backend: str, model: str, warmup_dir: str):
    """
    Warm up a PostgreSQL connection using a single-image embed_image_directory call.

    Ensures the connection is fully established, pg_gembed is loaded, and the
    requested backend/model is cached in shared memory before measurement begins.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM embed_image_directory(%s, %s, %s) LIMIT 1",
        (backend, model, warmup_dir)
    )
    cur.fetchall()
    conn.commit()
    cur.close()


def run_pg_benchmark(conn, backend: str, model: str, tmp_dir_path: str):
    """Run the in-DB image embedding benchmark for a specific backend and model."""
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM embed_image_directory(%s, %s, %s) LIMIT 1",
        (backend, model, tmp_dir_path)
    )
    _ = cur.fetchall()
    cur.close()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark 6: Image Embedding Backends Comparison")
    parser.add_argument('--sizes', type=int, nargs='+', required=True,
                        help='Number of images to benchmark')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Run identifier for file naming')
    args = parser.parse_args()

    test_sizes = args.sizes
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    methods = [f"{b}_{m}" for m in DEFAULT_MODELS for b in BACKENDS]

    print(f"\nBenchmark 6: Image Embedding Backends Comparison")
    print(f"Run ID     : {run_id}")
    print(f"Sizes      : {test_sizes}")
    print(f"Backends   : {BACKENDS}")
    print(f"Models     : {DEFAULT_MODELS}")

    py_pid = os.getpid()

    image_paths_by_size = {size: get_all_image_paths(size) for size in test_sizes}

    print("Warming up...")
    warmup_paths = get_all_image_paths(8)
    with temp_image_dir(warmup_paths, prefix="temp_b6_warmup") as warmup_tmp:
        warmup_tmp_path = str(warmup_tmp.absolute())
        for model in DEFAULT_MODELS:
            for backend in BACKENDS:
                conn, _ = connect_and_get_pid()
                try:
                    warmup_pg_connection_images(conn, backend, model, warmup_tmp_path)
                finally:
                    conn.close()

    results_by_size = {size: {m: None for m in methods} for size in test_sizes}

    for size in test_sizes:
        print(f"\nSize: {size}", flush=True)
        image_paths = image_paths_by_size[size]

        with temp_image_dir(image_paths, prefix=f"temp_b6_size_{size}") as tmp_dir:
            tmp_path = str(tmp_dir.absolute())

            for model in DEFAULT_MODELS:
                for backend in BACKENDS:
                    method_name = f"{backend}_{model}"
                    conn, pg_pid = connect_and_get_pid()
                    try:
                        warmup_pg_connection_images(conn, backend, model, tmp_path)

                        elapsed, _, stats = ResourceMonitor.measure(
                            py_pid, pg_pid,
                            lambda: run_pg_benchmark(conn, backend, model, tmp_path)
                        )

                        conn.commit()
                        results_by_size[size][method_name] = BenchmarkResult(
                            time_s=elapsed, stats=stats
                        )
                        print(f"  {method_name}: {elapsed:.3f}s ({size / elapsed:.1f} img/s)",
                              flush=True)
                    except Exception as e:
                        print(f"  {method_name}: FAILED: {e}", flush=True)
                    finally:
                        conn.close()

    # Collect metrics and save
    all_results = []
    for size in test_sizes:
        entry = {'size': size}
        for method_name in methods:
            result = results_by_size[size][method_name]
            if result is None:
                continue
            entry[method_name] = {
                'time_s': result.time_s,
                'throughput': size / result.time_s if result.time_s > 0 else 0,
                'py_cpu': result.stats.py_cpu,
                'py_mem_delta': result.stats.py_delta_mb,
                'py_mem_peak': result.stats.py_peak_mb,
                'pg_cpu': result.stats.pg_cpu,
                'pg_mem_delta': result.stats.pg_delta_mb,
                'pg_mem_peak': result.stats.pg_peak_mb,
                'sys_cpu': result.stats.sys_cpu,
                'sys_mem': result.stats.sys_mem_mb,
            }
        all_results.append(entry)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_single_run_csv(all_results, OUTPUT_DIR, run_id, methods)
    print("Run completed!")


if __name__ == "__main__":
    main()
