import argparse
import os
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Callable, List

import grpc
import numpy as np
import requests
import tei_pb2 as pb2
import tei_pb2_grpc as pb2_grpc
from data.loader import get_review_texts
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
from utils.benchmark_utils import (
    EMBED_ANYTHING_MODEL,
    BenchmarkResult, ResourceMonitor,
    EmbedAnythingDirectClient,
    safe_stdev, calc_iqr, compute_metrics,
    connect_and_get_pid, warmup_pg_connection, clear_model_cache,
)
from utils.plot_utils import save_single_run_csv

OUTPUT_DIR = Path(__file__).parent / "output"


# =============================================================================
# Embedding Clients (Benchmark 1-specific: gRPC and HTTP)
# =============================================================================

class EmbedAnythingGrpcClient:
    """gRPC client for EmbedAnything server."""
    UNLIMITED = -1

    def __init__(self, address: str = "localhost:50051"):
        options = [
            ("grpc.max_send_message_length", self.UNLIMITED),
            ("grpc.max_receive_message_length", self.UNLIMITED),
        ]

        # When creating the channel
        self.channel = grpc.insecure_channel(address, options=options)
        self.stub = pb2_grpc.EmbedStub(self.channel)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via gRPC."""
        request = pb2.EmbedBatchRequest(
            inputs=texts,
            truncate=True,
            normalize=True,
            truncation_direction=0,
            model=EMBED_ANYTHING_MODEL
        )
        response = self.stub.EmbedBatch(request)
        return [list(emb.values) for emb in response.embeddings]

    def clear_cache(self):
        try:
            request = pb2.ClearCacheRequest()
            self.stub.ClearCache(request)
        except grpc.RpcError:
            pass

    def close(self):
        self.channel.close()


class EmbedAnythingHttpClient:
    """HTTP client for EmbedAnything server."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via HTTP."""
        response = self.session.post(
            f"{self.base_url}/v1/embed",
            json={
                "model": EMBED_ANYTHING_MODEL,
                "input": texts
            }
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]

    def clear_cache(self):
        try:
            self.session.post(f"{self.base_url}/v1/clear_cache")
        except requests.exceptions.RequestException:
            pass


# =============================================================================
# PostgreSQL Functions
# =============================================================================

def setup_database(conn):
    """Initialize database schema."""
    cur = conn.cursor()
    cur.execute("""
                CREATE
                EXTENSION IF NOT EXISTS vector;
                CREATE
                EXTENSION IF NOT EXISTS pg_gembed;
                DROP TABLE IF EXISTS embeddings_test;
                CREATE TABLE embeddings_test
                (
                    id        SERIAL PRIMARY KEY,
                    text      TEXT,
                    embedding vector(384)
                );
                """)
    cur.close()


def truncate_table(conn):
    """Clear test table."""
    cur = conn.cursor()
    cur.execute("TRUNCATE embeddings_test;")
    cur.close()


def benchmark_internal_db_gen(conn, texts: List[str], backend: str, model: str):
    """Benchmark pg_gembed internal generation."""
    cur = conn.cursor()

    sql = """
          WITH input_data AS (SELECT %s::text[] AS texts)
          INSERT
          INTO embeddings_test (text, embedding)
          SELECT t, e
          FROM input_data,
               unnest(texts, embed_texts(%s, %s, texts)) AS x(t, e)
          """
    cur.execute(sql, (texts, backend, model))

    cur.close()


def benchmark_external_client_gen(conn, texts: List[str], embed_fn: Callable):
    """Benchmark external embedding client."""
    cur = conn.cursor()

    embeddings = embed_fn(texts)

    execute_values(
        cur,
        "INSERT INTO embeddings_test (text, embedding) VALUES %s",
        list(zip(texts, [np.array(e) for e in embeddings])),
        page_size=len(texts)
    )

    cur.close()


def run_benchmark_iteration(conn, py_pid: int, pg_pid: int, benchmark_fn: Callable) -> BenchmarkResult:
    """Run a single benchmark iteration with resource monitoring."""
    truncate_table(conn)

    elapsed, _, stats = ResourceMonitor.measure(py_pid, pg_pid, lambda: benchmark_fn(conn))

    return BenchmarkResult(time_s=elapsed, stats=stats)


def setup_method_connection(texts: List[str], benchmark_fn: Callable, is_external: bool = False):
    """Create and warmup a connection for a method, returning connection and PIDs."""
    conn, pg_pid = connect_and_get_pid()
    if is_external:
        register_vector(conn)
    py_pid = os.getpid()

    # Warm up this specific connection
    warmup_pg_connection(conn)

    # Method-specific warm-up with small data
    warmup_texts = get_review_texts(8, shuffle=True)
    truncate_table(conn)
    benchmark_fn(conn, warmup_texts)

    conn.commit()
    return conn, py_pid, pg_pid


def run_single_iteration(conn, py_pid, pg_pid, texts: List[str],
                         benchmark_fn: Callable) -> BenchmarkResult:
    """Run a single benchmark iteration on an existing connection."""
    truncate_table(conn)
    res = run_benchmark_iteration(conn, py_pid, pg_pid,
                                  lambda c: benchmark_fn(c, texts))
    conn.commit()
    return res


# =============================================================================
# Output Functions
# =============================================================================

def print_header(title):
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def print_detailed_header():
    """Print detailed benchmark results header."""
    lbl_w, time_w, col_w, med_w = 12, 14, 13, 7

    print("\nBenchmark Results:", flush=True)
    header = (
            "  " +
            f"{'':{lbl_w}}{'':{med_w}} | {'Time (s)':>{time_w}} | "
            f"{'Py Δ MB':>{col_w}} | {'Py Peak MB':>{col_w}} | {'Py CPU%':>{col_w}} | "
            f"{'PG Δ MB':>{col_w}} | {'PG Peak MB':>{col_w}} | {'PG CPU%':>{col_w}} | "
            f"{'Sys MB':>{col_w}} | {'Sys CPU%':>{col_w}}"
    )
    print(header, flush=True)
    print("=" * len(header), flush=True)


def print_result(label: str, results: List[BenchmarkResult]):
    """Print aggregated results with mean and median rows."""
    times = [r.time_s for r in results]

    py_deltas = [r.stats.py_delta_mb for r in results]
    py_peaks = [r.stats.py_peak_mb for r in results]
    py_cpus = [r.stats.py_cpu for r in results]

    pg_deltas = [r.stats.pg_delta_mb for r in results]
    pg_peaks = [r.stats.pg_peak_mb for r in results]
    pg_cpus = [r.stats.pg_cpu for r in results]

    sys_mems = [r.stats.sys_mem_mb for r in results]
    sys_cpus = [r.stats.sys_cpu for r in results]

    def fmt(vals, p=1):
        return f"{mean(vals):.{p}f}±{safe_stdev(vals):.{p}f}"

    def fmt_med(vals, p=1):
        return f"{median(vals):.{p}f}±{calc_iqr(vals):.{p}f}"

    # Mean row
    row_fmt = (
        "  {label:<12}{med:>7} | {time:>14} | "
        "{pyd:>13} | {pyp:>13} | {pyc:>13} | "
        "{pgd:>13} | {pgp:>13} | {pgc:>13} | "
        "{sysm:>13} | {sysc:>13}"
    )

    print(row_fmt.format(
        label=label, med='',
        time=fmt(times, 3),
        pyd=fmt(py_deltas), pyp=fmt(py_peaks), pyc=fmt(py_cpus),
        pgd=fmt(pg_deltas), pgp=fmt(pg_peaks), pgc=fmt(pg_cpus),
        sysm=fmt(sys_mems, 0), sysc=fmt(sys_cpus)
    ), flush=True)

    # Median row
    print(row_fmt.format(
        label=label, med=' (med)',
        time=fmt_med(times, 3),
        pyd=fmt_med(py_deltas), pyp=fmt_med(py_peaks), pyc=fmt_med(py_cpus),
        pgd=fmt_med(pg_deltas), pgp=fmt_med(pg_peaks), pgc=fmt_med(pg_cpus),
        sysm=fmt_med(sys_mems, 0), sysc=fmt_med(sys_cpus)
    ), flush=True)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run benchmark with all sizes once (single run)."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Benchmark 1: Internal vs External Generation')
    parser.add_argument('--sizes', type=int, nargs='+', required=True,
                        help='List of test sizes to benchmark')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Run identifier for file naming')
    args = parser.parse_args()

    test_sizes = args.sizes
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Starting Benchmark 1: Internal vs External Generation")
    print(f"Run ID: {run_id}")
    print(f"Sizes: {test_sizes}")

    # Initialize setup connection (just for schema setup)
    conn, _ = connect_and_get_pid()
    setup_database(conn)
    conn.commit()
    conn.close()

    # Initialize external clients (these are reused across connections)
    http_client = EmbedAnythingHttpClient()
    grpc_client = EmbedAnythingGrpcClient()
    direct_client = EmbedAnythingDirectClient()

    try:
        # Pre-load all test data
        test_data = {size: get_review_texts(size, shuffle=True) for size in test_sizes}

        # Define all methods to benchmark
        # Internal methods (pg_local, pg_grpc) do NOT need register_vector.
        # External methods (ext_*) DO need register_vector — handled in setup_method_connection.
        EXTERNAL_METHODS = {'ext_direct', 'ext_grpc', 'ext_http'}

        def make_methods(direct_client, grpc_client, http_client):
            return {
                'pg_local': lambda c, t: benchmark_internal_db_gen(
                    c, t, "embed_anything", EMBED_ANYTHING_MODEL
                ),
                'pg_grpc': lambda c, t: benchmark_internal_db_gen(
                    c, t, "grpc", EMBED_ANYTHING_MODEL
                ),
                'pg_http': lambda c, t: benchmark_internal_db_gen(
                    c, t, "http", EMBED_ANYTHING_MODEL
                ),
                'ext_direct': lambda c, t: benchmark_external_client_gen(
                    c, t, direct_client.embed
                ),
                'ext_grpc': lambda c, t: benchmark_external_client_gen(
                    c, t, grpc_client.embed
                ),
                'ext_http': lambda c, t: benchmark_external_client_gen(
                    c, t, http_client.embed
                ),
            }

        methods = make_methods(direct_client, grpc_client, http_client)
        method_names = list(methods.keys())

        # Initialize results storage (single iteration per size)
        results_by_size = {size: {name: None for name in method_names} for size in test_sizes}

        # Loop through sizes
        for size in test_sizes:
            texts = test_data[size]
            print(f"\nSize: {size}", flush=True)

            # Loop through each method individually
            for method_name, benchmark_fn in methods.items():
                is_ext = method_name in EXTERNAL_METHODS
                conn, py_pid, pg_pid = setup_method_connection(texts, benchmark_fn, is_external=is_ext)
                try:
                    result = run_single_iteration(conn, py_pid, pg_pid, texts, benchmark_fn)
                    results_by_size[size][method_name] = result
                    print(f"  {method_name}: {result.time_s:.2f}s", flush=True)
                    clear_model_cache()
                    grpc_client.clear_cache()
                    http_client.clear_cache()
                finally:
                    conn.close()

        # Compute metrics for each size
        all_results = []
        for size in test_sizes:
            results = results_by_size[size]
            result_entry = {'size': size}

            for method_name in method_names:
                result = results[method_name]
                # For single run, just extract the raw metrics from the BenchmarkResult
                result_entry[method_name] = {
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

            all_results.append(result_entry)

        # Save results to CSV
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_single_run_csv(all_results, OUTPUT_DIR, run_id, method_names)
        print(f"Run completed!")

    finally:
        grpc_client.close()


if __name__ == "__main__":
    main()
