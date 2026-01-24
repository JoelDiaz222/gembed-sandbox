import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev, median, quantiles
from typing import Callable, List

import embed_anything
import grpc
import psutil
import psycopg2
import requests
import tei_pb2 as pb2
import tei_pb2_grpc as pb2_grpc
from data.loader import get_review_texts
from embed_anything import EmbeddingModel, WhichModel
from plot_utils import save_results_csv, generate_plots
from psycopg2.extras import execute_values

# Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'joeldiaz',
    'user': 'joeldiaz',
}

TEST_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048]
EMBED_ANYTHING_MODEL = "Qdrant/all-MiniLM-L6-v2-onnx"
RUNS_PER_SIZE = 5

OUTPUT_DIR = Path(__file__).parent / "output"

model_cache = {}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ResourceStats:
    """Resource statistics for Python and PostgreSQL processes."""
    py_delta_mb: float
    py_peak_mb: float
    py_cpu: float
    pg_delta_mb: float
    pg_peak_mb: float
    pg_cpu: float
    sys_mem_mb: float
    sys_cpu: float


@dataclass
class BenchmarkResult:
    time_s: float
    stats: ResourceStats


# =============================================================================
# Resource Monitoring
# =============================================================================

class ResourceMonitor:
    """Monitor resource usage for Python and PostgreSQL processes."""

    def __init__(self, py_pid: int, pg_pid: int = None):
        self.py_pid = py_pid
        self.pg_pid = pg_pid

        # Python process
        self.py_process = psutil.Process(py_pid)
        try:
            mem_info = self.py_process.memory_full_info()
            self.py_baseline = mem_info.uss if hasattr(mem_info, 'uss') else mem_info.rss
        except (psutil.AccessDenied, AttributeError):
            mem_info = self.py_process.memory_info()
            self.py_baseline = mem_info.rss
        self.py_process.cpu_percent()

        # PostgreSQL process
        self.pg_process = None
        self.pg_baseline = 0
        if pg_pid:
            try:
                self.pg_process = psutil.Process(pg_pid)
                try:
                    mem_info = self.pg_process.memory_full_info()
                    self.pg_baseline = mem_info.uss if hasattr(mem_info, 'uss') else mem_info.rss
                except (psutil.AccessDenied, AttributeError):
                    mem_info = self.pg_process.memory_info()
                    self.pg_baseline = mem_info.rss
                self.pg_process.cpu_percent()
            except psutil.NoSuchProcess:
                self.pg_process = None

        time.sleep(0.1)

    @staticmethod
    def measure(py_pid: int, pg_pid: int, func: Callable):
        """Measure resource usage for both Python and PostgreSQL processes."""
        monitor = ResourceMonitor(py_pid, pg_pid)
        result = func()

        # Python process final stats
        try:
            mem_info = monitor.py_process.memory_full_info()
            py_peak = mem_info.uss if hasattr(mem_info, 'uss') else mem_info.rss
        except (psutil.AccessDenied, AttributeError):
            mem_info = monitor.py_process.memory_info()
            py_peak = mem_info.rss
        py_cpu = monitor.py_process.cpu_percent()

        py_baseline_mb = monitor.py_baseline / (1024 * 1024)
        py_peak_mb = py_peak / (1024 * 1024)
        py_delta_mb = py_peak_mb - py_baseline_mb

        # PostgreSQL process final stats
        pg_delta_mb = 0.0
        pg_peak_mb = 0.0
        pg_cpu = 0.0
        if monitor.pg_process:
            try:
                try:
                    mem_info = monitor.pg_process.memory_full_info()
                    pg_peak = mem_info.uss if hasattr(mem_info, 'uss') else mem_info.rss
                except (psutil.AccessDenied, AttributeError):
                    mem_info = monitor.pg_process.memory_info()
                    pg_peak = mem_info.rss
                pg_cpu = monitor.pg_process.cpu_percent()

                pg_baseline_mb = monitor.pg_baseline / (1024 * 1024)
                pg_peak_mb = pg_peak / (1024 * 1024)
                pg_delta_mb = pg_peak_mb - pg_baseline_mb
            except psutil.NoSuchProcess:
                pass

        # System-wide stats
        sys_mem = psutil.virtual_memory()
        sys_mem_mb = sys_mem.used / (1024 * 1024)
        sys_cpu = psutil.cpu_percent()

        stats = ResourceStats(
            py_delta_mb=py_delta_mb,
            py_peak_mb=py_peak_mb,
            py_cpu=py_cpu,
            pg_delta_mb=pg_delta_mb,
            pg_peak_mb=pg_peak_mb,
            pg_cpu=pg_cpu,
            sys_mem_mb=sys_mem_mb,
            sys_cpu=sys_cpu
        )

        return result, stats


# =============================================================================
# Embedding Clients
# =============================================================================

class EmbedAnythingGrpcClient:
    """gRPC client for EmbedAnything server."""

    def __init__(self, address: str = "localhost:50051"):
        self.channel = grpc.insecure_channel(address)
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


class EmbedAnythingDirectClient:
    """Direct Python client for EmbedAnything."""

    def __init__(self):
        pass

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings directly via Python."""
        model = self._get_model(EMBED_ANYTHING_MODEL)
        data = embed_anything.embed_query(texts, embedder=model)
        return [item.embedding for item in data]

    @staticmethod
    def _get_model(model_name: str):
        """Get or load model from cache."""
        if model_name not in model_cache:
            model_cache[model_name] = EmbeddingModel.from_pretrained_onnx(
                WhichModel.Bert,
                hf_model_id=model_name
            )
        return model_cache[model_name]


# =============================================================================
# PostgreSQL Functions
# =============================================================================

def connect_and_get_pid():
    """Connect to PostgreSQL and get backend PID."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("SELECT pg_backend_pid();")
    pid = cur.fetchone()[0]
    cur.close()
    return conn, pid


def setup_database(conn):
    """Initialize database schema."""
    cur = conn.cursor()
    cur.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE EXTENSION IF NOT EXISTS pg_gembed;
                DROP TABLE IF EXISTS embeddings_test;
                CREATE TABLE embeddings_test
                (
                    id        SERIAL PRIMARY KEY,
                    text      TEXT,
                    embedding vector(384)
                );
                """)
    conn.commit()
    cur.close()


def truncate_table(conn):
    """Clear test table."""
    cur = conn.cursor()
    cur.execute("TRUNCATE embeddings_test;")
    conn.commit()
    cur.close()


def benchmark_internal_db_gen(conn, texts: List[str], provider: str, model: str) -> float:
    """Benchmark pg_gembed internal generation."""
    cur = conn.cursor()
    start = time.perf_counter()

    sql = """
          WITH input_data AS (SELECT %s::text[] AS texts)
          INSERT
          INTO embeddings_test (text, embedding)
          SELECT t, e
          FROM input_data,
               unnest(texts, embed_texts(%s, %s, texts)) AS x(t, e)
          """
    cur.execute(sql, (texts, provider, model))

    conn.commit()
    elapsed = time.perf_counter() - start
    cur.close()

    return elapsed


def benchmark_external_client_gen(conn, texts: List[str], embed_fn: Callable) -> float:
    """Benchmark external embedding client."""
    cur = conn.cursor()
    start = time.perf_counter()

    embeddings = embed_fn(texts)

    values = [(text, embedding) for text, embedding in zip(texts, embeddings)]
    execute_values(
        cur,
        "INSERT INTO embeddings_test (text, embedding) VALUES %s",
        values,
        template="(%s, %s::vector)",
        page_size=len(texts)
    )

    conn.commit()
    elapsed = time.perf_counter() - start
    cur.close()

    return elapsed


def run_benchmark_iteration(conn, py_pid: int, pg_pid: int, benchmark_fn: Callable) -> BenchmarkResult:
    """Run a single benchmark iteration with resource monitoring."""
    truncate_table(conn)

    def execute_benchmark():
        return benchmark_fn(conn)

    elapsed, stats = ResourceMonitor.measure(py_pid, pg_pid, execute_benchmark)

    return BenchmarkResult(time_s=elapsed, stats=stats)


def run_multiple_iterations(conn, py_pid: int, pg_pid: int, benchmark_fn: Callable,
                            runs: int) -> List[BenchmarkResult]:
    """Run multiple benchmark iterations."""
    results = []
    for _ in range(runs):
        result = run_benchmark_iteration(conn, py_pid, pg_pid, benchmark_fn)
        results.append(result)
    return results


def run_method_with_fresh_connection(texts: List[str], benchmark_fn: Callable,
                                     runs: int) -> List[BenchmarkResult]:
    """Create a fresh connection, warm it up, then run all benchmark iterations."""
    # Create fresh connection for this method - pg_pid is the backend PID from pg_backend_pid()
    conn, pg_pid = connect_and_get_pid()
    py_pid = os.getpid()

    from data.loader import get_review_texts

    try:
        # Warm up this specific method on the fresh connection
        warmup_texts = get_review_texts(8, shuffle=False)
        truncate_table(conn)
        benchmark_fn(conn, warmup_texts)

        # Now run the actual benchmark iterations
        results = run_multiple_iterations(
            conn, py_pid, pg_pid,
            lambda c: benchmark_fn(c, texts),
            runs
        )
        return results
    finally:
        conn.close()


# =============================================================================
# Statistics Functions
# =============================================================================

def safe_stdev(values: List[float]) -> float:
    """Calculate standard deviation, returning 0 for single values."""
    return stdev(values) if len(values) > 1 else 0.0


def calc_iqr(values: List[float]) -> float:
    """Calculate interquartile range (Q3 - Q1)."""
    if len(values) < 4:
        return 0.0
    q = quantiles(values, n=4)
    return q[2] - q[0]  # Q3 - Q1


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


def compute_metrics(size: int, results: List[BenchmarkResult]) -> dict:
    """Compute mean/std and median/IQR for all metrics from benchmark results."""
    times = [r.time_s for r in results]
    py_cpu = [r.stats.py_cpu for r in results]
    py_delta = [r.stats.py_delta_mb for r in results]
    py_peak = [r.stats.py_peak_mb for r in results]
    pg_cpu = [r.stats.pg_cpu for r in results]
    pg_delta = [r.stats.pg_delta_mb for r in results]
    pg_peak = [r.stats.pg_peak_mb for r in results]
    sys_cpu = [r.stats.sys_cpu for r in results]
    sys_mem = [r.stats.sys_mem_mb for r in results]

    return {
        # Throughput (mean-based)
        'throughput': size / mean(times),
        'throughput_std': size / mean(times) * safe_stdev(times) / mean(times) if len(times) > 1 else 0,
        'throughput_median': size / median(times),
        'throughput_iqr': size / median(times) * calc_iqr(times) / median(times) if len(times) >= 4 else 0,
        # Time
        'time_s': mean(times),
        'time_s_std': safe_stdev(times),
        'time_s_median': median(times),
        'time_s_iqr': calc_iqr(times),
        # Python process
        'py_cpu': mean(py_cpu),
        'py_cpu_std': safe_stdev(py_cpu),
        'py_cpu_median': median(py_cpu),
        'py_cpu_iqr': calc_iqr(py_cpu),
        'py_mem_delta': mean(py_delta),
        'py_mem_delta_std': safe_stdev(py_delta),
        'py_mem_delta_median': median(py_delta),
        'py_mem_delta_iqr': calc_iqr(py_delta),
        'py_mem_peak': mean(py_peak),
        'py_mem_peak_std': safe_stdev(py_peak),
        'py_mem_peak_median': median(py_peak),
        'py_mem_peak_iqr': calc_iqr(py_peak),
        # PostgreSQL process
        'pg_cpu': mean(pg_cpu),
        'pg_cpu_std': safe_stdev(pg_cpu),
        'pg_cpu_median': median(pg_cpu),
        'pg_cpu_iqr': calc_iqr(pg_cpu),
        'pg_mem_delta': mean(pg_delta),
        'pg_mem_delta_std': safe_stdev(pg_delta),
        'pg_mem_delta_median': median(pg_delta),
        'pg_mem_delta_iqr': calc_iqr(pg_delta),
        'pg_mem_peak': mean(pg_peak),
        'pg_mem_peak_std': safe_stdev(pg_peak),
        'pg_mem_peak_median': median(pg_peak),
        'pg_mem_peak_iqr': calc_iqr(pg_peak),
        # System-wide
        'sys_cpu': mean(sys_cpu),
        'sys_cpu_std': safe_stdev(sys_cpu),
        'sys_cpu_median': median(sys_cpu),
        'sys_cpu_iqr': calc_iqr(sys_cpu),
        'sys_mem': mean(sys_mem),
        'sys_mem_std': safe_stdev(sys_mem),
        'sys_mem_median': median(sys_mem),
        'sys_mem_iqr': calc_iqr(sys_mem),
    }


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    # Initialize setup connection (just for schema setup)
    conn, _ = connect_and_get_pid()
    setup_database(conn)
    conn.close()

    # Initialize external clients (these are reused across connections)
    http_client = EmbedAnythingHttpClient()
    grpc_client = EmbedAnythingGrpcClient()
    direct_client = EmbedAnythingDirectClient()

    try:
        print_detailed_header()

        all_results = []

        for size in TEST_SIZES:
            texts = get_review_texts(size, shuffle=False)

            print(f"Size: {size}", flush=True)

            # Benchmark PG EmbedAnything (fresh connection per method)
            pg_local_results = run_method_with_fresh_connection(
                texts,
                lambda c, t: benchmark_internal_db_gen(
                    c, t, "embed_anything", EMBED_ANYTHING_MODEL
                ),
                RUNS_PER_SIZE
            )

            # Benchmark PG gRPC (fresh connection per method)
            pg_grpc_results = run_method_with_fresh_connection(
                texts,
                lambda c, t: benchmark_internal_db_gen(
                    c, t, "grpc", EMBED_ANYTHING_MODEL
                ),
                RUNS_PER_SIZE
            )

            # Benchmark Direct Python (fresh connection per method)
            ext_direct_results = run_method_with_fresh_connection(
                texts,
                lambda c, t: benchmark_external_client_gen(
                    c, t, direct_client.embed
                ),
                RUNS_PER_SIZE
            )

            # Benchmark External gRPC (fresh connection per method)
            ext_grpc_results = run_method_with_fresh_connection(
                texts,
                lambda c, t: benchmark_external_client_gen(
                    c, t, grpc_client.embed
                ),
                RUNS_PER_SIZE
            )

            # Benchmark External HTTP (fresh connection per method)
            ext_http_results = run_method_with_fresh_connection(
                texts,
                lambda c, t: benchmark_external_client_gen(
                    c, t, http_client.embed
                ),
                RUNS_PER_SIZE
            )

            print_result("PG local", pg_local_results)
            print_result("PG gRPC", pg_grpc_results)
            print_result("Ext Direct", ext_direct_results)
            print_result("Ext gRPC", ext_grpc_results)
            print_result("Ext HTTP", ext_http_results)
            print()

            # Store metrics for summary and plots
            all_results.append({
                'size': size,
                'pg_local': compute_metrics(size, pg_local_results),
                'pg_grpc': compute_metrics(size, pg_grpc_results),
                'ext_direct': compute_metrics(size, ext_direct_results),
                'ext_grpc': compute_metrics(size, ext_grpc_results),
                'ext_http': compute_metrics(size, ext_http_results),
            })

        # Print summary
        print("=" * 105)

        avg_pg_local = mean([r['pg_local']['throughput'] for r in all_results])
        avg_pg_grpc = mean([r['pg_grpc']['throughput'] for r in all_results])
        avg_ext_direct = mean([r['ext_direct']['throughput'] for r in all_results])
        avg_ext_grpc = mean([r['ext_grpc']['throughput'] for r in all_results])
        avg_ext_http = mean([r['ext_http']['throughput'] for r in all_results])

        print("\nAverage Throughput Across All Sizes (texts/sec):")
        print(f"  PG local:          {avg_pg_local:.2f}")
        print(f"  PG gRPC:           {avg_pg_grpc:.2f}")
        print(f"  External Direct:   {avg_ext_direct:.2f}")
        print(f"  External gRPC:     {avg_ext_grpc:.2f}")
        print(f"  External HTTP:     {avg_ext_http:.2f}")

        # Save results to CSV and generate plots
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        methods = ['pg_local', 'pg_grpc', 'ext_direct', 'ext_grpc', 'ext_http']

        save_results_csv(all_results, OUTPUT_DIR, timestamp, methods)
        generate_plots(all_results, OUTPUT_DIR, timestamp, methods)

    finally:
        grpc_client.close()


if __name__ == "__main__":
    main()
