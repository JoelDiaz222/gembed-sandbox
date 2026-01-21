import csv
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev, median, quantiles
from typing import Callable, List

import embed_anything
import grpc
import matplotlib.pyplot as plt
import psutil
import psycopg2
import requests
import tei_pb2 as pb2
import tei_pb2_grpc as pb2_grpc
from embed_anything import EmbeddingModel, WhichModel
from psycopg2.extras import execute_values

# Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'joeldiaz',
    'user': 'joeldiaz',
}

TEST_SIZES = [16, 32, 64, 128, 256, 512]
BATCH_SIZE = 32
EMBED_ANYTHING_MODEL = "Qdrant/all-MiniLM-L6-v2-onnx"
RUNS_PER_SIZE = 5  # Number of runs per test size

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output"

# Global model cache for direct Python calls
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


def make_inputs(n: int) -> List[str]:
    """Load review texts from TPCx-AI dataset."""
    from data.loader import get_review_texts
    return get_review_texts(n, shuffle=True)


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_internal_db_gen(conn, texts: List[str], provider: str, model: str) -> float:
    """Benchmark pg_gembed internal generation."""
    cur = conn.cursor()
    start = time.perf_counter()

    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i:i + BATCH_SIZE]

        sql = """
              INSERT INTO embeddings_test (text, embedding)
              SELECT t, e
              FROM unnest(%s::text[]) t,
                   unnest(embed_texts(%s, %s, %s::text[])) e \
              """
        cur.execute(sql, (chunk, provider, model, chunk))

    conn.commit()
    elapsed = time.perf_counter() - start
    cur.close()

    return elapsed


def benchmark_external_client_gen(conn, texts: List[str], embed_fn: Callable) -> float:
    """Benchmark external embedding client."""
    cur = conn.cursor()
    start = time.perf_counter()

    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i:i + BATCH_SIZE]
        embeddings = embed_fn(chunk)

        values = [(text, embedding) for text, embedding in zip(chunk, embeddings)]
        execute_values(
            cur,
            "INSERT INTO embeddings_test (text, embedding) VALUES %s",
            values,
            template="(%s, %s::vector)"
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

    try:
        # Warm up this specific method on the fresh connection
        warmup_texts = make_inputs(8)
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

def print_header():
    """Print benchmark results header."""
    # Column widths
    lbl_w = 12
    time_w = 12
    col_w = 11

    print("Benchmark Results:", flush=True)
    # Include a small med-marker column between label and values
    med_w = 7
    header = (
            "  " +
            f"{'':{lbl_w}}{'':{med_w}} | {'Time (s) μ±σ':>{time_w}} | {'Py Δ MB μ±σ':>{col_w}} | {'Py Peak μ±σ':>{col_w}} | {'Py CPU% μ±σ':>{col_w}} | "
            f"{'PG Δ MB μ±σ':>{col_w}} | {'PG Peak μ±σ':>{col_w}} | {'PG CPU% μ±σ':>{col_w}} | {'Sys MB μ±σ':>{col_w}} | {'Sys CPU% μ±σ':>{col_w}}"
    )
    print(header, flush=True)
    print("=" * len(header), flush=True)


def print_result(label: str, results: List[BenchmarkResult]):
    """Print aggregated results from multiple runs with ± std dev."""
    times = [r.time_s for r in results]
    py_delta = [r.stats.py_delta_mb for r in results]
    py_peak = [r.stats.py_peak_mb for r in results]
    py_cpu = [r.stats.py_cpu for r in results]
    pg_delta = [r.stats.pg_delta_mb for r in results]
    pg_peak = [r.stats.pg_peak_mb for r in results]
    pg_cpu = [r.stats.pg_cpu for r in results]
    sys_mem = [r.stats.sys_mem_mb for r in results]
    sys_cpu = [r.stats.sys_cpu for r in results]

    def fmt_mean(values: List[float], precision: int = 1) -> str:
        avg = mean(values)
        std = safe_stdev(values)
        if precision == 3:
            return f"{avg:.3f}±{std:.3f}"
        elif precision == 0:
            return f"{avg:.0f}±{std:.0f}"
        else:
            return f"{avg:.1f}±{std:.1f}"

    def fmt_median_iqr(values: List[float], precision: int = 1) -> str:
        med = median(values)
        iqr = calc_iqr(values)
        if precision == 3:
            return f"{med:.3f}±{iqr:.3f}"
        elif precision == 0:
            return f"{med:.0f}±{iqr:.0f}"
        else:
            return f"{med:.1f}±{iqr:.1f}"

    # Print mean±std
    row_fmt = (
        "  {label:<12}{med:>7} | {time:>12} | {pyd:>11} | {pyp:>11} | {pyc:>11} | {pgd:>11} | {pgp:>11} | {pgc:>11} | {sysm:>11} | {sysc:>11}"
    )
    print(row_fmt.format(
        label=label,
        med='',
        time=fmt_mean(times, 3),
        pyd=fmt_mean(py_delta),
        pyp=fmt_mean(py_peak),
        pyc=fmt_mean(py_cpu),
        pgd=fmt_mean(pg_delta),
        pgp=fmt_mean(pg_peak),
        pgc=fmt_mean(pg_cpu),
        sysm=fmt_mean(sys_mem, 0),
        sysc=fmt_mean(sys_cpu),
    ), flush=True)
    # Also print median ± IQR for more robust central tendency
    print(row_fmt.format(
        label=label,
        med=' (med)',
        time=fmt_median_iqr(times, 3),
        pyd=fmt_median_iqr(py_delta),
        pyp=fmt_median_iqr(py_peak),
        pyc=fmt_median_iqr(py_cpu),
        pgd=fmt_median_iqr(pg_delta),
        pgp=fmt_median_iqr(pg_peak),
        pgc=fmt_median_iqr(pg_cpu),
        sysm=fmt_median_iqr(sys_mem, 0),
        sysc=fmt_median_iqr(sys_cpu),
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
        print_header()

        all_results = []

        for size in TEST_SIZES:
            texts = make_inputs(size)

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
        save_results_csv(all_results)
        generate_plots(all_results)

    finally:
        grpc_client.close()


def save_results_csv(all_results: List[dict]):
    """Save benchmark results to CSV file with mean and std values."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"benchmark_{timestamp}.csv"

    methods = ['pg_local', 'pg_grpc', 'ext_direct', 'ext_grpc', 'ext_http']
    metrics = ['throughput', 'time_s',
               'py_cpu', 'py_mem_delta', 'py_mem_peak',
               'pg_cpu', 'pg_mem_delta', 'pg_mem_peak',
               'sys_cpu', 'sys_mem']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: size, then method_metric, _std, _median, _iqr for each combination
        header = ['size']
        for method in methods:
            for metric in metrics:
                header.append(f"{method}_{metric}")
                header.append(f"{method}_{metric}_std")
                header.append(f"{method}_{metric}_median")
                header.append(f"{method}_{metric}_iqr")
        writer.writerow(header)

        for r in all_results:
            row = [r['size']]
            for method in methods:
                for metric in metrics:
                    row.append(r[method].get(metric))
                    row.append(r[method].get(f"{metric}_std", 0))
                    row.append(r[method].get(f"{metric}_median", 0))
                    row.append(r[method].get(f"{metric}_iqr", 0))
            writer.writerow(row)

    print(f"\nResults saved to: {csv_path}")


def generate_plots(all_results: List[dict]):
    """Generate comparison plots for throughput, CPU, and memory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sizes = [r['size'] for r in all_results]
    methods = ['pg_local', 'pg_grpc', 'ext_direct', 'ext_grpc', 'ext_http']
    labels = ['PG Local', 'PG gRPC', 'In-Process', 'Ext gRPC', 'Ext HTTP']
    colors = ['#2ecc71', '#27ae60', '#3498db', '#e74c3c', '#c0392b']
    markers = ['o', 's', '^', 'd', 'x']

    # Plot 1: Throughput comparison
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        y_vals = [r[method]['throughput_median'] for r in all_results]
        y_errs = [r[method]['throughput_iqr'] for r in all_results]
        plt.errorbar(sizes, y_vals, yerr=y_errs, fmt=f'{marker}-', label=label,
                     linewidth=2, color=color, capsize=3, capthick=1)
    plt.xlabel('Number of Texts')
    plt.ylabel('Throughput (texts/sec)')
    plt.title(f'Embedding Generation: Throughput (Median ± IQR, batch size={BATCH_SIZE})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"throughput_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Python Process CPU Usage
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        y_vals = [r[method]['py_cpu_median'] for r in all_results]
        y_errs = [r[method]['py_cpu_iqr'] for r in all_results]
        plt.errorbar(sizes, y_vals, yerr=y_errs, fmt=f'{marker}-', label=label,
                     linewidth=2, color=color, capsize=3, capthick=1)
    plt.xlabel('Number of Texts')
    plt.ylabel('Python Process CPU Usage (%)')
    plt.title(f'Python Process CPU Usage (Median ± IQR, batch size={BATCH_SIZE})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"py_cpu_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: PostgreSQL Process CPU Usage
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        y_vals = [r[method]['pg_cpu_median'] for r in all_results]
        y_errs = [r[method]['pg_cpu_iqr'] for r in all_results]
        plt.errorbar(sizes, y_vals, yerr=y_errs, fmt=f'{marker}-', label=label,
                     linewidth=2, color=color, capsize=3, capthick=1)
    plt.xlabel('Number of Texts')
    plt.ylabel('PostgreSQL Process CPU Usage (%)')
    plt.title(f'PostgreSQL Process CPU Usage (Median ± IQR, batch size={BATCH_SIZE})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"pg_cpu_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 4: System CPU Usage
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        y_vals = [r[method]['sys_cpu_median'] for r in all_results]
        y_errs = [r[method]['sys_cpu_iqr'] for r in all_results]
        plt.errorbar(sizes, y_vals, yerr=y_errs, fmt=f'{marker}-', label=label,
                     linewidth=2, color=color, capsize=3, capthick=1)
    plt.xlabel('Number of Texts')
    plt.ylabel('System CPU Usage (%)')
    plt.title(f'Embedding Generation: System CPU (Median ± IQR, batch size={BATCH_SIZE})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"cpu_system_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 5: Python Process Peak Memory
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        y_vals = [r[method]['py_mem_peak_median'] for r in all_results]
        y_errs = [r[method]['py_mem_peak_iqr'] for r in all_results]
        plt.errorbar(sizes, y_vals, yerr=y_errs, fmt=f'{marker}-', label=label,
                     linewidth=2, color=color, capsize=3, capthick=1)
    plt.xlabel('Number of Texts')
    plt.ylabel('Python Peak Memory (MB)')
    plt.title(f'Python Process Peak Memory (Median ± IQR, batch size={BATCH_SIZE})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"py_memory_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 6: PostgreSQL Process Peak Memory
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        y_vals = [r[method]['pg_mem_peak'] for r in all_results]
        y_errs = [r[method]['pg_mem_peak_std'] for r in all_results]
        plt.errorbar(sizes, y_vals, yerr=y_errs, fmt=f'{marker}-', label=label,
                     linewidth=2, color=color, capsize=3, capthick=1)
    plt.xlabel('Number of Texts')
    plt.ylabel('PostgreSQL Peak Memory (MB)')
    plt.title(f'PostgreSQL Process Peak Memory (Median ± IQR, batch size={BATCH_SIZE})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"pg_memory_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 7: System Memory
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        y_vals = [r[method]['sys_mem'] for r in all_results]
        y_errs = [r[method]['sys_mem_std'] for r in all_results]
        plt.errorbar(sizes, y_vals, yerr=y_errs, fmt=f'{marker}-', label=label,
                     linewidth=2, color=color, capsize=3, capthick=1)
    plt.xlabel('Number of Texts')
    plt.ylabel('System Memory (MB)')
    plt.title(f'Embedding Generation: System Memory (Median ± IQR, batch size={BATCH_SIZE})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"memory_system_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 8: Summary bar chart (2x4 grid) with error bars
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle(f'Average Metrics (batch size={BATCH_SIZE})', fontsize=14)
    x_pos = range(len(methods))

    # Throughput
    avgs = [mean([r[m]['throughput'] for r in all_results]) for m in methods]
    stds = [safe_stdev([r[m]['throughput'] for r in all_results]) for m in methods]
    bars = axes[0, 0].bar(x_pos, avgs, yerr=stds, color=colors, capsize=3)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(labels)
    axes[0, 0].set_ylabel('Throughput (texts/sec)')
    axes[0, 0].set_title('Throughput')
    axes[0, 0].bar_label(bars, fmt='%.1f')
    axes[0, 0].tick_params(axis='x', rotation=15)

    # Python CPU
    avgs = [mean([r[m]['py_cpu'] for r in all_results]) for m in methods]
    stds = [safe_stdev([r[m]['py_cpu'] for r in all_results]) for m in methods]
    bars = axes[0, 1].bar(x_pos, avgs, yerr=stds, color=colors, capsize=3)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(labels)
    axes[0, 1].set_ylabel('CPU Usage (%)')
    axes[0, 1].set_title('Python CPU')
    axes[0, 1].bar_label(bars, fmt='%.1f')
    axes[0, 1].tick_params(axis='x', rotation=15)

    # PostgreSQL CPU
    avgs = [mean([r[m]['pg_cpu'] for r in all_results]) for m in methods]
    stds = [safe_stdev([r[m]['pg_cpu'] for r in all_results]) for m in methods]
    bars = axes[0, 2].bar(x_pos, avgs, yerr=stds, color=colors, capsize=3)
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(labels)
    axes[0, 2].set_ylabel('CPU Usage (%)')
    axes[0, 2].set_title('PostgreSQL CPU')
    axes[0, 2].bar_label(bars, fmt='%.1f')
    axes[0, 2].tick_params(axis='x', rotation=15)

    # System CPU
    avgs = [mean([r[m]['sys_cpu'] for r in all_results]) for m in methods]
    stds = [safe_stdev([r[m]['sys_cpu'] for r in all_results]) for m in methods]
    bars = axes[0, 3].bar(x_pos, avgs, yerr=stds, color=colors, capsize=3)
    axes[0, 3].set_xticks(x_pos)
    axes[0, 3].set_xticklabels(labels)
    axes[0, 3].set_ylabel('CPU Usage (%)')
    axes[0, 3].set_title('System CPU')
    axes[0, 3].bar_label(bars, fmt='%.1f')
    axes[0, 3].tick_params(axis='x', rotation=15)

    # Python Memory Peak
    avgs = [mean([r[m]['py_mem_peak'] for r in all_results]) for m in methods]
    stds = [safe_stdev([r[m]['py_mem_peak'] for r in all_results]) for m in methods]
    bars = axes[1, 0].bar(x_pos, avgs, yerr=stds, color=colors, capsize=3)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_ylabel('Memory (MB)')
    axes[1, 0].set_title('Python Peak Mem')
    axes[1, 0].bar_label(bars, fmt='%.1f')
    axes[1, 0].tick_params(axis='x', rotation=15)

    # PostgreSQL Memory Peak
    avgs = [mean([r[m]['pg_mem_peak'] for r in all_results]) for m in methods]
    stds = [safe_stdev([r[m]['pg_mem_peak'] for r in all_results]) for m in methods]
    bars = axes[1, 1].bar(x_pos, avgs, yerr=stds, color=colors, capsize=3)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_ylabel('Memory (MB)')
    axes[1, 1].set_title('PostgreSQL Peak Mem')
    axes[1, 1].bar_label(bars, fmt='%.1f')
    axes[1, 1].tick_params(axis='x', rotation=15)

    # System Memory
    avgs = [mean([r[m]['sys_mem'] for r in all_results]) for m in methods]
    stds = [safe_stdev([r[m]['sys_mem'] for r in all_results]) for m in methods]
    bars = axes[1, 2].bar(x_pos, avgs, yerr=stds, color=colors, capsize=3)
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(labels)
    axes[1, 2].set_ylabel('Memory (MB)')
    axes[1, 2].set_title('System Memory')
    axes[1, 2].bar_label(bars, fmt='%.0f')
    axes[1, 2].tick_params(axis='x', rotation=15)

    # Time
    avgs = [mean([r[m]['time_s'] for r in all_results]) for m in methods]
    stds = [safe_stdev([r[m]['time_s'] for r in all_results]) for m in methods]
    bars = axes[1, 3].bar(x_pos, avgs, yerr=stds, color=colors, capsize=3)
    axes[1, 3].set_xticks(x_pos)
    axes[1, 3].set_xticklabels(labels)
    axes[1, 3].set_ylabel('Time (s)')
    axes[1, 3].set_title('Avg Time')
    axes[1, 3].bar_label(bars, fmt='%.2f')
    axes[1, 3].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"summary_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
