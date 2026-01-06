import csv
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev, median, quantiles
from typing import Callable, List

import chromadb
import embed_anything
import matplotlib.pyplot as plt
import psutil
import psycopg2
from embed_anything import EmbeddingModel, WhichModel

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

# Output directory (set dynamically based on index mode)
OUTPUT_DIR = Path(__file__).parent / "output"

# Global model cache for direct Python calls
model_cache = {}


# Simple wrapper matching Chroma's EmbeddingFunction interface
class EmbeddingWrapper:
    def __init__(self, fn: Callable):
        self._fn = fn

    def __call__(self, input):
        return self._fn(list(input))


@dataclass
class ResourceStats:
    """Resource statistics for both Python and PostgreSQL processes."""
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


def safe_stdev(values: List[float]) -> float:
    """Calculate standard deviation, returning 0 for lists with <2 elements."""
    return stdev(values) if len(values) > 1 else 0.0


def calc_iqr(values: List[float]) -> float:
    """Calculate interquartile range (Q3 - Q1)."""
    if len(values) < 4:
        return 0.0
    q = quantiles(values, n=4)
    return q[2] - q[0]  # Q3 - Q1


# =============================================================================
# Resource Monitoring
# =============================================================================

class ResourceMonitor:
    """Monitor resource usage for Python and PostgreSQL processes."""

    def __init__(self, py_pid: int, pg_pid: int = None):
        self.py_pid = py_pid
        self.pg_pid = pg_pid
        self.py_process = psutil.Process(py_pid)
        self.pg_process = psutil.Process(pg_pid) if pg_pid else None

        # Get Python baseline
        try:
            mem_info = self.py_process.memory_full_info()
            self.py_baseline = mem_info.uss if hasattr(mem_info, 'uss') else mem_info.rss
        except (psutil.AccessDenied, AttributeError):
            mem_info = self.py_process.memory_info()
            self.py_baseline = mem_info.rss

        # Get PostgreSQL baseline if available
        self.pg_baseline = 0
        if self.pg_process:
            try:
                mem_info = self.pg_process.memory_full_info()
                self.pg_baseline = mem_info.uss if hasattr(mem_info, 'uss') else mem_info.rss
            except (psutil.AccessDenied, AttributeError):
                try:
                    mem_info = self.pg_process.memory_info()
                    self.pg_baseline = mem_info.rss
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass

        # Prime CPU measurements
        self.py_process.cpu_percent()
        if self.pg_process:
            try:
                self.pg_process.cpu_percent()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
        time.sleep(0.1)

    @staticmethod
    def measure(py_pid: int, pg_pid: int, func: Callable):
        """Measure resource usage during function execution."""
        monitor = ResourceMonitor(py_pid, pg_pid)
        result = func()

        # Measure Python process
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

        # Measure PostgreSQL process if available
        pg_delta_mb = 0.0
        pg_peak_mb = 0.0
        pg_cpu = 0.0
        if monitor.pg_process:
            try:
                mem_info = monitor.pg_process.memory_full_info()
                pg_peak = mem_info.uss if hasattr(mem_info, 'uss') else mem_info.rss
            except (psutil.AccessDenied, AttributeError):
                try:
                    mem_info = monitor.pg_process.memory_info()
                    pg_peak = mem_info.rss
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pg_peak = monitor.pg_baseline
            try:
                pg_cpu = monitor.pg_process.cpu_percent()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
            pg_baseline_mb = monitor.pg_baseline / (1024 * 1024)
            pg_peak_mb = pg_peak / (1024 * 1024)
            pg_delta_mb = pg_peak_mb - pg_baseline_mb

        # System-wide metrics
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
# Embedding Client
# =============================================================================

class EmbedAnythingDirectClient:
    """Direct Python client for EmbedAnything."""

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


# --- PostgreSQL Functions ---

def connect_and_get_pid():
    """Connect to PostgreSQL and get backend PID."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("SELECT pg_backend_pid();")
    pid = cur.fetchone()[0]
    cur.close()
    return conn, pid


def setup_pg_database(conn, use_index: bool = True):
    """Initialize PostgreSQL schema, optionally with HNSW index."""
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
    if use_index:
        cur.execute("""
                    CREATE INDEX ON embeddings_test
                        USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 100);
                    """)
    conn.commit()
    cur.close()


def truncate_pg_table(conn):
    """Clear PostgreSQL test table."""
    cur = conn.cursor()
    cur.execute("TRUNCATE embeddings_test;")
    conn.commit()
    cur.close()


def make_inputs(n: int) -> List[str]:
    """Load review texts from TPCx-AI dataset."""
    from data.loader import get_review_texts
    return get_review_texts(n, shuffle=True)


# --- PostgreSQL Benchmarks ---

def benchmark_pg_internal(conn, texts: List[str], provider: str, model: str) -> float:
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


# --- ChromaDB Functions ---

def create_chroma_client(base_path: str = "./chroma_bench", use_index: bool = True, embed_fn: Callable = None):
    """Create a fresh ChromaDB persistent client.
    
    When use_index=False, minimizes HNSW parameters to reduce indexing overhead.
    Note: ChromaDB always uses HNSW internally, but we can minimize its impact.
    """
    db_path = f"{base_path}_{time.time_ns()}"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    client = chromadb.PersistentClient(path=db_path)
    
    if use_index:
        # Standard HNSW configuration for optimal search
        configuration = {
            "hnsw": {
                "space": "cosine",
                "max_neighbors": 16,
                "ef_construction": 100
            }
        }
    else:
        # Minimal HNSW configuration to reduce indexing overhead (safe minimum)
        configuration = {
            "hnsw": {
                "space": "cosine",
                "max_neighbors": 2,
                "ef_construction": 0
            }
        }
    
    # Always register the external embedding function for Chroma collections.
    emb_obj = EmbeddingWrapper(embed_fn)
    collection = client.create_collection("bench", embedding_function=emb_obj, configuration=configuration)
    return client, collection, db_path


def cleanup_chroma(client, db_path: str):
    """Clean up ChromaDB client and files."""
    del client
    time.sleep(0.2)
    if os.path.exists(db_path):
        shutil.rmtree(db_path)


def benchmark_chroma(collection, texts: List[str], embed_fn: Callable) -> float:
    """Benchmark ChromaDB with external embeddings."""
    start = time.perf_counter()

    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i:i + BATCH_SIZE]
        embeddings = embed_fn(chunk)

        collection.add(
            ids=[f"id_{i + j}" for j in range(len(chunk))],
            embeddings=embeddings,
            documents=chunk
        )

    elapsed = time.perf_counter() - start
    return elapsed


# --- Benchmark Runner Functions ---

def run_benchmark_iteration(py_pid: int, pg_pid: int, truncate_fn: Callable,
                            benchmark_fn: Callable) -> BenchmarkResult:
    """Run a single benchmark iteration with resource monitoring."""
    truncate_fn()

    elapsed, stats = ResourceMonitor.measure(py_pid, pg_pid, benchmark_fn)
    return BenchmarkResult(time_s=elapsed, stats=stats)


def run_multiple_iterations(py_pid: int, pg_pid: int, truncate_fn: Callable,
                            benchmark_fn: Callable, runs: int) -> List[BenchmarkResult]:
    """Run multiple benchmark iterations."""
    results = []
    for _ in range(runs):
        result = run_benchmark_iteration(py_pid, pg_pid, truncate_fn, benchmark_fn)
        results.append(result)
    return results


def run_pg_method(texts: List[str], provider: str, runs: int) -> List[BenchmarkResult]:
    """Run PG benchmark with fresh connection, warmup, then timed runs."""
    conn, pg_pid = connect_and_get_pid()  # pg_pid is the backend process handling this connection
    py_pid = os.getpid()

    try:
        # Warm up
        warmup_texts = make_inputs(8)
        truncate_pg_table(conn)
        benchmark_pg_internal(conn, warmup_texts, provider, EMBED_ANYTHING_MODEL)

        # Run benchmark
        results = run_multiple_iterations(
            py_pid,
            pg_pid,
            lambda: truncate_pg_table(conn),
            lambda: benchmark_pg_internal(conn, texts, provider, EMBED_ANYTHING_MODEL),
            runs
        )
        return results
    finally:
        conn.close()


def run_chroma_method(texts: List[str], embed_client: EmbedAnythingDirectClient,
                      runs: int, use_index: bool = True) -> List[BenchmarkResult]:
    """Run ChromaDB benchmark with fresh client, warmup, then timed runs."""
    py_pid = os.getpid()

    # Create fresh client for warmup (always register embed function)
    client, collection, db_path = create_chroma_client(use_index=use_index, embed_fn=embed_client.embed)

    try:
        # Warm up
        warmup_texts = make_inputs(8)
        benchmark_chroma(collection, warmup_texts, embed_client.embed)
    finally:
        cleanup_chroma(client, db_path)

    # Run actual benchmark iterations (no PG process for ChromaDB)
    results = []
    for _ in range(runs):
        client, collection, db_path = create_chroma_client(use_index=use_index, embed_fn=embed_client.embed)
        try:
            elapsed, stats = ResourceMonitor.measure(
                py_pid,
                None,  # No PostgreSQL process for ChromaDB
                lambda: benchmark_chroma(collection, texts, embed_client.embed)
            )
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        finally:
            cleanup_chroma(client, db_path)

    return results


# --- Output Functions ---

def print_header():
    """Print benchmark results header."""
    # Column widths
    lbl_w = 12
    time_w = 12
    col_w = 11

    print("Benchmark Results:", flush=True)
    med_w = 7
    header = (
        "  " +
        f"{'':{lbl_w}}{'':{med_w}} | {'Time (s) μ±σ':>{time_w}} | {'Py Δ MB μ±σ':>{col_w}} | {'Py Peak μ±σ':>{col_w}} | {'Py CPU μ±σ':>{col_w}} | "
        f"{'PG Δ MB μ±σ':>{col_w}} | {'PG Peak μ±σ':>{col_w}} | {'PG CPU μ±σ':>{col_w}} | {'Sys MB μ±σ':>{col_w}} | {'Sys CPU% μ±σ':>{col_w}}"
    )
    print(header, flush=True)
    print("=" * len(header), flush=True)


def print_result(label: str, results: List[BenchmarkResult]):
    """Print aggregated results from multiple runs with ± std dev."""
    times = [r.time_s for r in results]
    py_delta_mems = [r.stats.py_delta_mb for r in results]
    py_peak_mems = [r.stats.py_peak_mb for r in results]
    py_cpus = [r.stats.py_cpu for r in results]
    pg_delta_mems = [r.stats.pg_delta_mb for r in results]
    pg_peak_mems = [r.stats.pg_peak_mb for r in results]
    pg_cpus = [r.stats.pg_cpu for r in results]
    sys_mems = [r.stats.sys_mem_mb for r in results]
    sys_cpus = [r.stats.sys_cpu for r in results]

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

    # Row format: two-space prefix, label (left), med column, then numeric columns
    row_fmt = (
        "  {label:<12}{med:>7} | {time:>12} | {pyd:>11} | {pyp:>11} | {pyc:>11} | {pgd:>11} | {pgp:>11} | {pgc:>11} | {sysm:>11} | {sysc:>11}"
    )
    print(row_fmt.format(
        label=label,
        med='',
        time=fmt_mean(times, 3),
        pyd=fmt_mean(py_delta_mems),
        pyp=fmt_mean(py_peak_mems),
        pyc=fmt_mean(py_cpus),
        pgd=fmt_mean(pg_delta_mems),
        pgp=fmt_mean(pg_peak_mems),
        pgc=fmt_mean(pg_cpus),
        sysm=fmt_mean(sys_mems, 0),
        sysc=fmt_mean(sys_cpus),
    ), flush=True)
    # median ± IQR
    print(row_fmt.format(
        label=label,
        med=' (med)',
        time=fmt_median_iqr(times, 3),
        pyd=fmt_median_iqr(py_delta_mems),
        pyp=fmt_median_iqr(py_peak_mems),
        pyc=fmt_median_iqr(py_cpus),
        pgd=fmt_median_iqr(pg_delta_mems),
        pgp=fmt_median_iqr(pg_peak_mems),
        pgc=fmt_median_iqr(pg_cpus),
        sysm=fmt_median_iqr(sys_mems, 0),
        sysc=fmt_median_iqr(sys_cpus),
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


def run_benchmark(use_index: bool):
    """Run the full benchmark with the specified index configuration."""
    global OUTPUT_DIR
    
    # Set output directory based on index mode
    mode_name = "with_index" if use_index else "without_index"
    OUTPUT_DIR = Path(__file__).parent / "output" / mode_name
    
    print(f"\n{'='*105}")
    print(f"BENCHMARK MODE: {'WITH INDEX' if use_index else 'WITHOUT INDEX'}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*105}\n")
    
    if use_index:
        print("pgvector: HNSW index (m=16, ef_construction=100)")
        print("ChromaDB: HNSW index (max_neighbors=16, ef_construction=100)")
    else:
        print("pgvector: No index")
        print("ChromaDB: Minimal HNSW index (max_neighbors=2, ef_construction=0)")
    print()

    # Initialize PostgreSQL schema
    conn, _ = connect_and_get_pid()
    setup_pg_database(conn, use_index=use_index)
    conn.close()

    # Initialize embedding client (reused across all runs)
    embed_client = EmbedAnythingDirectClient()

    print_header()

    all_results = []

    for size in TEST_SIZES:
        texts = make_inputs(size)
        print(f"Size: {size}", flush=True)

        # Benchmark PG with EmbedAnything local
        pg_local_results = run_pg_method(texts, "embed_anything", RUNS_PER_SIZE)

        # Benchmark PG with gRPC
        pg_grpc_results = run_pg_method(texts, "grpc", RUNS_PER_SIZE)

        # Benchmark ChromaDB with EmbedAnything
        chroma_results = run_chroma_method(texts, embed_client, RUNS_PER_SIZE, use_index=use_index)

        print_result("PG local", pg_local_results)
        print_result("PG gRPC", pg_grpc_results)
        print_result("Chroma", chroma_results)
        print()

        all_results.append({
            'size': size,
            'pg_local': compute_metrics(size, pg_local_results),
            'pg_grpc': compute_metrics(size, pg_grpc_results),
            'chroma': compute_metrics(size, chroma_results),
        })

    # Print summary
    print("=" * 95)

    avg_pg_local = mean([r['pg_local']['throughput'] for r in all_results])
    avg_pg_grpc = mean([r['pg_grpc']['throughput'] for r in all_results])
    avg_chroma = mean([r['chroma']['throughput'] for r in all_results])

    print("\nAverage Throughput Across All Sizes (texts/sec):")
    print(f"  PG local:    {avg_pg_local:.2f}")
    print(f"  PG gRPC:     {avg_pg_grpc:.2f}")
    print(f"  Chroma:      {avg_chroma:.2f}")

    # Save results to CSV and generate plots
    save_results_csv(all_results)
    generate_plots(all_results)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run benchmarks in both index modes."""
    # Check for command-line argument
    if len(sys.argv) > 1:
        if sys.argv[1] == "--with-index":
            run_benchmark(use_index=True)
        elif sys.argv[1] == "--without-index":
            run_benchmark(use_index=False)
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Usage: python benchmark.py [--with-index | --without-index]")
            print("       Without arguments, runs both modes.")
            sys.exit(1)
    else:
        # Run both modes
        run_benchmark(use_index=True)
        run_benchmark(use_index=False)


def save_results_csv(all_results: List[dict]):
    """Save benchmark results to CSV file with mean and std values."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"benchmark_{timestamp}.csv"

    methods = ['pg_local', 'pg_grpc', 'chroma']
    metrics = ['throughput', 'time_s', 'py_cpu', 'py_mem_delta', 'py_mem_peak',
               'pg_cpu', 'pg_mem_delta', 'pg_mem_peak', 'sys_cpu', 'sys_mem']

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
    """Generate comparison plots for throughput, CPU, and memory with error bars."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sizes = [r['size'] for r in all_results]
    methods = ['pg_local', 'pg_grpc', 'chroma']
    labels = ['PG Local', 'PG gRPC', 'ChromaDB']
    colors = ['#2ecc71', '#27ae60', '#e74c3c']
    markers = ['o', 's', '^']

    # Plot 1: Throughput
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        y_vals = [r[method]['throughput_median'] for r in all_results]
        y_errs = [r[method]['throughput_iqr'] for r in all_results]
        plt.errorbar(sizes, y_vals, yerr=y_errs, fmt=f'{marker}-', label=label,
                     linewidth=2, color=color, capsize=3, capthick=1)
    plt.xlabel('Number of Texts')
    plt.ylabel('Throughput (texts/sec)')
    plt.title(f'PG+pgvector vs ChromaDB: Throughput (Median ± IQR, batch size={BATCH_SIZE})')
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
    plt.ylabel('Python CPU Usage (%)')
    plt.title(f'Python Process CPU Usage (Median ± IQR, batch size={BATCH_SIZE})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"cpu_python_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: PostgreSQL Process CPU Usage
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        y_vals = [r[method]['pg_cpu_median'] for r in all_results]
        y_errs = [r[method]['pg_cpu_iqr'] for r in all_results]
        plt.errorbar(sizes, y_vals, yerr=y_errs, fmt=f'{marker}-', label=label,
                     linewidth=2, color=color, capsize=3, capthick=1)
    plt.xlabel('Number of Texts')
    plt.ylabel('PostgreSQL CPU Usage (%)')
    plt.title(f'PostgreSQL Process CPU Usage (Median ± IQR, batch size={BATCH_SIZE})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"cpu_postgres_{timestamp}.png", dpi=150, bbox_inches='tight')
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
    plt.title(f'PG+pgvector vs ChromaDB: System CPU (Median ± IQR, batch size={BATCH_SIZE})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"cpu_system_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 5: Python Process Memory Delta
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        y_vals = [r[method]['py_mem_delta_median'] for r in all_results]
        y_errs = [r[method]['py_mem_delta_iqr'] for r in all_results]
        plt.errorbar(sizes, y_vals, yerr=y_errs, fmt=f'{marker}-', label=label,
                     linewidth=2, color=color, capsize=3, capthick=1)
    plt.xlabel('Number of Texts')
    plt.ylabel('Python Memory Delta (MB)')
    plt.title(f'Python Process Memory Change (Median ± IQR, batch size={BATCH_SIZE})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"memory_python_delta_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 6: PostgreSQL Process Memory Delta
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        y_vals = [r[method]['pg_mem_delta_median'] for r in all_results]
        y_errs = [r[method]['pg_mem_delta_iqr'] for r in all_results]
        plt.errorbar(sizes, y_vals, yerr=y_errs, fmt=f'{marker}-', label=label,
                     linewidth=2, color=color, capsize=3, capthick=1)
    plt.xlabel('Number of Texts')
    plt.ylabel('PostgreSQL Memory Delta (MB)')
    plt.title(f'PostgreSQL Process Memory Change (Median ± IQR, batch size={BATCH_SIZE})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"memory_postgres_delta_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 7: Python Process Peak Memory
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
    plt.savefig(OUTPUT_DIR / f"memory_python_peak_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 8: PostgreSQL Process Peak Memory
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        y_vals = [r[method]['pg_mem_peak_median'] for r in all_results]
        y_errs = [r[method]['pg_mem_peak_iqr'] for r in all_results]
        plt.errorbar(sizes, y_vals, yerr=y_errs, fmt=f'{marker}-', label=label,
                     linewidth=2, color=color, capsize=3, capthick=1)
    plt.xlabel('Number of Texts')
    plt.ylabel('PostgreSQL Peak Memory (MB)')
    plt.title(f'PostgreSQL Process Peak Memory (Median ± IQR, batch size={BATCH_SIZE})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"memory_postgres_peak_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 9: System Memory
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        y_vals = [r[method]['sys_mem_median'] for r in all_results]
        y_errs = [r[method]['sys_mem_iqr'] for r in all_results]
        plt.errorbar(sizes, y_vals, yerr=y_errs, fmt=f'{marker}-', label=label,
                     linewidth=2, color=color, capsize=3, capthick=1)
    plt.xlabel('Number of Texts')
    plt.ylabel('System Memory (MB)')
    plt.title(f'PG+pgvector vs ChromaDB: System Memory (Median ± IQR, batch size={BATCH_SIZE})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"memory_system_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 10: Summary bar chart (2x4 grid) with error bars
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Average Metrics (batch size={BATCH_SIZE})', fontsize=14)
    x_pos = range(len(methods))

    # Throughput
    avgs = [median([r[m]['throughput_median'] for r in all_results]) for m in methods]
    stds = [median([r[m]['throughput_iqr'] for r in all_results]) for m in methods]
    bars = axes[0, 0].bar(x_pos, avgs, yerr=stds, color=colors, capsize=3)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(labels, fontsize=8)
    axes[0, 0].set_ylabel('texts/sec')
    axes[0, 0].set_title('Throughput')
    axes[0, 0].bar_label(bars, fmt='%.1f', fontsize=7)

    # Python CPU
    avgs = [median([r[m]['py_cpu_median'] for r in all_results]) for m in methods]
    stds = [median([r[m]['py_cpu_iqr'] for r in all_results]) for m in methods]
    bars = axes[0, 1].bar(x_pos, avgs, yerr=stds, color=colors, capsize=3)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(labels, fontsize=8)
    axes[0, 1].set_ylabel('CPU %')
    axes[0, 1].set_title('Python CPU')
    axes[0, 1].bar_label(bars, fmt='%.1f', fontsize=7)

    # PostgreSQL CPU
    avgs = [median([r[m]['pg_cpu_median'] for r in all_results]) for m in methods]
    stds = [median([r[m]['pg_cpu_iqr'] for r in all_results]) for m in methods]
    bars = axes[0, 2].bar(x_pos, avgs, yerr=stds, color=colors, capsize=3)
    axes[0, 2].set_xticks(x_pos)
    axes[0, 2].set_xticklabels(labels, fontsize=8)
    axes[0, 2].set_ylabel('CPU %')
    axes[0, 2].set_title('PostgreSQL CPU')
    axes[0, 2].bar_label(bars, fmt='%.1f', fontsize=7)

    # System CPU
    avgs = [median([r[m]['sys_cpu_median'] for r in all_results]) for m in methods]
    stds = [median([r[m]['sys_cpu_iqr'] for r in all_results]) for m in methods]
    bars = axes[0, 3].bar(x_pos, avgs, yerr=stds, color=colors, capsize=3)
    axes[0, 3].set_xticks(x_pos)
    axes[0, 3].set_xticklabels(labels, fontsize=8)
    axes[0, 3].set_ylabel('CPU %')
    axes[0, 3].set_title('System CPU')
    axes[0, 3].bar_label(bars, fmt='%.1f', fontsize=7)

    # Python Memory Delta
    avgs = [median([r[m]['py_mem_delta_median'] for r in all_results]) for m in methods]
    stds = [median([r[m]['py_mem_delta_iqr'] for r in all_results]) for m in methods]
    bars = axes[1, 0].bar(x_pos, avgs, yerr=stds, color=colors, capsize=3)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(labels, fontsize=8)
    axes[1, 0].set_ylabel('MB')
    axes[1, 0].set_title('Python Mem Δ')
    axes[1, 0].bar_label(bars, fmt='%.1f', fontsize=7)

    # PostgreSQL Memory Delta
    avgs = [median([r[m]['pg_mem_delta_median'] for r in all_results]) for m in methods]
    stds = [median([r[m]['pg_mem_delta_iqr'] for r in all_results]) for m in methods]
    bars = axes[1, 1].bar(x_pos, avgs, yerr=stds, color=colors, capsize=3)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(labels, fontsize=8)
    axes[1, 1].set_ylabel('MB')
    axes[1, 1].set_title('PostgreSQL Mem Δ')
    axes[1, 1].bar_label(bars, fmt='%.1f', fontsize=7)

    # Python Peak Memory
    avgs = [median([r[m]['py_mem_peak_median'] for r in all_results]) for m in methods]
    stds = [median([r[m]['py_mem_peak_iqr'] for r in all_results]) for m in methods]
    bars = axes[1, 2].bar(x_pos, avgs, yerr=stds, color=colors, capsize=3)
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(labels, fontsize=8)
    axes[1, 2].set_ylabel('MB')
    axes[1, 2].set_title('Python Peak Mem')
    axes[1, 2].bar_label(bars, fmt='%.1f', fontsize=7)

    # System Memory
    avgs = [median([r[m]['sys_mem_median'] for r in all_results]) for m in methods]
    stds = [median([r[m]['sys_mem_iqr'] for r in all_results]) for m in methods]
    bars = axes[1, 3].bar(x_pos, avgs, yerr=stds, color=colors, capsize=3)
    axes[1, 3].set_xticks(x_pos)
    axes[1, 3].set_xticklabels(labels, fontsize=8)
    axes[1, 3].set_ylabel('MB')
    axes[1, 3].set_title('System Memory')
    axes[1, 3].bar_label(bars, fmt='%.0f', fontsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"summary_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
