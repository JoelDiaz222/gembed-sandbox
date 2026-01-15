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
import docker
import embed_anything
import matplotlib.pyplot as plt
import psutil
import psycopg2
from embed_anything import EmbeddingModel, WhichModel
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct

# Configuration
POSTGRESQL_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'joeldiaz',
    'user': 'joeldiaz',
}

QDRANT_URL = "http://localhost:6333"
QDRANT_CONTAINER_NAME = "qdrant"

TEST_SIZES = [16, 32, 64, 128, 256, 512]
BATCH_SIZE = 32
EMBED_ANYTHING_MODEL = "Qdrant/all-MiniLM-L6-v2-onnx"
RUNS_PER_SIZE = 5  # Number of runs per test size

OUTPUT_DIR = Path(__file__).parent / "output"

model_cache = {}


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
    qd_delta_mb: float
    qd_peak_mb: float
    qd_cpu: float
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
    return q[2] - q[0]


# =============================================================================
# Resource Monitoring
# =============================================================================

class ResourceMonitor:
    def __init__(self, py_pid: int, pg_pid: int = None, qd_name: str = QDRANT_CONTAINER_NAME):
        self.py_process = psutil.Process(py_pid)
        self.pg_process = psutil.Process(pg_pid) if pg_pid else None

        # Docker initialization
        self.docker_client = docker.from_env()
        try:
            self.container = self.docker_client.containers.get(qd_name)
        except Exception:
            self.container = None

        # Baselines
        self.py_baseline = self._get_py_mem()
        self.pg_baseline = self._get_pg_mem()
        self.qd_baseline_stats = self._get_qd_stats() if self.container else None

        # Warm up CPU counters
        self.py_process.cpu_percent()
        if self.pg_process: self.pg_process.cpu_percent()
        time.sleep(0.1)

    def _get_py_mem(self):
        m = self.py_process.memory_full_info()
        return m.uss if hasattr(m, 'uss') else m.rss

    def _get_pg_mem(self):
        if not self.pg_process: return 0
        try:
            m = self.pg_process.memory_full_info()
            return m.uss if hasattr(m, 'uss') else m.rss
        except:
            return 0

    def _get_qd_stats(self):
        """Returns a snapshot of Docker stats."""
        return self.container.stats(stream=False)

    def _calculate_qd_cpu(self, start_stats, end_stats):
        """Calculates CPU percentage similar to 'docker stats'."""
        cpu_delta = end_stats['cpu_stats']['cpu_usage']['total_usage'] - \
                    start_stats['cpu_stats']['cpu_usage']['total_usage']
        system_delta = end_stats['cpu_stats']['system_cpu_usage'] - \
                       start_stats['cpu_stats']['system_cpu_usage']

        if system_delta > 0.0 and cpu_delta > 0.0:
            # We multiply by number of cores to get a 0-100% per-core scaled value
            cpus = end_stats['cpu_stats'].get('online_cpus', psutil.cpu_count())
            return (cpu_delta / system_delta) * cpus * 100.0
        return 0.0

    @staticmethod
    def measure(py_pid: int, pg_pid: int, func: Callable):
        monitor = ResourceMonitor(py_pid, pg_pid)
        result = func()

        # Gather final stats
        py_peak = monitor._get_py_mem()
        py_cpu = monitor.py_process.cpu_percent()

        pg_peak = monitor._get_pg_mem()
        pg_cpu = monitor.pg_process.cpu_percent() if monitor.pg_process else 0.0

        qd_delta_mb = 0.0
        qd_peak_mb = 0.0
        qd_cpu = 0.0

        if monitor.container:
            end_qd_stats = monitor._get_qd_stats()
            qd_peak_raw = end_qd_stats['memory_stats']['usage']
            qd_peak_mb = qd_peak_raw / (1024 * 1024)

            baseline_raw = monitor.qd_baseline_stats['memory_stats']['usage']
            qd_delta_mb = qd_peak_mb - (baseline_raw / (1024 * 1024))
            qd_cpu = monitor._calculate_qd_cpu(monitor.qd_baseline_stats, end_qd_stats)

        sys_v = psutil.virtual_memory()

        stats = ResourceStats(
            py_delta_mb=(py_peak - monitor.py_baseline) / 1e6,
            py_peak_mb=py_peak / 1e6,
            py_cpu=py_cpu,
            pg_delta_mb=(pg_peak - monitor.pg_baseline) / 1e6,
            pg_peak_mb=pg_peak / 1e6,
            pg_cpu=pg_cpu,
            qd_delta_mb=qd_delta_mb,
            qd_peak_mb=qd_peak_mb,
            qd_cpu=qd_cpu,
            sys_mem_mb=sys_v.used / 1e6,
            sys_cpu=psutil.cpu_percent()
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


# =============================================================================
# PostgreSQL Functions
# =============================================================================

def connect_and_get_pid():
    """Connect to PostgreSQL and get backend PID."""
    conn = psycopg2.connect(**POSTGRESQL_CONFIG)
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
                   unnest(embed_texts(%s, %s, %s::text[])) e
              """
        cur.execute(sql, (chunk, provider, model, chunk))

    conn.commit()
    elapsed = time.perf_counter() - start
    cur.close()
    return elapsed


# =============================================================================
# ChromaDB Functions
# =============================================================================

def create_chroma_client(base_path: str = "./chroma_bench", use_index: bool = True, embed_fn: Callable = None):
    """Create a fresh ChromaDB persistent client."""
    db_path = f"{base_path}_{time.time_ns()}"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    client = chromadb.PersistentClient(path=db_path)

    if use_index:
        configuration = {
            "hnsw": {
                "space": "cosine",
                "max_neighbors": 16,
                "ef_construction": 100
            }
        }
    else:
        configuration = {
            "hnsw": {
                "space": "cosine",
                "max_neighbors": 2,
                "ef_construction": 0
            }
        }

    emb_obj = EmbeddingWrapper(embed_fn)
    collection = client.create_collection("bench", embedding_function=emb_obj, configuration=configuration)
    return client, collection, db_path


def cleanup_chroma(client, db_path: str):
    """Clean up ChromaDB client and files."""
    del client
    time.sleep(0.1)
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


# =============================================================================
# Qdrant Functions
# =============================================================================

def create_qdrant_client(use_index: bool = True):
    """Create a fresh Qdrant client."""
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists("bench"):
        client.delete_collection("bench")

    # Configure HNSW index parameters
    if use_index:
        hnsw_config = models.HnswConfigDiff(
            m=16,
            ef_construct=100,
        )
    else:
        # Deactivate HNSW index
        hnsw_config = models.HnswConfigDiff(
            m=0,
        )

    client.create_collection(
        collection_name="bench",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE, hnsw_config=hnsw_config)
    )

    return client


def cleanup_qdrant(client):
    """Clean up Qdrant client and files."""
    if client.collection_exists("bench"):
        client.delete_collection(collection_name="bench")

    client.close()
    time.sleep(0.1)


def benchmark_qdrant(client, texts: List[str], embed_fn: Callable) -> float:
    """Benchmark Qdrant with external embeddings."""
    start = time.perf_counter()

    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i:i + BATCH_SIZE]
        embeddings = embed_fn(chunk)

        points = [
            PointStruct(
                id=i + j,
                vector=embeddings[j],
                payload={"text": chunk[j]}
            )
            for j in range(len(chunk))
        ]

        client.upsert(
            collection_name="bench",
            points=points
        )

    elapsed = time.perf_counter() - start
    return elapsed


# =============================================================================
# Benchmark Runner Functions
# =============================================================================

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
    conn, pg_pid = connect_and_get_pid()
    py_pid = os.getpid()

    try:
        warmup_texts = make_inputs(8)
        truncate_pg_table(conn)
        benchmark_pg_internal(conn, warmup_texts, provider, EMBED_ANYTHING_MODEL)

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

    client, collection, db_path = create_chroma_client(use_index=use_index, embed_fn=embed_client.embed)
    try:
        warmup_texts = make_inputs(8)
        benchmark_chroma(collection, warmup_texts, embed_client.embed)
    finally:
        cleanup_chroma(client, db_path)

    results = []
    for _ in range(runs):
        client, collection, db_path = create_chroma_client(use_index=use_index, embed_fn=embed_client.embed)
        try:
            elapsed, stats = ResourceMonitor.measure(
                py_pid,
                None,
                lambda: benchmark_chroma(collection, texts, embed_client.embed)
            )
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        finally:
            cleanup_chroma(client, db_path)

    return results


def run_qdrant_method(texts: List[str], embed_client: EmbedAnythingDirectClient,
                      runs: int, use_index: bool = True) -> List[BenchmarkResult]:
    """Run Qdrant benchmark with fresh client, warmup, then timed runs."""
    py_pid = os.getpid()

    client = create_qdrant_client(use_index=use_index)
    try:
        warmup_texts = make_inputs(8)
        benchmark_qdrant(client, warmup_texts, embed_client.embed)
    finally:
        cleanup_qdrant(client)

    results = []
    for _ in range(runs):
        client = create_qdrant_client(use_index=use_index)
        try:
            elapsed, stats = ResourceMonitor.measure(
                py_pid,
                None,
                lambda: benchmark_qdrant(client, texts, embed_client.embed)
            )
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        finally:
            cleanup_qdrant(client)

    return results


# =============================================================================
# Output Functions
# =============================================================================

def print_header():
    """Print benchmark results header with QD (Qdrant) columns."""
    lbl_w, time_w, col_w, med_w = 12, 14, 13, 7

    print("\nBenchmark Results (Peak RAM in MB):", flush=True)
    header = (
        f"{'':{lbl_w}}{'':{med_w}} | {'Time (s)':>{time_w}} | "
        f"{'Py Δ MB':>{col_w}} | {'Py Peak':>{col_w}} | {'Py CPU%':>{col_w}} | "
        f"{'PG Δ MB':>{col_w}} | {'PG Peak':>{col_w}} | {'PG CPU%':>{col_w}} | "
        f"{'QD Δ MB':>{col_w}} | {'QD Peak':>{col_w}} | {'QD CPU%':>{col_w}} | "
        f"{'Sys MB':>{col_w}} | {'Sys CPU%':>{col_w}}"
    )
    print(header, flush=True)
    print("=" * len(header), flush=True)


def print_result(label: str, results: List[BenchmarkResult]):
    times = [r.time_s for r in results]

    py_deltas = [r.stats.py_delta_mb for r in results]
    py_peaks = [r.stats.py_peak_mb for r in results]
    py_cpus = [r.stats.py_cpu for r in results]

    pg_deltas = [r.stats.pg_delta_mb for r in results]
    pg_peaks = [r.stats.pg_peak_mb for r in results]
    pg_cpus = [r.stats.pg_cpu for r in results]

    qd_deltas = [r.stats.qd_delta_mb for r in results]
    qd_peaks = [r.stats.qd_peak_mb for r in results]
    qd_cpus = [r.stats.qd_cpu for r in results]

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
        "{qdd:>13} | {qdp:>13} | {qdc:>13} | "
        "{sysm:>13} | {sysc:>13}"
    )

    print(row_fmt.format(
        label=label, med='',
        time=fmt(times, 3),
        pyd=fmt(py_deltas), pyp=fmt(py_peaks), pyc=fmt(py_cpus),
        pgd=fmt(pg_deltas), pgp=fmt(pg_peaks), pgc=fmt(pg_cpus),
        qdd=fmt(qd_deltas), qdp=fmt(qd_peaks), qdc=fmt(qd_cpus),
        sysm=fmt(sys_mems, 0), sysc=fmt(sys_cpus)
    ), flush=True)

    # Median row
    print(row_fmt.format(
        label=label, med=' (med)',
        time=fmt_med(times, 3),
        pyd=fmt_med(py_deltas), pyp=fmt_med(py_peaks), pyc=fmt_med(py_cpus),
        pgd=fmt_med(pg_deltas), pgp=fmt_med(pg_peaks), pgc=fmt_med(pg_cpus),
        qdd=fmt_med(qd_deltas), qdp=fmt_med(qd_peaks), qdc=fmt_med(qd_cpus),
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
    qd_cpu = [r.stats.qd_cpu for r in results]
    qd_delta = [r.stats.qd_delta_mb for r in results]
    qd_peak = [r.stats.qd_peak_mb for r in results]
    sys_cpu = [r.stats.sys_cpu for r in results]
    sys_mem = [r.stats.sys_mem_mb for r in results]

    return {
        'throughput': size / mean(times),
        'throughput_std': size / mean(times) * safe_stdev(times) / mean(times) if len(times) > 1 else 0,
        'throughput_median': size / median(times),
        'throughput_iqr': size / median(times) * calc_iqr(times) / median(times) if len(times) >= 4 else 0,
        'time_s': mean(times),
        'time_s_std': safe_stdev(times),
        'time_s_median': median(times),
        'time_s_iqr': calc_iqr(times),
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
        'qd_cpu': mean(qd_cpu),
        'qd_cpu_std': safe_stdev(qd_cpu),
        'qd_cpu_median': median(qd_cpu),
        'qd_cpu_iqr': calc_iqr(qd_cpu),
        'qd_mem_delta': mean(qd_delta),
        'qd_mem_delta_std': safe_stdev(qd_delta),
        'qd_mem_delta_median': median(qd_delta),
        'qd_mem_delta_iqr': calc_iqr(qd_delta),
        'qd_mem_peak': mean(qd_peak),
        'qd_mem_peak_std': safe_stdev(qd_peak),
        'qd_mem_peak_median': median(qd_peak),
        'qd_mem_peak_iqr': calc_iqr(qd_peak),
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

    mode_name = "with_index" if use_index else "without_index"
    OUTPUT_DIR = Path(__file__).parent / "output" / mode_name

    print(f"\n{'=' * 105}")
    print(f"BENCHMARK MODE: {'WITH INDEX' if use_index else 'WITHOUT INDEX'}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'=' * 105}\n")

    if use_index:
        print("pgvector: HNSW index (m=16, ef_construction=100)")
        print("ChromaDB: HNSW index (max_neighbors=16, ef_construction=100)")
        print("Qdrant: HNSW index (m=16, ef_construct=100)")
    else:
        print("pgvector: No index")
        print("ChromaDB: Minimal HNSW index (max_neighbors=2, ef_construction=0)")
        print("Qdrant: HNSW index disabled (m=0)")
    print()

    conn, _ = connect_and_get_pid()
    setup_pg_database(conn, use_index=use_index)
    conn.close()

    embed_client = EmbedAnythingDirectClient()

    print_header()

    all_results = []

    for size in TEST_SIZES:
        texts = make_inputs(size)
        print(f"Size: {size}", flush=True)

        pg_local_results = run_pg_method(texts, "embed_anything", RUNS_PER_SIZE)
        pg_grpc_results = run_pg_method(texts, "grpc", RUNS_PER_SIZE)
        chroma_results = run_chroma_method(texts, embed_client, RUNS_PER_SIZE, use_index=use_index)
        qdrant_results = run_qdrant_method(texts, embed_client, RUNS_PER_SIZE, use_index=use_index)

        print_result("PG local", pg_local_results)
        print_result("PG gRPC", pg_grpc_results)
        print_result("Chroma", chroma_results)
        print_result("Qdrant", qdrant_results)
        print()

        all_results.append({
            'size': size,
            'pg_local': compute_metrics(size, pg_local_results),
            'pg_grpc': compute_metrics(size, pg_grpc_results),
            'chroma': compute_metrics(size, chroma_results),
            'qdrant': compute_metrics(size, qdrant_results),
        })

    print("=" * 95)

    avg_pg_local = mean([r['pg_local']['throughput'] for r in all_results])
    avg_pg_grpc = mean([r['pg_grpc']['throughput'] for r in all_results])
    avg_chroma = mean([r['chroma']['throughput'] for r in all_results])
    avg_qdrant = mean([r['qdrant']['throughput'] for r in all_results])

    print("\nAverage Throughput Across All Sizes (texts/sec):")
    print(f"  PG local:    {avg_pg_local:.2f}")
    print(f"  PG gRPC:     {avg_pg_grpc:.2f}")
    print(f"  Chroma:      {avg_chroma:.2f}")
    print(f"  Qdrant:      {avg_qdrant:.2f}")

    save_results_csv(all_results)
    generate_plots(all_results)


def save_results_csv(all_results: List[dict]):
    """Save benchmark results to CSV file with mean and std values."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"benchmark_{timestamp}.csv"

    methods = ['pg_local', 'pg_grpc', 'chroma', 'qdrant']
    metrics = ['throughput', 'time_s', 'py_cpu', 'py_mem_delta', 'py_mem_peak',
               'pg_cpu', 'pg_mem_delta', 'pg_mem_peak',
               'qd_cpu', 'qd_mem_delta', 'qd_mem_peak',
               'sys_cpu', 'sys_mem']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
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
    methods = ['pg_local', 'pg_grpc', 'chroma', 'qdrant']
    labels = ['PG Local', 'PG gRPC', 'ChromaDB', 'Qdrant']
    colors = ['#2ecc71', '#27ae60', '#e74c3c', '#3498db']
    markers = ['o', 's', '^', 'd']

    # Plot 1: Throughput
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        y_vals = [r[method]['throughput_median'] for r in all_results]
        y_errs = [r[method]['throughput_iqr'] for r in all_results]
        plt.errorbar(sizes, y_vals, yerr=y_errs, fmt=f'{marker}-', label=label,
                     linewidth=2, color=color, capsize=3, capthick=1)
    plt.xlabel('Number of Texts')
    plt.ylabel('Throughput (texts/sec)')
    plt.title(f'Vector Database Comparison: Throughput (Median ± IQR, batch size={BATCH_SIZE})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"throughput_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Python Memory Peak
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        y_vals = [r[method]['py_mem_peak_median'] for r in all_results]
        y_errs = [r[method]['py_mem_peak_iqr'] for r in all_results]
        plt.errorbar(sizes, y_vals, yerr=y_errs, fmt=f'{marker}-', label=label,
                     linewidth=2, color=color, capsize=3, capthick=1)
    plt.xlabel('Number of Texts')
    plt.ylabel('Peak Python Memory (MB)')
    plt.title('Python Client Memory Usage (Median ± IQR)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"py_memory_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Postgres Memory Peak
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        y_vals = [r[method]['pg_mem_peak_median'] for r in all_results]
        y_errs = [r[method]['pg_mem_peak_iqr'] for r in all_results]
        plt.errorbar(sizes, y_vals, yerr=y_errs, fmt=f'{marker}-', label=label,
                     linewidth=2, color=color, capsize=3, capthick=1)
    plt.xlabel('Number of Texts')
    plt.ylabel('Peak PostgreSQL Memory (MB)')
    plt.title('PostgreSQL Backend Memory Usage (Median ± IQR)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"pg_memory_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 4: Qdrant Memory Peak (Only valid for Qdrant method)
    plt.figure(figsize=(10, 6))
    # We only plot Qdrant series for Qdrant metrics, or maybe we want to see if others trigger it (they shouldn't)
    # But for comparison, we only have Qdrant container stats when running Qdrant.
    # Actually, the logic in ResourceMonitor only captures qd stats if container exists.
    # So for other methods, it might be 0 or baseline.
    # Let's just plot it for all methods to be safe and see if there's noise.
    for method, label, color, marker in zip(methods, labels, colors, markers):
        y_vals = [r[method]['qd_mem_peak_median'] for r in all_results]
        y_errs = [r[method]['qd_mem_peak_iqr'] for r in all_results]
        plt.errorbar(sizes, y_vals, yerr=y_errs, fmt=f'{marker}-', label=label,
                     linewidth=2, color=color, capsize=3, capthick=1)
    plt.xlabel('Number of Texts')
    plt.ylabel('Peak Qdrant Memory (MB)')
    plt.title('Qdrant Container Memory Usage (Median ± IQR)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"qd_memory_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 5: System Memory
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        y_vals = [r[method]['sys_mem_median'] for r in all_results]
        y_errs = [r[method]['sys_mem_iqr'] for r in all_results]
        plt.errorbar(sizes, y_vals, yerr=y_errs, fmt=f'{marker}-', label=label,
                     linewidth=2, color=color, capsize=3, capthick=1)
    plt.xlabel('Number of Texts')
    plt.ylabel('System Memory (MB)')
    plt.title('Total System Memory Usage (Median ± IQR)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"memory_system_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 6: System CPU
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        y_vals = [r[method]['sys_cpu_median'] for r in all_results]
        y_errs = [r[method]['sys_cpu_iqr'] for r in all_results]
        plt.errorbar(sizes, y_vals, yerr=y_errs, fmt=f'{marker}-', label=label,
                     linewidth=2, color=color, capsize=3, capthick=1)
    plt.xlabel('Number of Texts')
    plt.ylabel('System CPU (%)')
    plt.title('Total System CPU Usage (Median ± IQR)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"cpu_system_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to: {OUTPUT_DIR}/")


def main():
    """Run benchmarks in both index modes."""
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
        run_benchmark(use_index=True)
        run_benchmark(use_index=False)


if __name__ == "__main__":
    main()
