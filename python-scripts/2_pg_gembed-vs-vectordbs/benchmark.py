import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev, median, quantiles
from typing import Callable, List

import chromadb
import docker
import embed_anything
import psutil
import psycopg2
from data.loader import get_review_texts
from embed_anything import EmbeddingModel, WhichModel
from plot_utils import save_results_csv, generate_plots
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct

# Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'joeldiaz',
    'user': 'joeldiaz',
}

QDRANT_URL = "http://localhost:6333"
QDRANT_CONTAINER_NAME = "qdrant"

TEST_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048]
EMBED_ANYTHING_MODEL = "Qdrant/all-MiniLM-L6-v2-onnx"
RUNS_PER_SIZE = 5

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
            return self.pg_process.memory_full_info().uss
        except (psutil.AccessDenied, AttributeError):
            try:
                return self.pg_process.memory_info().rss
            except:
                return 0
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
        start_time = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start_time

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
        return elapsed, stats


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
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("SELECT pg_backend_pid();")
    pid = cur.fetchone()[0]
    cur.close()
    return conn, pid


def setup_pg_schema(conn):
    """Initialize PostgreSQL schema."""
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


def populate_pg_database(conn, texts: List[str], provider: str):
    """Insert data using internal generation."""
    cur = conn.cursor()
    sql = """
          WITH input_data AS (SELECT %s::text[] AS texts)
          INSERT
          INTO embeddings_test (text, embedding)
          SELECT t, e
          FROM input_data,
               unnest(texts, embed_texts(%s, %s, texts)) AS x(t, e)
          """
    cur.execute(sql, (texts, provider, EMBED_ANYTHING_MODEL))
    conn.commit()
    cur.close()


def setup_pg_indexed(conn, texts: List[str], provider: str):
    """Index exists BEFORE embedding generation."""
    setup_pg_schema(conn)
    cur = conn.cursor()
    cur.execute(
        "CREATE INDEX ON embeddings_test USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=100);")
    conn.commit()
    cur.close()
    populate_pg_database(conn, texts, provider)


def setup_pg_deferred(conn, texts: List[str], provider: str):
    """Index created AFTER embedding generation."""
    setup_pg_schema(conn)
    populate_pg_database(conn, texts, provider)
    cur = conn.cursor()
    cur.execute(
        "CREATE INDEX ON embeddings_test USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=100);")
    conn.commit()
    cur.close()


# =============================================================================
# ChromaDB Functions
# =============================================================================

def create_chroma_client(base_path: str = "./chroma_bench", embed_fn: Callable = None):
    """Create a fresh ChromaDB persistent client."""
    db_path = f"{base_path}_{time.time_ns()}"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    client = chromadb.PersistentClient(path=db_path)

    configuration = {
        "hnsw": {
            "space": "cosine",
            "max_neighbors": 16,
            "ef_construction": 100
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


def benchmark_chroma(collection, texts: List[str], embed_fn: Callable):
    """Benchmark ChromaDB with external embeddings."""
    embeddings = embed_fn(texts)
    collection.add(
        ids=[f"id_{j}" for j in range(len(texts))],
        embeddings=embeddings,
        documents=texts
    )


# =============================================================================
# Qdrant Functions
# =============================================================================

def setup_qdrant(client, texts: List[str], embed_fn: Callable, deferred: bool):
    """Initialize Qdrant and ingest data."""
    if client.collection_exists("bench"):
        client.delete_collection("bench")

    hnsw_config = models.HnswConfigDiff(m=16, ef_construct=100)
    client.create_collection(
        collection_name="bench",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE, hnsw_config=hnsw_config)
    )

    if deferred:
        client.update_collection("bench", optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0))

    embeddings = embed_fn(texts)
    points = [PointStruct(id=j, vector=embeddings[j], payload={"text": texts[j]}) for j in range(len(texts))]

    client.upsert(collection_name="bench", points=points, wait=True)

    if deferred:
        client.update_collection("bench", optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000))


def cleanup_qdrant(client):
    """Clean up Qdrant client."""
    if client.collection_exists("bench"):
        client.delete_collection(collection_name="bench")
    client.close()
    time.sleep(0.1)


# =============================================================================
# Benchmark Runner Functions
# =============================================================================

def run_pg_method(texts: List[str], provider: str, strategy: str, runs: int) -> List[BenchmarkResult]:
    """Run PG benchmark."""
    py_pid = os.getpid()
    results = []
    for _ in range(runs):
        conn, pg_pid = connect_and_get_pid()
        fn = setup_pg_indexed if strategy == "indexed" else setup_pg_deferred
        elapsed, stats = ResourceMonitor.measure(py_pid, pg_pid, lambda: fn(conn, texts, provider))
        results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        conn.close()
    return results


def run_chroma_method(texts: List[str], embed_client: EmbedAnythingDirectClient, runs: int) -> List[BenchmarkResult]:
    """Run ChromaDB benchmark."""
    py_pid = os.getpid()
    results = []
    for _ in range(runs):
        client, collection, db_path = create_chroma_client(embed_fn=embed_client.embed)
        try:
            elapsed, stats = ResourceMonitor.measure(py_pid, None,
                                                     lambda: benchmark_chroma(collection, texts, embed_client.embed))
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        finally:
            cleanup_chroma(client, db_path)
    return results


def run_qdrant_method(texts: List[str], embed_client: EmbedAnythingDirectClient, runs: int, deferred: bool) -> List[
    BenchmarkResult]:
    """Run Qdrant benchmark."""
    py_pid = os.getpid()
    results = []
    for _ in range(runs):
        client = QdrantClient(url=QDRANT_URL)
        try:
            elapsed, stats = ResourceMonitor.measure(py_pid, None,
                                                     lambda: setup_qdrant(client, texts, embed_client.embed, deferred))
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        finally:
            cleanup_qdrant(client)
    return results


# =============================================================================
# Output Functions
# =============================================================================

def print_header():
    """Print detailed benchmark results header."""
    lbl_w, time_w, col_w, med_w = 12, 14, 13, 7
    print("\nBenchmark Results:", flush=True)
    header = (
            "  " +
            f"{'':{lbl_w}}{'':{med_w}} | {'Time (s)':>{time_w}} | "
            f"{'Py Δ MB':>{col_w}} | {'Py Peak MB':>{col_w}} | {'Py CPU%':>{col_w}} | "
            f"{'PG Δ MB':>{col_w}} | {'PG Peak MB':>{col_w}} | {'PG CPU%':>{col_w}} | "
            f"{'QD Δ MB':>{col_w}} | {'QD Peak MB':>{col_w}} | {'QD CPU%':>{col_w}} | "
            f"{'Sys MB':>{col_w}} | {'Sys CPU%':>{col_w}}"
    )
    print(header, flush=True)
    print("=" * len(header), flush=True)


def print_result(label: str, results: List[BenchmarkResult]):
    times = [r.time_s for r in results]
    py_deltas = [r.stats.py_delta_mb for r in results];
    py_peaks = [r.stats.py_peak_mb for r in results];
    py_cpus = [r.stats.py_cpu for r in results]
    pg_deltas = [r.stats.pg_delta_mb for r in results];
    pg_peaks = [r.stats.pg_peak_mb for r in results];
    pg_cpus = [r.stats.pg_cpu for r in results]
    qd_deltas = [r.stats.qd_delta_mb for r in results];
    qd_peaks = [r.stats.qd_peak_mb for r in results];
    qd_cpus = [r.stats.qd_cpu for r in results]
    sys_mems = [r.stats.sys_mem_mb for r in results];
    sys_cpus = [r.stats.sys_cpu for r in results]

    def fmt(vals, p=1): return f"{mean(vals):.{p}f}±{safe_stdev(vals):.{p}f}"

    def fmt_med(vals, p=1): return f"{median(vals):.{p}f}±{calc_iqr(vals):.{p}f}"

    row_fmt = "  {label:<12}{med:>7} | {time:>14} | {pyd:>13} | {pyp:>13} | {pyc:>13} | {pgd:>13} | {pgp:>13} | {pgc:>13} | {qdd:>13} | {qdp:>13} | {qdc:>13} | {sysm:>13} | {sysc:>13}"
    print(
        row_fmt.format(label=label, med='', time=fmt(times, 3), pyd=fmt(py_deltas), pyp=fmt(py_peaks), pyc=fmt(py_cpus),
                       pgd=fmt(pg_deltas), pgp=fmt(pg_peaks), pgc=fmt(pg_cpus), qdd=fmt(qd_deltas), qdp=fmt(qd_peaks),
                       qdc=fmt(qd_cpus), sysm=fmt(sys_mems, 0), sysc=fmt(sys_cpus)), flush=True)
    print(
        row_fmt.format(label=label, med=' (med)', time=fmt_med(times, 3), pyd=fmt_med(py_deltas), pyp=fmt_med(py_peaks),
                       pyc=fmt_med(py_cpus), pgd=fmt_med(pg_deltas), pgp=fmt_med(pg_peaks), pgc=fmt_med(pg_cpus),
                       qdd=fmt_med(qd_deltas), qdp=fmt_med(qd_peaks), qdc=fmt_med(qd_cpus), sysm=fmt_med(sys_mems, 0),
                       sysc=fmt_med(sys_cpus)), flush=True)


def compute_metrics(size: int, results: List[BenchmarkResult]) -> dict:
    """Compute mean/std and median/IQR for all metrics from benchmark results."""
    times = [r.time_s for r in results]
    py_cpu = [r.stats.py_cpu for r in results];
    py_delta = [r.stats.py_delta_mb for r in results];
    py_peak = [r.stats.py_peak_mb for r in results]
    pg_cpu = [r.stats.pg_cpu for r in results];
    pg_delta = [r.stats.pg_delta_mb for r in results];
    pg_peak = [r.stats.pg_peak_mb for r in results]
    qd_cpu = [r.stats.qd_cpu for r in results];
    qd_delta = [r.stats.qd_delta_mb for r in results];
    qd_peak = [r.stats.qd_peak_mb for r in results]
    sys_cpu = [r.stats.sys_cpu for r in results];
    sys_mem = [r.stats.sys_mem_mb for r in results]

    return {
        'throughput': size / mean(times),
        'throughput_std': size / mean(times) * safe_stdev(times) / mean(times) if len(times) > 1 else 0,
        'throughput_median': size / median(times),
        'throughput_iqr': size / median(times) * calc_iqr(times) / median(times) if len(times) >= 4 else 0,
        'time_s': mean(times), 'time_s_std': safe_stdev(times), 'time_s_median': median(times),
        'time_s_iqr': calc_iqr(times),
        'py_cpu': mean(py_cpu), 'py_cpu_std': safe_stdev(py_cpu), 'py_cpu_median': median(py_cpu),
        'py_cpu_iqr': calc_iqr(py_cpu),
        'py_mem_delta': mean(py_delta), 'py_mem_peak': mean(py_peak), 'pg_cpu': mean(pg_cpu),
        'pg_mem_delta': mean(pg_delta), 'pg_mem_peak': mean(pg_peak),
        'qd_cpu': mean(qd_cpu), 'qd_mem_delta': mean(qd_delta), 'qd_mem_peak': mean(qd_peak), 'sys_cpu': mean(sys_cpu),
        'sys_mem': mean(sys_mem),
    }


def main():
    print(f"\n{'=' * 105}\nBENCHMARK 2: PG GEMBED VS VECTOR DBS (INDEXED VS DEFERRED)\n{'=' * 105}\n")
    embed_client = EmbedAnythingDirectClient()
    print_header()
    all_results = []

    for size in TEST_SIZES:
        texts = get_review_texts(size, shuffle=False)
        print(f"Size: {size}", flush=True)

        res_pg_local_idx = run_pg_method(texts, "embed_anything", "indexed", RUNS_PER_SIZE)
        res_pg_local_def = run_pg_method(texts, "embed_anything", "deferred", RUNS_PER_SIZE)
        res_pg_grpc_idx = run_pg_method(texts, "grpc", "indexed", RUNS_PER_SIZE)
        res_pg_grpc_def = run_pg_method(texts, "grpc", "deferred", RUNS_PER_SIZE)
        res_qd_idx = run_qdrant_method(texts, embed_client, RUNS_PER_SIZE, deferred=False)
        res_qd_def = run_qdrant_method(texts, embed_client, RUNS_PER_SIZE, deferred=True)
        res_ch = run_chroma_method(texts, embed_client, RUNS_PER_SIZE)

        print_result("PG Local Indexed", res_pg_local_idx)
        print_result("PG Local Deferred", res_pg_local_def)
        print_result("PG gRPC Indexed", res_pg_grpc_idx)
        print_result("PG gRPC Deferred", res_pg_grpc_def)
        print_result("QD Indexed", res_qd_idx)
        print_result("QD Deferred", res_qd_def)
        print_result("Chroma", res_ch)
        print()

        all_results.append({
            'size': size,
            'pg_local_indexed': compute_metrics(size, res_pg_local_idx),
            'pg_local_deferred': compute_metrics(size, res_pg_local_def),
            'pg_grpc_indexed': compute_metrics(size, res_pg_grpc_idx),
            'pg_grpc_deferred': compute_metrics(size, res_pg_grpc_def),
            'qd_indexed': compute_metrics(size, res_qd_idx),
            'qd_deferred': compute_metrics(size, res_qd_def),
            'chroma': compute_metrics(size, res_ch),
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    methods = ['pg_local_indexed', 'pg_local_deferred', 'pg_grpc_indexed', 'pg_grpc_deferred', 'qd_indexed',
               'qd_deferred', 'chroma']
    save_results_csv(all_results, OUTPUT_DIR, timestamp, methods)
    generate_plots(all_results, OUTPUT_DIR, timestamp, methods)


if __name__ == "__main__":
    main()
