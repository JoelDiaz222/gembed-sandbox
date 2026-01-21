import csv
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev, quantiles, median
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

# Add parent directory to path to import data.loader
sys.path.append(str(Path(__file__).parent.parent))
from data.loader import get_reviews_with_labels

# Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'joeldiaz',
    'user': 'joeldiaz',
}

QDRANT_URL = "http://localhost:6333"
QDRANT_CONTAINER_NAME = "qdrant"

EMBED_ANYTHING_MODEL = "Qdrant/all-MiniLM-L6-v2-onnx"

INGEST_BATCH_SIZE = 32
INGESTION_SET_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048]
RUNS_INGESTION = 5

SERVING_TEST_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048]
RUNS_SERVING = 5

OUTPUT_DIR = Path(__file__).parent / "output"

model_cache = {}


@dataclass
class ResourceStats:
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
    return stdev(values) if len(values) > 1 else 0.0


def calc_iqr(values: List[float]) -> float:
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
        self.docker_client = docker.from_env()
        try:
            self.container = self.docker_client.containers.get(qd_name)
        except Exception:
            self.container = None

        self.py_baseline = self._get_py_mem()
        self.pg_baseline = self._get_pg_mem()
        self.qd_baseline_stats = self._get_qd_stats() if self.container else None

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
        return self.container.stats(stream=False)

    def _calculate_qd_cpu(self, start_stats, end_stats):
        cpu_delta = end_stats['cpu_stats']['cpu_usage']['total_usage'] - start_stats['cpu_stats']['cpu_usage'][
            'total_usage']
        system_delta = end_stats['cpu_stats']['system_cpu_usage'] - start_stats['cpu_stats']['system_cpu_usage']

        if system_delta > 0.0 and cpu_delta > 0.0:
            cpus = end_stats['cpu_stats'].get('online_cpus', psutil.cpu_count())
            return (cpu_delta / system_delta) * cpus * 100.0
        return 0.0

    @staticmethod
    def measure(py_pid: int, pg_pid: int, func: Callable):
        monitor = ResourceMonitor(py_pid, pg_pid)
        start_time = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start_time

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
# Client & Connectors
# =============================================================================

class EmbedAnythingDirectClient:
    def embed(self, texts: List[str]) -> List[List[float]]:
        model = self._get_model(EMBED_ANYTHING_MODEL)
        data = embed_anything.embed_query(texts, embedder=model)
        return [item.embedding for item in data]

    @staticmethod
    def _get_model(model_name: str):
        if model_name not in model_cache:
            model_cache[model_name] = EmbeddingModel.from_pretrained_onnx(
                WhichModel.Bert, hf_model_id=model_name
            )
        return model_cache[model_name]


class EmbeddingWrapper:
    def __init__(self, fn: Callable):
        self._fn = fn

    def __call__(self, input):
        return self._fn(list(input))


def connect_pg():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    return conn


def get_pg_pid(conn):
    cur = conn.cursor()
    cur.execute("SELECT pg_backend_pid();")
    pid = cur.fetchone()[0]
    cur.close()
    return pid


def create_chroma_client(base_path: str = "./chroma_bench_uc4"):
    db_path = f"{base_path}_{time.time_ns()}"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    client = chromadb.PersistentClient(path=db_path)
    return client, db_path


def cleanup_chroma(client, db_path):
    del client
    time.sleep(1.0)
    if os.path.exists(db_path):
        shutil.rmtree(db_path)


def create_qdrant_client():
    return QdrantClient(url=QDRANT_URL)


# =============================================================================
# Setup / Ingestion Functions
# =============================================================================

def setup_pg_schema(conn):
    cur = conn.cursor()
    cur.execute('''
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE EXTENSION IF NOT EXISTS pg_gembed;
                DROP TABLE IF EXISTS reviews;
                CREATE TABLE reviews
                (
                    id        SERIAL PRIMARY KEY,
                    text      TEXT,
                    spam      BOOLEAN,
                    embedding vector(384)
                );
                ''')
    cur.close()


def populate_pg_database(conn, ingestion_data):
    cur = conn.cursor()
    from psycopg2.extras import execute_values

    # Process in batches
    for i in range(0, len(ingestion_data), INGEST_BATCH_SIZE):
        batch = ingestion_data[i:i + INGEST_BATCH_SIZE]
        args_list = [(t, s) for t, s in batch]
        execute_values(cur, "INSERT INTO reviews (text, spam) VALUES %s", args_list)

        # Update embeddings for the newly inserted rows (where embedding IS NULL)
        cur.execute('''
                    UPDATE reviews t
                    SET embedding = e.embedding
                    FROM embed_texts_with_ids('embed_anything', %s,
                                              (SELECT array_agg(id) FROM reviews WHERE embedding IS NULL),
                                              (SELECT array_agg(text) FROM reviews WHERE embedding IS NULL)) e
                    WHERE t.id = e.sentence_id;
                    ''', (EMBED_ANYTHING_MODEL,))
    cur.close()


def setup_pg_indexed(conn, ingestion_data):
    """Index exists BEFORE embedding generation."""
    setup_pg_schema(conn)
    cur = conn.cursor()
    cur.execute(
        "CREATE INDEX ON reviews USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=100);")
    cur.close()

    populate_pg_database(conn, ingestion_data)
    conn.commit()


def setup_pg_deferred(conn, ingestion_data):
    """Index created AFTER embedding generation."""
    setup_pg_schema(conn)
    populate_pg_database(conn, ingestion_data)

    cur = conn.cursor()
    cur.execute(
        "CREATE INDEX ON reviews USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=100);")
    conn.commit()
    cur.close()


def setup_qdrant_common(client, embed_client, ingestion_data, deferred: bool):
    if client.collection_exists("reviews"):
        client.delete_collection("reviews")

    # Configure HNSW index parameters
    hnsw_config = models.HnswConfigDiff(
        m=16,
        ef_construct=100,
    )

    client.create_collection(
        collection_name="reviews",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE, hnsw_config=hnsw_config)
    )

    texts = [t for t, s in ingestion_data]
    embeddings = embed_client.embed(texts)
    points = [PointStruct(id=i, vector=embeddings[i], payload={"text": t, "spam": s})
              for i, (t, s) in enumerate(ingestion_data)]

    if deferred:
        client.update_collection("reviews", optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0))

    for i in range(0, len(points), INGEST_BATCH_SIZE):
        client.upsert("reviews", points[i:i + INGEST_BATCH_SIZE], wait=True)

    if deferred:
        client.update_collection("reviews",
                                 optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000))


def setup_chroma(client, embed_client, ingestion_data):
    configuration = {
        "hnsw": {
            "space": "cosine",
            "max_neighbors": 16,
            "ef_construction": 100
        }
    }
    emb_obj = EmbeddingWrapper(embed_client.embed)
    collection = client.create_collection("reviews", embedding_function=emb_obj, configuration=configuration)

    texts = [t for t, s in ingestion_data]
    embeddings = embed_client.embed(texts)
    spams = [{"spam": s} for t, s in ingestion_data]
    ids = [str(i) for i in range(len(ingestion_data))]

    for i in range(0, len(texts), INGEST_BATCH_SIZE):
        end = min(i + INGEST_BATCH_SIZE, len(texts))
        collection.add(ids=ids[i:end], embeddings=embeddings[i:end], metadatas=spams[i:end], documents=texts[i:end])
    return collection


# =============================================================================
# Serving Functions
# =============================================================================

def serve_pg(conn, input_texts):
    cur = conn.cursor()
    # Process entire batch at once
    cur.execute('''
                SELECT i.txt, t.spam
                FROM unnest(%s::text[], embed_texts('embed_anything', %s, %s::text[])) AS i(txt, embedding)
                         CROSS JOIN LATERAL (
                    SELECT spam FROM reviews t ORDER BY t.embedding <-> i.embedding LIMIT 5
                    ) t
                ''', (input_texts, EMBED_ANYTHING_MODEL, input_texts))
    _ = cur.fetchall()
    conn.commit()
    cur.close()


def serve_chroma(collection, embed_client, input_texts):
    # Process entire batch at once
    embs = embed_client.embed(input_texts)
    collection.query(query_embeddings=embs, n_results=5, include=["metadatas"])


def serve_qdrant(client, embed_client, input_texts):
    # Process entire batch at once
    embs = embed_client.embed(input_texts)
    # Qdrant search is per-vector, so we must loop, but the embedding is batched.
    for emb in embs:
        client.query_points("reviews", query=emb, limit=5, with_payload=True)


# =============================================================================
# Reporting
# =============================================================================

def compute_metrics(size: int, results: List[BenchmarkResult]) -> dict:
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
            f"{'QD Δ MB':>{col_w}} | {'QD Peak MB':>{col_w}} | {'QD CPU%':>{col_w}} | "
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


def save_serving_csv(all_results, timestamp):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"serving_results_{timestamp}.csv"

    methods = ['pg', 'qdrant', 'chroma']
    metrics = ['throughput', 'time_s', 'py_cpu', 'py_mem_delta', 'py_mem_peak',
               'pg_cpu', 'pg_mem_delta', 'pg_mem_peak',
               'qd_cpu', 'qd_mem_delta', 'qd_mem_peak',
               'sys_cpu', 'sys_mem']

    with open(path, "w", newline='') as f:
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
    print(f"\nServing results saved to {path}")


def generate_plots(metrics, timestamp):
    sizes = [m['size'] for m in metrics]
    methods = ['pg', 'qdrant', 'chroma']
    labels = {'pg': 'PG Unified', 'qdrant': 'Qdrant', 'chroma': 'Chroma'}

    plt.figure(figsize=(10, 6))
    for method in methods:
        y_vals = [m[method]['throughput_median'] for m in metrics]
        y_errs = [m[method]['throughput_iqr'] for m in metrics]
        plt.errorbar(sizes, y_vals, yerr=y_errs, fmt='o-', label=labels[method], capsize=3)

    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (items/s)')
    plt.title('Spam Classification Throughput (Serving) [Median ± IQR]')
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / f"serving_throughput_{timestamp}.png")
    plt.close()
    print(f"Plots saved to {OUTPUT_DIR}")


# =============================================================================
# Main
# =============================================================================

def main():
    # Setup connection and PIDs early for resource monitoring
    pg_conn = connect_pg()
    pg_pid = get_pg_pid(pg_conn)
    py_pid = os.getpid()

    max_ingestion = max(INGESTION_SET_SIZES)
    print(f"Loading {max_ingestion} reviews...")
    full_data_raw = get_reviews_with_labels(max_ingestion + max(SERVING_TEST_SIZES), shuffle=True,
                                            legitimate_only=False)
    full_data = [(text, bool(spam)) for text, spam in full_data_raw]

    # Reserve test pool from the end of the loaded data
    test_pool = full_data[max_ingestion:]

    # Phase 1: Ingestion
    embed_client = EmbedAnythingDirectClient()
    final_chroma_path = None

    print_header("Phase 1: Ingestion Benchmark")
    print_detailed_header()

    for ingestion_size in INGESTION_SET_SIZES:
        print(f"Test Size: {ingestion_size}")

        current_ingestion_data = full_data[:ingestion_size]

        # We need to collect results across runs for each method to compute stats
        results_pg_indexed = []
        results_pg_deferred = []
        results_qd_indexed = []
        results_qd_deferred = []
        results_chroma = []

        for run_idx in range(RUNS_INGESTION):
            # Postgres Indexed
            conn = connect_pg()
            curr_pid = get_pg_pid(conn)
            elapsed, stats = ResourceMonitor.measure(py_pid, curr_pid,
                                                     lambda: setup_pg_indexed(conn, current_ingestion_data))
            results_pg_indexed.append(BenchmarkResult(elapsed, stats))
            conn.close()

            # Postgres Deferred
            conn = connect_pg()
            curr_pid = get_pg_pid(conn)
            elapsed, stats = ResourceMonitor.measure(py_pid, curr_pid,
                                                     lambda: setup_pg_deferred(conn, current_ingestion_data))
            results_pg_deferred.append(BenchmarkResult(elapsed, stats))
            conn.close()

            # Qdrant Indexed
            qd_client = create_qdrant_client()
            elapsed, stats = ResourceMonitor.measure(py_pid, None, lambda: setup_qdrant_common(qd_client, embed_client,
                                                                                               current_ingestion_data,
                                                                                               deferred=False))
            results_qd_indexed.append(BenchmarkResult(elapsed, stats))
            qd_client.close()

            # Qdrant Deferred
            qd_client = create_qdrant_client()
            elapsed, stats = ResourceMonitor.measure(py_pid, None, lambda: setup_qdrant_common(qd_client, embed_client,
                                                                                               current_ingestion_data,
                                                                                               deferred=True))
            results_qd_deferred.append(BenchmarkResult(elapsed, stats))
            qd_client.close()

            # Chroma
            c_client, c_path = create_chroma_client()
            elapsed, stats = ResourceMonitor.measure(py_pid, None, lambda: setup_chroma(c_client, embed_client,
                                                                                        current_ingestion_data))
            results_chroma.append(BenchmarkResult(elapsed, stats))

            # Check if this is the very last run of the very last size
            is_last_run = (run_idx == RUNS_INGESTION - 1) and (ingestion_size == INGESTION_SET_SIZES[-1])

            if is_last_run:
                final_chroma_path = c_path
                del c_client  # Close connection but keep files
                print(f"Keeping Chroma DB at {c_path} for Serving Phase.")
            else:
                cleanup_chroma(c_client, c_path)

        # Print results for this size
        print_result("PG Indexed", results_pg_indexed)
        print_result("PG Deferred", results_pg_deferred)
        print_result("QD Indexed", results_qd_indexed)
        print_result("QD Deferred", results_qd_deferred)
        print_result("Chroma", results_chroma)
    # Phase 2: Serving
    print_header("Phase 2: Serving Benchmark")
    print("Using DBs populated in the last run of Phase 1...")

    # Qdrant (Data left from Qdrant Deferred)
    qd_client = create_qdrant_client()

    # Chroma (Data left at final_chroma_path)
    if final_chroma_path:
        c_client = chromadb.PersistentClient(path=final_chroma_path)
        c_collection = c_client.get_collection("reviews")
    else:
        print("Error: No Chroma DB path retained.")
        return

    all_metrics = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print_detailed_header()

    for size in SERVING_TEST_SIZES:
        print(f"Test Size: {size}")
        test_inputs = [t for t, l in test_pool[:size]]

        m_pg = []
        for _ in range(RUNS_SERVING):
            elapsed, stats = ResourceMonitor.measure(py_pid, pg_pid, lambda: serve_pg(pg_conn, test_inputs))
            m_pg.append(BenchmarkResult(elapsed, stats))

        m_qd = []
        for _ in range(RUNS_SERVING):
            elapsed, stats = ResourceMonitor.measure(py_pid, None,
                                                     lambda: serve_qdrant(qd_client, embed_client, test_inputs))
            m_qd.append(BenchmarkResult(elapsed, stats))

        m_ch = []
        for _ in range(RUNS_SERVING):
            elapsed, stats = ResourceMonitor.measure(py_pid, None,
                                                     lambda: serve_chroma(c_collection, embed_client, test_inputs))
            m_ch.append(BenchmarkResult(elapsed, stats))

        print_result("PG Unified", m_pg)
        print_result("Qdrant", m_qd)
        print_result("Chroma", m_ch)
        print()

        res = {
            'size': size,
            'pg': compute_metrics(size, m_pg),
            'qdrant': compute_metrics(size, m_qd),
            'chroma': compute_metrics(size, m_ch)
        }
        all_metrics.append(res)

    save_serving_csv(all_metrics, timestamp)
    generate_plots(all_metrics, timestamp)

    # Cleanup: Prune all testing data
    print("\nPruning testing data...")

    # Postgres
    cur = pg_conn.cursor()
    cur.execute("DROP TABLE IF EXISTS reviews;")
    pg_conn.commit()
    cur.close()
    pg_conn.close()

    # Qdrant
    if qd_client.collection_exists("reviews"):
        qd_client.delete_collection("reviews")
    qd_client.close()

    # Chroma
    if final_chroma_path:
        cleanup_chroma(c_client, final_chroma_path)


if __name__ == "__main__":
    main()
