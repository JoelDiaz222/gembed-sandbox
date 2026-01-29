import glob
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev, quantiles
from typing import Callable, List

import chromadb
import docker
import embed_anything
import psutil
import psycopg2
from embed_anything import EmbeddingModel, WhichModel
from plot_utils import save_results_csv, generate_plots
from psycopg2.extras import execute_values
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'joeldiaz',
    'user': 'joeldiaz',
}

QDRANT_URL = "http://localhost:6333"
QDRANT_CONTAINER_NAME = "qdrant"

MODEL_NAME = "openai/clip-vit-base-patch32"

INGESTION_SET_SIZES = [16, 32, 64]
RUNS_INGESTION = 3

SERVING_TEST_SIZES = [16, 32, 64]
RUNS_SERVING = 5

# Directories
OUTPUT_DIR = Path(__file__).parent / "output"
DATA_DIR = Path(__file__).parent.parent / "data" / "CUSTOMER_IMAGES"

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
        try:
            cpu_delta = end_stats['cpu_stats']['cpu_usage']['total_usage'] - start_stats['cpu_stats']['cpu_usage'][
                'total_usage']
            system_delta = end_stats['cpu_stats']['system_cpu_usage'] - start_stats['cpu_stats']['system_cpu_usage']

            if system_delta > 0.0 and cpu_delta > 0.0:
                cpus = end_stats['cpu_stats'].get('online_cpus', psutil.cpu_count())
                return (cpu_delta / system_delta) * cpus * 100.0
        except KeyError:
            pass
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
            # Docker stats parsing
            if 'usage' in end_qd_stats['memory_stats']:
                qd_peak_raw = end_qd_stats['memory_stats']['usage']
            else:
                qd_peak_raw = end_qd_stats['memory_stats'].get('limit', 0)

            qd_peak_mb = qd_peak_raw / (1024 * 1024)
            baseline_raw = 0
            if 'usage' in monitor.qd_baseline_stats['memory_stats']:
                baseline_raw = monitor.qd_baseline_stats['memory_stats']['usage']

            qd_delta_mb = qd_peak_mb - (baseline_raw / (1024 * 1024))
            qd_cpu = monitor._calculate_qd_cpu(monitor.qd_baseline_stats, end_qd_stats)

        sys_mem = psutil.virtual_memory()
        sys_mem_mb = sys_mem.used / (1024 * 1024)
        sys_cpu = psutil.cpu_percent()

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
            sys_mem_mb=sys_mem_mb,
            sys_cpu=sys_cpu
        )
        return elapsed, stats


# =============================================================================
# Client & Helpers
# =============================================================================

class EmbedAnythingImageClient:
    def embed_files(self, paths: List[str]) -> List[List[float]]:
        model = self._get_model()
        # Create a temporary directory for the batch
        base_temp = Path("temp_bench_imgs_batch")
        if base_temp.exists():
            shutil.rmtree(base_temp)
        base_temp.mkdir()

        try:
            # Copy files to temp dir with ordered names
            for idx, p in enumerate(paths):
                ext = Path(p).suffix
                fname = f"{idx:06d}{ext}"  # Use zero-padded index for sorting
                shutil.copy(p, base_temp / fname)

            # Embed directory
            res = embed_anything.embed_image_directory(str(base_temp), embedder=model)

            embeddings = []
            if isinstance(res, list):
                for item in res:
                    if hasattr(item, 'embedding'):
                        embeddings.append(item.embedding)
            return embeddings

        finally:
            if base_temp.exists():
                shutil.rmtree(base_temp)

    @staticmethod
    def _get_model():
        if MODEL_NAME not in model_cache:
            model_cache[MODEL_NAME] = EmbeddingModel.from_pretrained_hf(MODEL_NAME)
        return model_cache[MODEL_NAME]


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


def create_chroma_client(base_path: str = "./chroma_bench_uc9"):
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


def get_image_paths(n: int) -> List[str]:
    """Get n image paths from the data directory, looping if necessary."""
    all_images = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        all_images.extend(glob.glob(str(DATA_DIR / "**" / ext), recursive=True))

    if not all_images:
        raise ValueError(f"No images found in {DATA_DIR}")

    if len(all_images) < n:
        mult = (n // len(all_images)) + 1
        all_images = all_images * mult

    return [os.path.abspath(p) for p in all_images[:n]]


# =============================================================================
# Setup / Ingestion Functions
# =============================================================================

def setup_pg_schema(conn):
    cur = conn.cursor()
    cur.execute('''
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE EXTENSION IF NOT EXISTS pg_gembed;

                DROP TABLE IF EXISTS faces;
                CREATE TABLE faces
                (
                    id         SERIAL PRIMARY KEY,
                    path       TEXT,
                    image_data BYTEA,
                    embedding  vector(512)
                );
                ''')
    conn.commit()
    cur.close()


def ingest_pg_batch(conn, image_paths: List[str], store_blob: bool):
    """
    Direct ingestion: Client sends images -> DB Generates Vector -> DB Stores Row.
    """
    cur = conn.cursor()

    chunk_size = 32
    total = len(image_paths)

    for i in range(0, total, chunk_size):
        batch_paths = image_paths[i: i + chunk_size]
        batch_images = []
        for p in batch_paths:
            with open(p, "rb") as f:
                batch_images.append(f.read())

        if store_blob:
            sql = '''
                  INSERT INTO faces (path, image_data, embedding)
                  SELECT t.p, t.i, t.e
                  FROM unnest(%s::text[],
                              %s::bytea[],
                              embed_images('embed_anything', %s, %s::bytea[])) AS t(p, i, e)
                  '''
            cur.execute(sql, (batch_paths, batch_images, MODEL_NAME, batch_images))
        else:
            sql = '''
                  INSERT INTO faces (path, embedding)
                  SELECT t.p, t.e
                  FROM unnest(%s::text[],
                              embed_images('embed_anything', %s, %s::bytea[])) AS t(p, e)
                  '''
            cur.execute(sql, (batch_paths, MODEL_NAME, batch_images))

    conn.commit()
    cur.close()


def setup_pg_variant(conn, image_paths, store_blob: bool, deferred: bool):
    setup_pg_schema(conn)
    cur = conn.cursor()

    if not deferred:
        cur.execute("CREATE INDEX ON faces USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=100);")
        conn.commit()

    ingest_pg_batch(conn, image_paths, store_blob)

    if deferred:
        cur.execute("CREATE INDEX ON faces USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=100);")
        conn.commit()

    cur.close()


def setup_qdrant_common(client, image_paths, deferred: bool):
    if client.collection_exists("faces"):
        client.delete_collection("faces")

    hnsw_config = models.HnswConfigDiff(m=16, ef_construct=100)
    client.create_collection(
        collection_name="faces",
        vectors_config=VectorParams(size=512, distance=Distance.COSINE, hnsw_config=hnsw_config)
    )

    if deferred:
        client.update_collection("faces", optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0))

    embed_client = EmbedAnythingImageClient()

    # Process in chunks to avoid memory spikes on client side
    chunk_size = 32
    total = len(image_paths)
    for i in range(0, total, chunk_size):
        batch_paths = image_paths[i: i + chunk_size]
        embeddings = embed_client.embed_files(batch_paths)

        points = [
            PointStruct(id=i + j, vector=emb, payload={"path": path})
            for j, (emb, path) in enumerate(zip(embeddings, batch_paths))
        ]

        if points:
            client.upsert("faces", points, wait=True)

    if deferred:
        client.update_collection("faces", optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000))


def setup_chroma(client, image_paths):
    configuration = {
        "hnsw": {
            "space": "cosine",
            "max_neighbors": 16,
            "ef_construction": 100
        }
    }
    collection = client.create_collection("faces", configuration=configuration)

    embed_client = EmbedAnythingImageClient()

    chunk_size = 32
    total = len(image_paths)
    for i in range(0, total, chunk_size):
        batch_paths = image_paths[i: i + chunk_size]
        embeddings = embed_client.embed_files(batch_paths)

        ids = [str(i + j) for j in range(len(batch_paths))]
        metas = [{"path": p} for p in batch_paths]

        if embeddings:
            collection.add(ids=ids, embeddings=embeddings, metadatas=metas)


# =============================================================================
# Serving Functions
# =============================================================================

def serve_pg_classify(conn, query_image_paths):
    """Scenario 1: Classify (Search & Return Metadata only)"""
    images_data = []
    for p in query_image_paths:
        with open(p, "rb") as f:
            images_data.append(f.read())

    cur = conn.cursor()
    sql = '''
          WITH queries AS (SELECT i.ord, i.embedding
                           FROM unnest(embed_images('embed_anything', %s, %s::bytea[])) WITH ORDINALITY AS i(embedding, ord))
          SELECT q.ord, f.path
          FROM queries q
                   CROSS JOIN LATERAL (
              SELECT path
              FROM faces f
              ORDER BY f.embedding <-> q.embedding
              LIMIT 1
              ) f
          ORDER BY q.ord;
          '''
    cur.execute(sql, (MODEL_NAME, images_data))
    _ = cur.fetchall()
    conn.commit()
    cur.close()


def serve_pg_retrieve(conn, query_image_paths):
    """Scenario 2: Retrieve (Search & Return Actual Image Data)"""
    images_data = []
    for p in query_image_paths:
        with open(p, "rb") as f:
            images_data.append(f.read())

    cur = conn.cursor()
    sql = '''
          WITH queries AS (SELECT i.ord, i.embedding
                           FROM unnest(embed_images('embed_anything', %s, %s::bytea[])) WITH ORDINALITY AS i(embedding, ord))
          SELECT q.ord, f.path, f.image_data
          FROM queries q
                   CROSS JOIN LATERAL (
              SELECT path, image_data
              FROM faces f
              ORDER BY f.embedding <-> q.embedding
              LIMIT 1
              ) f
          ORDER BY q.ord;
          '''
    cur.execute(sql, (MODEL_NAME, images_data))
    _ = cur.fetchall()
    conn.commit()
    cur.close()


def serve_qdrant_classify(client, embed_client, query_image_paths):
    """Scenario 1: Classify (Search & Return Metadata)"""
    embeddings = embed_client.embed_files(query_image_paths)

    requests = [
        models.QueryRequest(query=emb, limit=1, with_payload=True)
        for emb in embeddings
    ]
    if requests:
        _ = client.query_batch_points(collection_name="faces", requests=requests)


def serve_qdrant_retrieve(client, embed_client, query_image_paths):
    """Scenario 2: Retrieve (Search & Fetch File from Disk)"""
    embeddings = embed_client.embed_files(query_image_paths)

    requests = [
        models.QueryRequest(query=emb, limit=1, with_payload=True)
        for emb in embeddings
    ]
    if requests:
        results = client.query_batch_points(collection_name="faces", requests=requests)
        for batch in results:
            for point in batch.points:
                path = point.payload.get('path')
                if path:
                    with open(path, 'rb') as f:
                        _ = f.read()


def serve_chroma_classify(collection, embed_client, query_image_paths):
    """Scenario 1: Classify (Search & Return Metadata)"""
    embeddings = embed_client.embed_files(query_image_paths)
    if embeddings:
        _ = collection.query(query_embeddings=embeddings, n_results=1)


def serve_chroma_retrieve(collection, embed_client, query_image_paths):
    """Scenario 2: Retrieve (Search & Fetch File from Disk)"""
    embeddings = embed_client.embed_files(query_image_paths)
    if embeddings:
        results = collection.query(query_embeddings=embeddings, n_results=1)
        if results.get('metadatas'):
            for batch_meta in results['metadatas']:
                for meta in batch_meta:
                    path = meta.get('path')
                    if path:
                        with open(path, 'rb') as f:
                            _ = f.read()


# =============================================================================
# Reporting & Main
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
        'time_s': mean(times),
        'py_cpu': mean(py_cpu),
        'py_mem_peak': mean(py_peak),
        'pg_cpu': mean(pg_cpu),
        'pg_mem_peak': mean(pg_peak),
        'qd_cpu': mean(qd_cpu),
        'qd_mem_peak': mean(qd_peak),
        'sys_cpu': mean(sys_cpu),
        'sys_mem': mean(sys_mem),
    }


def print_detailed_header():
    lbl_w, time_w, col_w = 20, 14, 13
    print("\nBenchmark Results:", flush=True)
    header = (
            "  " +
            f"{'':{lbl_w}} | {'Time (s)':>{time_w}} | "
            f"{'Py Δ MB':>{col_w}} | {'PG Peak MB':>{col_w}} | {'QD Peak MB':>{col_w}} | "
            f"{'Sys CPU%':>{col_w}}"
    )
    print(header, flush=True)
    print("=" * len(header), flush=True)


def print_result(label: str, results: List[BenchmarkResult]):
    times = [r.time_s for r in results]
    py_deltas = [r.stats.py_delta_mb for r in results]
    pg_peaks = [r.stats.pg_peak_mb for r in results]
    qd_peaks = [r.stats.qd_peak_mb for r in results]
    sys_cpus = [r.stats.sys_cpu for r in results]

    def fmt(vals, p=1):
        return f"{mean(vals):.{p}f}±{safe_stdev(vals):.{p}f}"

    row_fmt = (
        "  {label:<20} | {time:>14} | "
        "{pyd:>13} | {pgp:>13} | {qdp:>13} | "
        "{sysc:>13}"
    )
    print(row_fmt.format(
        label=label,
        time=fmt(times, 3),
        pyd=fmt(py_deltas), pgp=fmt(pg_peaks), qdp=fmt(qd_peaks),
        sysc=fmt(sys_cpus)
    ), flush=True)


def main():
    pg_conn = connect_pg()
    pg_pid = get_pg_pid(pg_conn)
    py_pid = os.getpid()

    # Pre-warm model cache
    print("Loading model...", flush=True)
    EmbedAnythingImageClient._get_model()

    print_detailed_header()

    # PHASE 1: INGESTION
    print("\n=== Phase 1: Ingestion ===")

    all_ingestion_metrics = []
    final_chroma_path = None
    embed_client = EmbedAnythingImageClient()

    for size in INGESTION_SET_SIZES:
        print(f"\nTest Size: {size}")
        current_images = get_image_paths(size)

        # PG Unified (Blob + Vector)
        m_pg_uni_idx = []
        for _ in range(RUNS_INGESTION):
            c = connect_pg()
            t, s = ResourceMonitor.measure(py_pid, get_pg_pid(c),
                                           lambda: setup_pg_variant(c, current_images, store_blob=True, deferred=False))
            m_pg_uni_idx.append(BenchmarkResult(t, s))
            c.close()
        print_result("PG Unified Idx", m_pg_uni_idx)

        m_pg_uni_def = []
        for _ in range(RUNS_INGESTION):
            c = connect_pg()
            t, s = ResourceMonitor.measure(py_pid, get_pg_pid(c),
                                           lambda: setup_pg_variant(c, current_images, store_blob=True, deferred=True))
            m_pg_uni_def.append(BenchmarkResult(t, s))
            c.close()
        print_result("PG Unified Def", m_pg_uni_def)

        # PG VectorOnly (Path + Vector)
        m_pg_vec_idx = []
        for _ in range(RUNS_INGESTION):
            c = connect_pg()
            t, s = ResourceMonitor.measure(py_pid, get_pg_pid(c),
                                           lambda: setup_pg_variant(c, current_images, store_blob=False,
                                                                    deferred=False))
            m_pg_vec_idx.append(BenchmarkResult(t, s))
            c.close()
        print_result("PG VectorOnly Idx", m_pg_vec_idx)

        m_pg_vec_def = []
        for _ in range(RUNS_INGESTION):
            c = connect_pg()
            t, s = ResourceMonitor.measure(py_pid, get_pg_pid(c),
                                           lambda: setup_pg_variant(c, current_images, store_blob=False, deferred=True))
            m_pg_vec_def.append(BenchmarkResult(t, s))
            c.close()
        print_result("PG VectorOnly Def", m_pg_vec_def)

        # Qdrant Indexed
        m_qd_i = []
        for _ in range(RUNS_INGESTION):
            qc = create_qdrant_client()
            t, s = ResourceMonitor.measure(py_pid, None, lambda: setup_qdrant_common(qc, current_images, False))
            m_qd_i.append(BenchmarkResult(t, s))
            qc.close()
        print_result("Qdrant Indexed", m_qd_i)

        # Qdrant Deferred
        m_qd_d = []
        for _ in range(RUNS_INGESTION):
            qc = create_qdrant_client()
            t, s = ResourceMonitor.measure(py_pid, None, lambda: setup_qdrant_common(qc, current_images, True))
            m_qd_d.append(BenchmarkResult(t, s))
            qc.close()
        print_result("Qdrant Deferred", m_qd_d)

        # Chroma
        m_ch = []
        for i in range(RUNS_INGESTION):
            cc, cp = create_chroma_client()
            t, s = ResourceMonitor.measure(py_pid, None, lambda: setup_chroma(cc, current_images))
            m_ch.append(BenchmarkResult(t, s))

            # Keep last run for Serving
            if size == max(SERVING_TEST_SIZES) and i == RUNS_INGESTION - 1:
                final_chroma_path = cp
                print(f"Retaining Chroma DB for serving at {cp}")
                del cc
            else:
                cleanup_chroma(cc, cp)
        print_result("Chroma", m_ch)

        all_ingestion_metrics.append({
            'size': size,
            'pg_unified_indexed': compute_metrics(size, m_pg_uni_idx),
            'pg_unified_deferred': compute_metrics(size, m_pg_uni_def),
            'pg_vectoronly_indexed': compute_metrics(size, m_pg_vec_idx),
            'pg_vectoronly_deferred': compute_metrics(size, m_pg_vec_def),
            'qd_indexed': compute_metrics(size, m_qd_i),
            'qd_deferred': compute_metrics(size, m_qd_d),
            'chroma': compute_metrics(size, m_ch),
        })

    # Save Ingestion Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ing_methods = ['pg_unified_indexed', 'pg_unified_deferred', 'pg_vectoronly_indexed', 'pg_vectoronly_deferred',
                   'qd_indexed', 'qd_deferred', 'chroma']
    save_results_csv(all_ingestion_metrics, OUTPUT_DIR / "ingestion", timestamp, ing_methods)
    generate_plots(all_ingestion_metrics, OUTPUT_DIR / "ingestion", timestamp, ing_methods)

    # PHASE 2: SERVING
    print("\n=== Phase 2: Serving ===")

    # Setup DBs with max serving size
    max_serving = max(SERVING_TEST_SIZES)
    serving_images = get_image_paths(max_serving)

    # PG: Populate once with max size (Using Unified Deferred for best retrieval performance)
    print("Preparing PG for Serving...")
    pg_serving_conn = connect_pg()
    setup_pg_variant(pg_serving_conn, serving_images, store_blob=True, deferred=True)

    # Qdrant: Populate once
    print("Preparing Qdrant for Serving...")
    qd_serving_client = create_qdrant_client()
    setup_qdrant_common(qd_serving_client, serving_images, deferred=True)

    # Chroma: Already populated (if logic above held, else re-pop)
    if not final_chroma_path:
        cc, final_chroma_path = create_chroma_client()
        setup_chroma(cc, serving_images)
        del cc

    c_client = chromadb.PersistentClient(path=final_chroma_path)
    c_collection = c_client.get_collection("faces")

    all_serving_metrics_classify = []
    all_serving_metrics_retrieve = []

    for size in SERVING_TEST_SIZES:
        print(f"\nTest Size: {size}")
        current_queries = get_image_paths(size)

        # --- Scenario 1: Classify (Search Only) ---
        print("  [Scenario: Classify]")

        # PG
        m_pg = []
        for _ in range(RUNS_SERVING):
            t, s = ResourceMonitor.measure(py_pid, get_pg_pid(pg_serving_conn),
                                           lambda: serve_pg_classify(pg_serving_conn, current_queries))
            m_pg.append(BenchmarkResult(t, s))
        print_result("PG Classify", m_pg)

        # Qdrant
        m_qd = []
        for _ in range(RUNS_SERVING):
            t, s = ResourceMonitor.measure(py_pid, None,
                                           lambda: serve_qdrant_classify(qd_serving_client, embed_client,
                                                                         current_queries))
            m_qd.append(BenchmarkResult(t, s))
        print_result("Qdrant Classify", m_qd)

        # Chroma
        m_ch = []
        for _ in range(RUNS_SERVING):
            t, s = ResourceMonitor.measure(py_pid, None,
                                           lambda: serve_chroma_classify(c_collection, embed_client, current_queries))
            m_ch.append(BenchmarkResult(t, s))
        print_result("Chroma Classify", m_ch)

        all_serving_metrics_classify.append({
            'size': size,
            'pg': compute_metrics(size, m_pg),
            'qdrant': compute_metrics(size, m_qd),
            'chroma': compute_metrics(size, m_ch)
        })

        # --- Scenario 2: Retrieve (Search + Fetch) ---
        print("  [Scenario: Retrieve]")

        # PG
        m_pg = []
        for _ in range(RUNS_SERVING):
            t, s = ResourceMonitor.measure(py_pid, get_pg_pid(pg_serving_conn),
                                           lambda: serve_pg_retrieve(pg_serving_conn, current_queries))
            m_pg.append(BenchmarkResult(t, s))
        print_result("PG Retrieve", m_pg)

        # Qdrant
        m_qd = []
        for _ in range(RUNS_SERVING):
            t, s = ResourceMonitor.measure(py_pid, None,
                                           lambda: serve_qdrant_retrieve(qd_serving_client, embed_client,
                                                                         current_queries))
            m_qd.append(BenchmarkResult(t, s))
        print_result("Qdrant Retrieve", m_qd)

        # Chroma
        m_ch = []
        for _ in range(RUNS_SERVING):
            t, s = ResourceMonitor.measure(py_pid, None,
                                           lambda: serve_chroma_retrieve(c_collection, embed_client, current_queries))
            m_ch.append(BenchmarkResult(t, s))
        print_result("Chroma Retrieve", m_ch)

        all_serving_metrics_retrieve.append({
            'size': size,
            'pg': compute_metrics(size, m_pg),
            'qdrant': compute_metrics(size, m_qd),
            'chroma': compute_metrics(size, m_ch)
        })

    # Save Serving Results
    serv_methods = ['pg', 'qdrant', 'chroma']
    save_results_csv(all_serving_metrics_classify, OUTPUT_DIR / "serving_classify", timestamp, serv_methods)
    generate_plots(all_serving_metrics_classify, OUTPUT_DIR / "serving_classify", timestamp, serv_methods)

    save_results_csv(all_serving_metrics_retrieve, OUTPUT_DIR / "serving_retrieve", timestamp, serv_methods)
    generate_plots(all_serving_metrics_retrieve, OUTPUT_DIR / "serving_retrieve", timestamp, serv_methods)

    # Cleanup
    pg_serving_conn.close()
    qd_serving_client.close()
    cleanup_chroma(c_client, final_chroma_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
