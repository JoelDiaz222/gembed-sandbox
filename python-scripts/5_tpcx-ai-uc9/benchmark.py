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

INGESTION_SET_SIZES = [100]
RUNS_INGESTION = 1

SERVING_TEST_SIZES = [100]
RUNS_SERVING = 1

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
        base_temp = Path("temp_bench_imgs_batch")
        if base_temp.exists():
            shutil.rmtree(base_temp)
        base_temp.mkdir()

        try:
            for idx, p in enumerate(paths):
                ext = Path(p).suffix
                fname = f"{idx:06d}{ext}"
                shutil.copy(p, base_temp / fname)

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


def connect_and_get_pid():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("SELECT pg_backend_pid();")
    pid = cur.fetchone()[0]
    cur.close()
    conn.commit()
    return conn, pid


def connect_pg():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    return conn


def get_pg_pid(conn):
    try:
        cur = conn.cursor()
        cur.execute("SELECT pg_backend_pid();")
        pid = cur.fetchone()[0]
        cur.close()
        return pid
    except:
        return None


def create_chroma_client(base_path: str = "./chroma_bench_uc9"):
    db_path = f"{base_path}_{time.time_ns()}"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    client = chromadb.PersistentClient(path=db_path)
    return client, db_path


def cleanup_chroma(client, db_path):
    del client
    time.sleep(0.5)
    if os.path.exists(db_path):
        shutil.rmtree(db_path)


def create_qdrant_client():
    return QdrantClient(url=QDRANT_URL)


def get_image_paths(n: int) -> List[str]:
    all_images = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        all_images.extend(glob.glob(str(DATA_DIR / "**" / ext), recursive=True))

    if not all_images:
        raise ValueError(f"No images found in {DATA_DIR}")

    if len(all_images) < n:
        mult = (n // len(all_images)) + 1
        all_images = all_images * mult

    return [os.path.abspath(p) for p in all_images[:n]]


def get_person_name(path: str) -> str:
    return Path(path).parent.name


def setup_pg_schema(conn):
    cur = conn.cursor()
    cur.execute('''
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE EXTENSION IF NOT EXISTS pg_gembed;

                DROP TABLE IF EXISTS faces;
                CREATE TABLE faces
                (
                    id          SERIAL PRIMARY KEY,
                    path        TEXT NOT NULL,
                    person_name TEXT NOT NULL,
                    image_data  BYTEA,
                    embedding   vector(512)
                );
                ''')
    conn.commit()
    cur.close()


def truncate_pg_table(conn):
    cur = conn.cursor()
    cur.execute("TRUNCATE faces RESTART IDENTITY")
    conn.commit()
    cur.close()


# =============================================================================
# Core Logic Functions
# =============================================================================

# --- Scenario 1: Paths Only ---

def s1_ingest_pg(conn, image_paths):
    cur = conn.cursor()
    # Note: TRUNCATE handled by runner

    # Ensure Index
    cur.execute(
        "CREATE INDEX IF NOT EXISTS faces_emb_idx ON faces USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=100);")

    chunk_size = 32
    total = len(image_paths)
    for i in range(0, total, chunk_size):
        batch_paths = image_paths[i: i + chunk_size]
        batch_names = [get_person_name(p) for p in batch_paths]
        batch_images = []
        for p in batch_paths:
            with open(p, "rb") as f:
                batch_images.append(f.read())

        sql = '''
              INSERT INTO faces (path, person_name, embedding)
              SELECT t.p, t.n, t.e
              FROM unnest(%s::text[],
                          %s::text[],
                          embed_images('embed_anything', %s, %s::bytea[])) AS t(p, n, e)
              '''
        cur.execute(sql, (batch_paths, batch_names, MODEL_NAME, batch_images))
    conn.commit()
    cur.close()


def s1_ingest_qdrant(client, image_paths, embed_client):
    if client.collection_exists("faces"):
        client.delete_collection("faces")

    hnsw_config = models.HnswConfigDiff(m=16, ef_construct=100)
    client.create_collection(
        collection_name="faces",
        vectors_config=VectorParams(size=512, distance=Distance.COSINE, hnsw_config=hnsw_config)
    )

    chunk_size = 32
    total = len(image_paths)
    for i in range(0, total, chunk_size):
        batch_paths = image_paths[i: i + chunk_size]
        embeddings = embed_client.embed_files(batch_paths)
        points = [
            PointStruct(
                id=i + j,
                vector=emb,
                payload={"path": p, "person_name": get_person_name(p)}
            )
            for j, (emb, p) in enumerate(zip(embeddings, batch_paths))
        ]
        if points:
            client.upsert("faces", points, wait=True)


def s1_ingest_chroma(collection, image_paths, embed_client):
    # Collection is assumed fresh/empty or handled by runner
    chunk_size = 32
    total = len(image_paths)
    for i in range(0, total, chunk_size):
        batch_paths = image_paths[i: i + chunk_size]
        embeddings = embed_client.embed_files(batch_paths)
        ids = [str(i + j) for j in range(len(batch_paths))]
        metas = [{"path": p, "person_name": get_person_name(p)} for p in batch_paths]
        if embeddings:
            collection.add(ids=ids, embeddings=embeddings, metadatas=metas)


# --- Scenario 2: Blobs in PG ---

def s2_ingest_pg(conn, image_paths):
    cur = conn.cursor()
    # TRUNCATE handled by runner

    cur.execute(
        "CREATE INDEX IF NOT EXISTS faces_emb_idx ON faces USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=100);")

    chunk_size = 32
    total = len(image_paths)
    for i in range(0, total, chunk_size):
        batch_paths = image_paths[i: i + chunk_size]
        batch_names = [get_person_name(p) for p in batch_paths]
        batch_images = []
        for p in batch_paths:
            with open(p, "rb") as f:
                batch_images.append(f.read())

        sql = '''
              INSERT INTO faces (path, person_name, image_data, embedding)
              SELECT t.p, t.n, t.i, t.e
              FROM unnest(%s::text[],
                          %s::text[],
                          %s::bytea[],
                          embed_images('embed_anything', %s, %s::bytea[])) AS t(p, n, i, e)
              '''
        cur.execute(sql, (batch_paths, batch_names, batch_images, MODEL_NAME, batch_images))
    conn.commit()
    cur.close()


def s2_ingest_dist_common(conn, image_paths):
    """Common PG ingestion for distributed S2 (Insert Blob, return IDs)."""
    cur = conn.cursor()
    # TRUNCATE handled by runner

    all_pg_ids = []

    chunk_size = 32
    total = len(image_paths)
    for i in range(0, total, chunk_size):
        batch_paths = image_paths[i: i + chunk_size]
        batch_names = [get_person_name(p) for p in batch_paths]
        batch_images = []
        for p in batch_paths:
            with open(p, "rb") as f:
                batch_images.append(f.read())

        # Insert blobs
        execute_values(
            cur,
            "INSERT INTO faces (path, person_name, image_data) VALUES %s RETURNING id",
            list(zip(batch_paths, batch_names, batch_images))
        )
        # Fetch IDs
        ids = [r[0] for r in cur.fetchall()]
        all_pg_ids.extend(ids)

    conn.commit()
    cur.close()
    return all_pg_ids


def s2_ingest_qdrant(conn, client, image_paths, embed_client):
    # 1. Insert to PG (PG is truncated by runner before this)
    pg_ids = s2_ingest_dist_common(conn, image_paths)

    # 2. Setup Qdrant
    if client.collection_exists("faces"):
        client.delete_collection("faces")
    hnsw_config = models.HnswConfigDiff(m=16, ef_construct=100)
    client.create_collection(
        collection_name="faces",
        vectors_config=VectorParams(size=512, distance=Distance.COSINE, hnsw_config=hnsw_config)
    )

    # 3. Embed & Push
    chunk_size = 32
    total = len(image_paths)
    current_idx = 0

    for i in range(0, total, chunk_size):
        batch_paths = image_paths[i: i + chunk_size]
        # Get corresponding PG IDs
        batch_ids = pg_ids[current_idx: current_idx + len(batch_paths)]
        current_idx += len(batch_paths)

        embeddings = embed_client.embed_files(batch_paths)

        points = [
            PointStruct(
                id=pg_id,
                vector=emb,
                payload={"path": p, "person_name": get_person_name(p), "pg_id": pg_id}
            )
            for pg_id, emb, p in zip(batch_ids, embeddings, batch_paths)
        ]
        if points:
            client.upsert("faces", points, wait=True)


def s2_ingest_chroma(conn, collection, image_paths, embed_client):
    # 1. Insert to PG
    pg_ids = s2_ingest_dist_common(conn, image_paths)

    # 2. Embed & Push
    chunk_size = 32
    total = len(image_paths)
    current_idx = 0

    for i in range(0, total, chunk_size):
        batch_paths = image_paths[i: i + chunk_size]
        batch_ids = pg_ids[current_idx: current_idx + len(batch_paths)]
        current_idx += len(batch_paths)

        embeddings = embed_client.embed_files(batch_paths)

        ids = [str(pg_id) for pg_id in batch_ids]
        metas = [{"path": p, "person_name": get_person_name(p), "pg_id": pg_id}
                 for p, pg_id in zip(batch_paths, batch_ids)]

        if embeddings:
            collection.add(ids=ids, embeddings=embeddings, metadatas=metas)


# --- Serving Functions ---

TOP_K = 5


def serve_s1_pg(conn, query_paths):
    images_data = [open(p, "rb").read() for p in query_paths]
    cur = conn.cursor()
    sql = '''
          WITH queries AS (SELECT i.ord, i.embedding
                           FROM unnest(embed_images('embed_anything', %s, %s::bytea[]))
                                    WITH ORDINALITY AS i(embedding, ord))
          SELECT q.ord, f.path
          FROM queries q
                   CROSS JOIN LATERAL (
              SELECT path
              FROM faces f
              ORDER BY f.embedding <-> q.embedding
              LIMIT %s
              ) f
          ORDER BY q.ord;
          '''
    cur.execute(sql, (MODEL_NAME, images_data, TOP_K))
    _ = cur.fetchall()
    conn.commit()
    cur.close()


def serve_s1_qdrant(client, embed_client, query_paths):
    embeddings = embed_client.embed_files(query_paths)
    requests = [
        models.QueryRequest(query=emb, limit=TOP_K, with_payload=True)
        for emb in embeddings
    ]
    if requests:
        _ = client.query_batch_points(collection_name="faces", requests=requests)


def serve_s1_chroma(collection, embed_client, query_paths):
    embeddings = embed_client.embed_files(query_paths)
    if embeddings:
        _ = collection.query(query_embeddings=embeddings, n_results=TOP_K)


def serve_s2_pg(conn, query_paths):
    images_data = [open(p, "rb").read() for p in query_paths]
    cur = conn.cursor()
    sql = '''
          WITH queries AS (SELECT i.ord, i.embedding
                           FROM unnest(embed_images('embed_anything', %s, %s::bytea[]))
                                    WITH ORDINALITY AS i(embedding, ord))
          SELECT q.ord, f.image_data
          FROM queries q
                   CROSS JOIN LATERAL (
              SELECT image_data
              FROM faces f
              ORDER BY f.embedding <-> q.embedding
              LIMIT %s
              ) f
          ORDER BY q.ord;
          '''
    cur.execute(sql, (MODEL_NAME, images_data, TOP_K))
    _ = cur.fetchall()
    conn.commit()
    cur.close()


def serve_s2_qdrant(client, conn, embed_client, query_paths):
    embeddings = embed_client.embed_files(query_paths)
    requests = [
        models.QueryRequest(query=emb, limit=TOP_K, with_payload=True)
        for emb in embeddings
    ]
    if not requests: return

    results = client.query_batch_points(collection_name="faces", requests=requests)

    all_pg_ids = []
    for batch in results:
        for point in batch.points:
            pid = point.payload.get('pg_id')
            if pid is not None:
                all_pg_ids.append(pid)

    if all_pg_ids:
        cur = conn.cursor()
        sql = "SELECT image_data FROM faces WHERE id = ANY(%s::int[])"
        cur.execute(sql, (all_pg_ids,))
        _ = cur.fetchall()
        cur.close()
        conn.commit()


def serve_s2_chroma(collection, conn, embed_client, query_paths):
    embeddings = embed_client.embed_files(query_paths)
    if not embeddings: return

    results = collection.query(query_embeddings=embeddings, n_results=TOP_K)

    all_pg_ids = []
    if results.get('metadatas'):
        for batch_meta in results['metadatas']:
            for meta in batch_meta:
                pid = meta.get('pg_id')
                if pid is not None:
                    all_pg_ids.append(pid)

    if all_pg_ids:
        cur = conn.cursor()
        sql = "SELECT image_data FROM faces WHERE id = ANY(%s::int[])"
        cur.execute(sql, (all_pg_ids,))
        _ = cur.fetchall()
        cur.close()
        conn.commit()


# =============================================================================
# Explicit Benchmark Runners (Cleanup separated from Logic)
# =============================================================================

def run_s1_ingest_pg(image_paths, runs):
    conn, pg_pid = connect_and_get_pid()
    py_pid = os.getpid()

    try:
        # Warmup
        warmup = get_image_paths(8)
        truncate_pg_table(conn)
        s1_ingest_pg(conn, warmup)

        results = []
        for _ in range(runs):
            truncate_pg_table(conn)
            elapsed, stats = ResourceMonitor.measure(py_pid, pg_pid,
                                                     lambda: s1_ingest_pg(conn, image_paths))
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        return results
    finally:
        conn.close()


def run_s1_ingest_qdrant(image_paths, runs, embed_client):
    client = create_qdrant_client()
    py_pid = os.getpid()

    try:
        warmup = get_image_paths(8)
        s1_ingest_qdrant(client, warmup, embed_client)

        results = []
        for _ in range(runs):
            elapsed, stats = ResourceMonitor.measure(py_pid, None,
                                                     lambda: s1_ingest_qdrant(client, image_paths, embed_client))
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        return results
    finally:
        client.close()


def run_s1_ingest_chroma(image_paths, runs, embed_client):
    client, path = create_chroma_client()
    py_pid = os.getpid()

    try:
        warmup = get_image_paths(8)

        try:
            client.delete_collection("faces")
        except:
            pass
        col = client.create_collection("faces", configuration={"hnsw": {"space": "cosine"}})
        s1_ingest_chroma(col, warmup, embed_client)

        results = []
        for _ in range(runs):
            try:
                client.delete_collection("faces")
            except:
                pass
            col = client.create_collection("faces", configuration={"hnsw": {"space": "cosine"}})

            elapsed, stats = ResourceMonitor.measure(py_pid, None,
                                                     lambda: s1_ingest_chroma(col, image_paths, embed_client))
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        return results
    finally:
        cleanup_chroma(client, path)


def run_s2_ingest_pg(image_paths, runs):
    conn, pg_pid = connect_and_get_pid()
    py_pid = os.getpid()

    try:
        warmup = get_image_paths(8)
        truncate_pg_table(conn)
        s2_ingest_pg(conn, warmup)

        results = []
        for _ in range(runs):
            truncate_pg_table(conn)
            elapsed, stats = ResourceMonitor.measure(py_pid, pg_pid,
                                                     lambda: s2_ingest_pg(conn, image_paths))
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        return results
    finally:
        conn.close()


def run_s2_ingest_qdrant(image_paths, runs, embed_client):
    client = create_qdrant_client()
    conn, pg_pid = connect_and_get_pid()
    py_pid = os.getpid()

    try:
        warmup = get_image_paths(8)
        truncate_pg_table(conn)
        s2_ingest_qdrant(conn, client, warmup, embed_client)

        results = []
        for _ in range(runs):
            truncate_pg_table(conn)
            elapsed, stats = ResourceMonitor.measure(py_pid, pg_pid,
                                                     lambda: s2_ingest_qdrant(conn, client, image_paths, embed_client))
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        return results
    finally:
        client.close()
        conn.close()


def run_s2_ingest_chroma(image_paths, runs, embed_client):
    client, path = create_chroma_client()
    conn, pg_pid = connect_and_get_pid()
    py_pid = os.getpid()

    try:
        warmup = get_image_paths(8)
        truncate_pg_table(conn)

        try:
            client.delete_collection("faces")
        except:
            pass
        col = client.create_collection("faces", configuration={"hnsw": {"space": "cosine"}})
        s2_ingest_chroma(conn, col, warmup, embed_client)

        results = []
        for _ in range(runs):
            truncate_pg_table(conn)
            try:
                client.delete_collection("faces")
            except:
                pass
            col = client.create_collection("faces", configuration={"hnsw": {"space": "cosine"}})

            elapsed, stats = ResourceMonitor.measure(py_pid, pg_pid,
                                                     lambda: s2_ingest_chroma(conn, col, image_paths, embed_client))
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        return results
    finally:
        conn.close()
        cleanup_chroma(client, path)


# --- Serving Runners ---

def run_s1_serve_pg(queries, max_img, runs):
    # Setup data (once)
    conn, _ = connect_and_get_pid()
    setup_pg_schema(conn)  # Ensure table exists
    truncate_pg_table(conn)
    s1_ingest_pg(conn, max_img)
    conn.close()

    # Serve loop
    serve_conn, pg_pid = connect_and_get_pid()
    py_pid = os.getpid()

    try:
        # Warmup
        warmup = get_image_paths(8)
        serve_s1_pg(serve_conn, warmup)

        results = []
        for _ in range(runs):
            elapsed, stats = ResourceMonitor.measure(py_pid, pg_pid,
                                                     lambda: serve_s1_pg(serve_conn, queries))
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        return results
    finally:
        serve_conn.close()


def run_s1_serve_qdrant(queries, max_img, runs, embed_client):
    client = create_qdrant_client()
    s1_ingest_qdrant(client, max_img, embed_client)
    py_pid = os.getpid()

    try:
        warmup = get_image_paths(8)
        serve_s1_qdrant(client, embed_client, warmup)

        results = []
        for _ in range(runs):
            elapsed, stats = ResourceMonitor.measure(py_pid, None,
                                                     lambda: serve_s1_qdrant(client, embed_client, queries))
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        return results
    finally:
        client.close()


def run_s1_serve_chroma(queries, max_img, runs, embed_client):
    client, path = create_chroma_client()
    try:
        client.delete_collection("faces")
    except:
        pass
    col = client.create_collection("faces", configuration={"hnsw": {"space": "cosine"}})
    s1_ingest_chroma(col, max_img, embed_client)
    py_pid = os.getpid()

    try:
        warmup = get_image_paths(8)
        serve_s1_chroma(col, embed_client, warmup)

        results = []
        for _ in range(runs):
            elapsed, stats = ResourceMonitor.measure(py_pid, None,
                                                     lambda: serve_s1_chroma(col, embed_client, queries))
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        return results
    finally:
        cleanup_chroma(client, path)


def run_s2_serve_pg(queries, max_img, runs):
    # Setup data
    conn, _ = connect_and_get_pid()
    setup_pg_schema(conn)
    truncate_pg_table(conn)
    s2_ingest_pg(conn, max_img)
    conn.close()

    # Serve loop
    serve_conn, pg_pid = connect_and_get_pid()
    py_pid = os.getpid()

    try:
        warmup = get_image_paths(8)
        serve_s2_pg(serve_conn, warmup)

        results = []
        for _ in range(runs):
            elapsed, stats = ResourceMonitor.measure(py_pid, pg_pid,
                                                     lambda: serve_s2_pg(serve_conn, queries))
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        return results
    finally:
        serve_conn.close()


def run_s2_serve_qdrant(queries, max_img, runs, embed_client):
    # Setup data (Shared PG + Qdrant)
    pg_data_conn = connect_pg()
    setup_pg_schema(pg_data_conn)
    truncate_pg_table(pg_data_conn)
    client = create_qdrant_client()
    s2_ingest_qdrant(pg_data_conn, client, max_img, embed_client)
    pg_data_conn.close()

    # Serve loop
    serve_conn, pg_pid = connect_and_get_pid()
    py_pid = os.getpid()

    try:
        warmup = get_image_paths(8)
        serve_s2_qdrant(client, serve_conn, embed_client, warmup)

        results = []
        for _ in range(runs):
            elapsed, stats = ResourceMonitor.measure(py_pid, pg_pid,
                                                     lambda: serve_s2_qdrant(client, serve_conn, embed_client, queries))
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        return results
    finally:
        serve_conn.close()
        client.close()


def run_s2_serve_chroma(queries, max_img, runs, embed_client):
    # Setup data (Shared PG + Chroma)
    pg_data_conn = connect_pg()
    setup_pg_schema(pg_data_conn)
    truncate_pg_table(pg_data_conn)
    client, path = create_chroma_client()

    try:
        client.delete_collection("faces")
    except:
        pass
    col = client.create_collection("faces", configuration={"hnsw": {"space": "cosine"}})
    s2_ingest_chroma(pg_data_conn, col, max_img, embed_client)
    pg_data_conn.close()

    # Serve loop
    serve_conn, pg_pid = connect_and_get_pid()
    py_pid = os.getpid()

    try:
        warmup = get_image_paths(8)
        serve_s2_chroma(col, serve_conn, embed_client, warmup)

        results = []
        for _ in range(runs):
            elapsed, stats = ResourceMonitor.measure(py_pid, pg_pid,
                                                     lambda: serve_s2_chroma(col, serve_conn, embed_client, queries))
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        return results
    finally:
        serve_conn.close()
        cleanup_chroma(client, path)


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
    setup_pg_schema(pg_conn)  # Initial table creation (DDL)
    pg_conn.close()

    print("Loading model...", flush=True)
    EmbedAnythingImageClient._get_model()
    embed_client = EmbedAnythingImageClient()

    print_detailed_header()

    # --- Scenario 1 Ingestion ---
    print("\n=== SCENARIO 1: Return Paths (No Blobs in DB) ===\n")
    print("--- Ingestion (S1) ---")
    s1_ing_results = []
    for size in INGESTION_SET_SIZES:
        print(f"Size: {size}")
        paths = get_image_paths(size)

        m_pg = run_s1_ingest_pg(paths, RUNS_INGESTION)
        print_result("PG Unified", m_pg)

        m_qd = run_s1_ingest_qdrant(paths, RUNS_INGESTION, embed_client)
        print_result("Qdrant", m_qd)

        m_ch = run_s1_ingest_chroma(paths, RUNS_INGESTION, embed_client)
        print_result("Chroma", m_ch)

        s1_ing_results.append({
            'size': size,
            'pg_unified': compute_metrics(size, m_pg),
            'dist_qdrant': compute_metrics(size, m_qd),
            'dist_chroma': compute_metrics(size, m_ch)
        })

    # --- Scenario 1 Serving ---
    print("\n--- Serving (S1) ---")
    s1_srv_results = []
    max_img = get_image_paths(max(SERVING_TEST_SIZES))

    for size in SERVING_TEST_SIZES:
        print(f"Size: {size}")
        queries = get_image_paths(size)

        m_pg = run_s1_serve_pg(queries, max_img, RUNS_SERVING)
        print_result("PG Unified", m_pg)

        m_qd = run_s1_serve_qdrant(queries, max_img, RUNS_SERVING, embed_client)
        print_result("Qdrant", m_qd)

        m_ch = run_s1_serve_chroma(queries, max_img, RUNS_SERVING, embed_client)
        print_result("Chroma", m_ch)

        s1_srv_results.append({
            'size': size,
            'pg_unified': compute_metrics(size, m_pg),
            'dist_qdrant': compute_metrics(size, m_qd),
            'dist_chroma': compute_metrics(size, m_ch)
        })

    # --- Scenario 2 Ingestion ---
    print("\n\n=== SCENARIO 2: Return Blobs (Images in PG) ===\n")
    print("--- Ingestion (S2) ---")
    s2_ing_results = []
    for size in INGESTION_SET_SIZES:
        print(f"Size: {size}")
        paths = get_image_paths(size)

        m_pg = run_s2_ingest_pg(paths, RUNS_INGESTION)
        print_result("PG Unified", m_pg)

        m_qd = run_s2_ingest_qdrant(paths, RUNS_INGESTION, embed_client)
        print_result("Qdrant", m_qd)

        m_ch = run_s2_ingest_chroma(paths, RUNS_INGESTION, embed_client)
        print_result("Chroma", m_ch)

        s2_ing_results.append({
            'size': size,
            'pg_unified': compute_metrics(size, m_pg),
            'dist_qdrant': compute_metrics(size, m_qd),
            'dist_chroma': compute_metrics(size, m_ch)
        })

    # --- Scenario 2 Serving ---
    print("\n--- Serving (S2) ---")
    s2_srv_results = []

    for size in SERVING_TEST_SIZES:
        print(f"Size: {size}")
        queries = get_image_paths(size)

        m_pg = run_s2_serve_pg(queries, max_img, RUNS_SERVING)
        print_result("PG Unified", m_pg)

        m_qd = run_s2_serve_qdrant(queries, max_img, RUNS_SERVING, embed_client)
        print_result("Qdrant", m_qd)

        m_ch = run_s2_serve_chroma(queries, max_img, RUNS_SERVING, embed_client)
        print_result("Chroma", m_ch)

        s2_srv_results.append({
            'size': size,
            'pg_unified': compute_metrics(size, m_pg),
            'dist_qdrant': compute_metrics(size, m_qd),
            'dist_chroma': compute_metrics(size, m_ch)
        })

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    methods = ['pg_unified', 'dist_qdrant', 'dist_chroma']

    save_results_csv(s1_ing_results, OUTPUT_DIR / "scenario1_ingestion", timestamp, methods)
    generate_plots(s1_ing_results, OUTPUT_DIR / "scenario1_ingestion", timestamp, methods)

    save_results_csv(s1_srv_results, OUTPUT_DIR / "scenario1_serving", timestamp, methods)
    generate_plots(s1_srv_results, OUTPUT_DIR / "scenario1_serving", timestamp, methods)

    save_results_csv(s2_ing_results, OUTPUT_DIR / "scenario2_ingestion", timestamp, methods)
    generate_plots(s2_ing_results, OUTPUT_DIR / "scenario2_ingestion", timestamp, methods)

    save_results_csv(s2_srv_results, OUTPUT_DIR / "scenario2_serving", timestamp, methods)
    generate_plots(s2_srv_results, OUTPUT_DIR / "scenario2_serving", timestamp, methods)


if __name__ == "__main__":
    main()
