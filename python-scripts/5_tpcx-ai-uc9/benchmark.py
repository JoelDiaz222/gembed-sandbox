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

INGEST_PER_PERSON = 64
QUERY_PER_PERSON = 16

INGESTION_SET_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
SERVING_TEST_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048]

RUNS_INGESTION = 5
RUNS_SERVING = 5

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


def get_image_paths(n_people: int, n_images_per_person: int, offset: int = 0) -> List[str]:
    """
    Collects images from people.
    """
    # Safe limit: 97 people have >= 80 images with the dataset generated.
    REQUIRED_TOTAL = INGEST_PER_PERSON + QUERY_PER_PERSON
    all_people_dirs = sorted([
        d for d in DATA_DIR.iterdir()
        if d.is_dir() and len(list(d.glob('*'))) >= REQUIRED_TOTAL
    ])

    if len(all_people_dirs) < n_people:
        mult = (n_people // len(all_people_dirs)) + 1
        all_people_dirs = all_people_dirs * mult

    selected_people = all_people_dirs[:n_people]
    all_paths = []

    for person_dir in selected_people:
        person_imgs = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            person_imgs.extend(sorted(list(person_dir.glob(ext))))

        all_paths.extend([str(p.absolute()) for p in person_imgs[offset: offset + n_images_per_person]])

    return all_paths


def get_person_name(path: str) -> str:
    return Path(path).parent.name


def setup_pg_schema(conn):
    cur = conn.cursor()
    cur.execute('''
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE EXTENSION IF NOT EXISTS pg_gembed;

                DROP TABLE IF EXISTS faces;
                DROP TABLE IF EXISTS persons;

                CREATE TABLE persons
                (
                    id   SERIAL PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE
                );

                CREATE TABLE faces
                (
                    id         SERIAL PRIMARY KEY,
                    person_id  INTEGER REFERENCES persons (id),
                    path       TEXT NOT NULL,
                    image_data BYTEA,
                    embedding  vector(512)
                );
                ''')
    conn.commit()
    cur.close()


def truncate_pg_table(conn):
    cur = conn.cursor()
    cur.execute("TRUNCATE faces, persons RESTART IDENTITY")
    conn.commit()
    cur.close()


# =============================================================================
# Core Logic Functions
# =============================================================================

# --- Scenario 1: Paths Only ---

def s1_populate_pg(conn, image_paths):
    cur = conn.cursor()
    batch_names = [get_person_name(p) for p in image_paths]
    batch_images = []
    for p in image_paths:
        with open(p, "rb") as f:
            batch_images.append(f.read())

    # Ensure persons exist
    unique_names = list(set(batch_names))
    execute_values(cur, "INSERT INTO persons (name) VALUES %s ON CONFLICT (name) DO NOTHING",
                   [(n,) for n in unique_names])

    sql = '''
          INSERT INTO faces (path, person_id, embedding)
          SELECT t.p, p.id, t.e
          FROM unnest(%s::text[],
                      %s::text[],
                      embed_images('embed_anything', %s, %s::bytea[])) AS t(p, n, e)
                   JOIN persons p ON t.n = p.name
          '''
    cur.execute(sql, (image_paths, batch_names, MODEL_NAME, batch_images))
    conn.commit()
    cur.close()


def s1_ingest_pg(conn, image_paths):
    s1_populate_pg(conn, image_paths)


def s1_ingest_pg_indexed(conn, image_paths):
    cur = conn.cursor()
    cur.execute("CREATE INDEX ON faces USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=100);")
    cur.close()
    s1_populate_pg(conn, image_paths)


def s1_ingest_pg_deferred(conn, image_paths):
    s1_populate_pg(conn, image_paths)
    cur = conn.cursor()
    cur.execute("CREATE INDEX ON faces USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=100);")
    conn.commit()
    cur.close()


def s1_ingest_qdrant(client, image_paths, embed_client, deferred: bool = False):
    if client.collection_exists("faces"):
        client.delete_collection("faces")

    hnsw_config = models.HnswConfigDiff(m=16, ef_construct=100)
    client.create_collection(
        collection_name="faces",
        vectors_config=VectorParams(size=512, distance=Distance.COSINE, hnsw_config=hnsw_config)
    )

    if deferred:
        client.update_collection("faces", optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0))

    embeddings = embed_client.embed_files(image_paths)
    points = [
        PointStruct(
            id=idx,
            vector=emb,
            payload={"path": p, "person_name": get_person_name(p)}
        )
        for idx, (emb, p) in enumerate(zip(embeddings, image_paths))
    ]
    if points:
        client.upsert("faces", points, wait=True)

    if deferred:
        client.update_collection("faces", optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000))


def s1_ingest_chroma(collection, image_paths, embed_client):
    embeddings = embed_client.embed_files(image_paths)
    ids = [str(idx) for idx in range(len(image_paths))]
    metas = [{"path": p, "person_name": get_person_name(p)} for p in image_paths]
    if embeddings:
        collection.add(ids=ids, embeddings=embeddings, metadatas=metas)


# --- Scenario 2: Blobs in PG ---

def s2_populate_pg(conn, image_paths):
    cur = conn.cursor()
    batch_names = [get_person_name(p) for p in image_paths]
    batch_images = []
    for p in image_paths:
        with open(p, "rb") as f:
            batch_images.append(f.read())

    # Ensure persons exist
    unique_names = list(set(batch_names))
    execute_values(cur, "INSERT INTO persons (name) VALUES %s ON CONFLICT (name) DO NOTHING",
                   [(n,) for n in unique_names])

    sql = '''
          INSERT INTO faces (path, person_id, image_data, embedding)
          SELECT t.p, p.id, t.i, t.e
          FROM unnest(%s::text[],
                      %s::text[],
                      %s::bytea[],
                      embed_images('embed_anything', %s, %s::bytea[])) AS t(p, n, i, e)
                   JOIN persons p ON t.n = p.name
          '''
    cur.execute(sql, (image_paths, batch_names, batch_images, MODEL_NAME, batch_images))
    conn.commit()
    cur.close()


def s2_ingest_pg_indexed(conn, image_paths):
    cur = conn.cursor()
    cur.execute("CREATE INDEX ON faces USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=100);")
    cur.close()
    s2_populate_pg(conn, image_paths)


def s2_ingest_pg_deferred(conn, image_paths):
    s2_populate_pg(conn, image_paths)
    cur = conn.cursor()
    cur.execute("CREATE INDEX ON faces USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=100);")
    conn.commit()
    cur.close()


def s2_ingest_dist_common(conn, image_paths):
    cur = conn.cursor()
    batch_names = [get_person_name(p) for p in image_paths]
    batch_images = []
    for p in image_paths:
        with open(p, "rb") as f:
            batch_images.append(f.read())

    # Ensure persons exist
    unique_names = list(set(batch_names))
    execute_values(cur, "INSERT INTO persons (name) VALUES %s ON CONFLICT (name) DO NOTHING",
                   [(n,) for n in unique_names])

    # Get name to id mapping
    cur.execute("SELECT name, id FROM persons WHERE name = ANY(%s)", (unique_names,))
    name_to_id = dict(cur.fetchall())
    person_ids = [name_to_id[n] for n in batch_names]

    execute_values(
        cur,
        "INSERT INTO faces (path, person_id, image_data) VALUES %s RETURNING id",
        list(zip(image_paths, person_ids, batch_images))
    )
    ids = [r[0] for r in cur.fetchall()]
    conn.commit()
    cur.close()
    return ids


def s2_ingest_qdrant(conn, client, image_paths, embed_client, deferred: bool = False):
    pg_ids = s2_ingest_dist_common(conn, image_paths)
    if client.collection_exists("faces"):
        client.delete_collection("faces")
    hnsw_config = models.HnswConfigDiff(m=16, ef_construct=100)
    client.create_collection(
        collection_name="faces",
        vectors_config=VectorParams(size=512, distance=Distance.COSINE, hnsw_config=hnsw_config)
    )

    if deferred:
        client.update_collection("faces", optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0))

    embeddings = embed_client.embed_files(image_paths)
    points = [
        PointStruct(
            id=pg_id,
            vector=emb,
            payload={"pg_id": pg_id}
        )
        for pg_id, emb in zip(pg_ids, embeddings)
    ]
    if points:
        client.upsert("faces", points, wait=True)

    if deferred:
        client.update_collection("faces", optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000))


def s2_ingest_chroma(conn, collection, image_paths, embed_client):
    pg_ids = s2_ingest_dist_common(conn, image_paths)
    embeddings = embed_client.embed_files(image_paths)
    ids = [str(pg_id) for pg_id in pg_ids]
    metas = [{"pg_id": pg_id} for pg_id in pg_ids]
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
    """Print detailed benchmark results header."""
    lbl_w, time_w, col_w = 20, 14, 13
    print("\nBenchmark Results:", flush=True)
    header = (
            "  " +
            f"{'':{lbl_w}} | {'Time (s)':>{time_w}} | "
            f"{'Py Δ MB':>{col_w}} | {'Py Peak MB':>{col_w}} | {'Py CPU%':>{col_w}} | "
            f"{'PG Peak MB':>{col_w}} | {'PG CPU%':>{col_w}} | "
            f"{'QD Peak MB':>{col_w}} | {'QD CPU%':>{col_w}} | "
            f"{'Sys CPU%':>{col_w}}"
    )
    print(header, flush=True)
    print("=" * len(header), flush=True)


def print_result(label: str, results: List[BenchmarkResult]):
    times = [r.time_s for r in results]
    py_deltas = [r.stats.py_delta_mb for r in results]
    py_peaks = [r.stats.py_peak_mb for r in results]
    py_cpus = [r.stats.py_cpu for r in results]
    pg_peaks = [r.stats.pg_peak_mb for r in results]
    pg_cpus = [r.stats.pg_cpu for r in results]
    qd_peaks = [r.stats.qd_peak_mb for r in results]
    qd_cpus = [r.stats.qd_cpu for r in results]
    sys_cpus = [r.stats.sys_cpu for r in results]

    def fmt(vals, p=1):
        return f"{mean(vals):.{p}f}±{safe_stdev(vals):.{p}f}"

    row_fmt = (
        "  {label:<20} | {time:>14} | "
        "{pyd:>13} | {pyp:>13} | {pyc:>13} | "
        "{pgp:>13} | {pgc:>13} | "
        "{qdp:>13} | {qdc:>13} | "
        "{sysc:>13}"
    )
    print(row_fmt.format(
        label=label,
        time=fmt(times, 3),
        pyd=fmt(py_deltas), pyp=fmt(py_peaks), pyc=fmt(py_cpus),
        pgp=fmt(pg_peaks), pgc=fmt(pg_cpus),
        qdp=fmt(qd_peaks), qdc=fmt(qd_cpus),
        sysc=fmt(sys_cpus)
    ), flush=True)


def main():
    py_pid = os.getpid()
    print("Loading model...", flush=True)
    EmbedAnythingImageClient._get_model()
    embed_client = EmbedAnythingImageClient()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    methods = ['pg_indexed', 'pg_deferred', 'qd_indexed', 'qd_deferred', 'chroma']

    for scenario_idx in [1, 2]:
        scenario_name = f"Scenario {scenario_idx}: {'Paths' if scenario_idx == 1 else 'Blobs'}"
        print(f"\n{'=' * 120}\n{scenario_name.upper()}\n{'=' * 120}")

        # --- Phase 1: Ingestion ---
        print("\n--- Phase 1: Ingestion ---")
        ing_results = []
        final_chroma_path = None

        for size in INGESTION_SET_SIZES:
            print(f"Ingestion Size: {size}")
            n_people = max(1, size // INGEST_PER_PERSON)
            imgs_per = INGEST_PER_PERSON if size >= INGEST_PER_PERSON else size
            paths = get_image_paths(n_people, imgs_per)
            res_pg_idx, res_pg_def, res_qd_idx, res_qd_def, res_ch = [], [], [], [], []

            for run in range(RUNS_INGESTION):
                is_last_run = (run == RUNS_INGESTION - 1) and (size == INGESTION_SET_SIZES[-1])

                # PG Indexed
                conn, pg_pid = connect_and_get_pid()
                setup_pg_schema(conn)
                truncate_pg_table(conn)
                fn = s1_ingest_pg_indexed if scenario_idx == 1 else s2_ingest_pg_indexed
                elapsed, stats = ResourceMonitor.measure(py_pid, pg_pid, lambda: fn(conn, paths))
                res_pg_idx.append(BenchmarkResult(elapsed, stats))
                conn.close()

                # PG Deferred
                conn, pg_pid = connect_and_get_pid()
                setup_pg_schema(conn)
                truncate_pg_table(conn)
                fn = s1_ingest_pg_deferred if scenario_idx == 1 else s2_ingest_pg_deferred
                elapsed, stats = ResourceMonitor.measure(py_pid, pg_pid, lambda: fn(conn, paths))
                res_pg_def.append(BenchmarkResult(elapsed, stats))
                if not is_last_run:
                    conn.close()
                else:
                    pg_persistent_conn = conn

                # Qdrant Indexed
                qd_client = create_qdrant_client()
                if scenario_idx == 1:
                    elapsed, stats = ResourceMonitor.measure(py_pid, None,
                                                             lambda: s1_ingest_qdrant(qd_client, paths, embed_client,
                                                                                      deferred=False))
                else:
                    pg_tmp = connect_pg();
                    setup_pg_schema(pg_tmp)
                    elapsed, stats = ResourceMonitor.measure(py_pid, get_pg_pid(pg_tmp),
                                                             lambda: s2_ingest_qdrant(pg_tmp, qd_client, paths,
                                                                                      embed_client, deferred=False))
                    pg_tmp.close()
                res_qd_idx.append(BenchmarkResult(elapsed, stats))
                qd_client.close()

                # Qdrant Deferred
                qd_client = create_qdrant_client()
                if scenario_idx == 1:
                    elapsed, stats = ResourceMonitor.measure(py_pid, None,
                                                             lambda: s1_ingest_qdrant(qd_client, paths, embed_client,
                                                                                      deferred=True))
                else:
                    pg_tmp = connect_pg();
                    setup_pg_schema(pg_tmp)
                    elapsed, stats = ResourceMonitor.measure(py_pid, get_pg_pid(pg_tmp),
                                                             lambda: s2_ingest_qdrant(pg_tmp, qd_client, paths,
                                                                                      embed_client, deferred=True))
                    pg_tmp.close()
                res_qd_def.append(BenchmarkResult(elapsed, stats))
                if not is_last_run:
                    qd_client.close()
                else:
                    qd_persistent_client = qd_client

                # Chroma
                c_client, c_path = create_chroma_client()
                col = c_client.create_collection("faces", configuration={"hnsw": {"space": "cosine"}})
                if scenario_idx == 1:
                    elapsed, stats = ResourceMonitor.measure(py_pid, None,
                                                             lambda: s1_ingest_chroma(col, paths, embed_client))
                else:
                    pg_tmp = connect_pg();
                    setup_pg_schema(pg_tmp)
                    elapsed, stats = ResourceMonitor.measure(py_pid, get_pg_pid(pg_tmp),
                                                             lambda: s2_ingest_chroma(pg_tmp, col, paths, embed_client))
                    pg_tmp.close()
                res_ch.append(BenchmarkResult(elapsed, stats))
                if is_last_run:
                    final_chroma_path = c_path;
                    del c_client
                else:
                    cleanup_chroma(c_client, c_path)

            print_detailed_header()
            print_result("PG Indexed", res_pg_idx);
            print_result("PG Deferred", res_pg_def)
            print_result("QD Indexed", res_qd_idx);
            print_result("QD Deferred", res_qd_def);
            print_result("Chroma", res_ch)

            ing_results.append({
                'size': size,
                'pg_indexed': compute_metrics(size, res_pg_idx), 'pg_deferred': compute_metrics(size, res_pg_def),
                'qd_indexed': compute_metrics(size, res_qd_idx), 'qd_deferred': compute_metrics(size, res_qd_def),
                'chroma': compute_metrics(size, res_ch)
            })

        save_results_csv(ing_results, OUTPUT_DIR / f"scenario{scenario_idx}_ingestion", timestamp, methods)
        generate_plots(ing_results, OUTPUT_DIR / f"scenario{scenario_idx}_ingestion", timestamp, methods)

        # --- Phase 2: Serving ---
        print("\n--- Phase 2: Serving (against max ingested set) ---")
        srv_results = []
        c_persistent_client = chromadb.PersistentClient(path=final_chroma_path)
        c_persistent_col = c_persistent_client.get_collection("faces")

        for size in SERVING_TEST_SIZES:
            print(f"Query Batch Size: {size}")
            n_people_max = INGESTION_SET_SIZES[-1] // INGEST_PER_PERSON
            n_people = max(1, size // QUERY_PER_PERSON);
            imgs_per = QUERY_PER_PERSON if size >= QUERY_PER_PERSON else size
            n_people = min(n_people, n_people_max)
            queries = get_image_paths(n_people, imgs_per, offset=INGEST_PER_PERSON)[:size]

            m_pg, m_qd, m_ch = [], [], []
            for _ in range(RUNS_SERVING):
                # PG (uses PG Deferred persistent data)
                serve_fn = serve_s1_pg if scenario_idx == 1 else serve_s2_pg
                elapsed, stats = ResourceMonitor.measure(py_pid, get_pg_pid(pg_persistent_conn),
                                                         lambda: serve_fn(pg_persistent_conn, queries))
                m_pg.append(BenchmarkResult(elapsed, stats))

                # Qdrant (uses QD Deferred persistent data)
                if scenario_idx == 1:
                    elapsed, stats = ResourceMonitor.measure(py_pid, None,
                                                             lambda: serve_s1_qdrant(qd_persistent_client, embed_client,
                                                                                     queries))
                else:
                    pg_tmp = connect_pg()
                    elapsed, stats = ResourceMonitor.measure(py_pid, get_pg_pid(pg_tmp),
                                                             lambda: serve_s2_qdrant(qd_persistent_client, pg_tmp,
                                                                                     embed_client, queries))
                    pg_tmp.close()
                m_qd.append(BenchmarkResult(elapsed, stats))

                # Chroma
                if scenario_idx == 1:
                    elapsed, stats = ResourceMonitor.measure(py_pid, None,
                                                             lambda: serve_s1_chroma(c_persistent_col, embed_client,
                                                                                     queries))
                else:
                    pg_tmp = connect_pg()
                    elapsed, stats = ResourceMonitor.measure(py_pid, get_pg_pid(pg_tmp),
                                                             lambda: serve_s2_chroma(c_persistent_col, pg_tmp,
                                                                                     embed_client, queries))
                    pg_tmp.close()
                m_ch.append(BenchmarkResult(elapsed, stats))

            print_detailed_header()
            print_result("PG Unified", m_pg);
            print_result("Qdrant", m_qd);
            print_result("Chroma", m_ch)
            srv_results.append(
                {'size': size, 'pg_unified': compute_metrics(size, m_pg), 'dist_qdrant': compute_metrics(size, m_qd),
                 'dist_chroma': compute_metrics(size, m_ch)})

        serving_methods = ['pg_unified', 'dist_qdrant', 'dist_chroma']
        save_results_csv(srv_results, OUTPUT_DIR / f"scenario{scenario_idx}_serving", timestamp, serving_methods)
        generate_plots(srv_results, OUTPUT_DIR / f"scenario{scenario_idx}_serving", timestamp, serving_methods)
        pg_persistent_conn.close();
        qd_persistent_client.close();
        cleanup_chroma(c_persistent_client, final_chroma_path)


if __name__ == "__main__":
    main()
