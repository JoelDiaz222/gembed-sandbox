import gc
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import List

import chromadb
import embed_anything
from benchmark_utils import (
    DB_CONFIG, QDRANT_URL, QDRANT_CONTAINER_NAME,
    BenchmarkResult, ResourceMonitor,
    safe_stdev, calc_iqr, compute_metrics,
    connect_pg, get_pg_pid, connect_and_get_pid, warmup_pg_connection,
)
from embed_anything import EmbeddingModel
from plot_utils import save_results_csv, generate_plots
from psycopg2.extras import execute_values
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct

MODEL_NAME = "openai/clip-vit-base-patch32"

INGEST_PER_PERSON = 64
QUERY_PER_PERSON = 16

INGESTION_SET_SIZES = [16, 32, 64, 128, 256, 512, 1024]
SERVING_TEST_SIZES = [16, 32, 64, 128, 256, 512, 1024]

RUNS_INGESTION = 15
RUNS_SERVING = 15

OUTPUT_DIR = Path(__file__).parent / "output"
DATA_DIR = Path(__file__).parent.parent / "data" / "CUSTOMER_IMAGES"

model_cache = {}


# =============================================================================
# Client & Helpers
# =============================================================================

class EmbedAnythingImageClient:
    def embed_files(self, paths: List[str]) -> List[List[float]]:
        """Generate image embeddings."""
        model = self._get_model()

        base_temp = Path(f"temp_bench_imgs")
        if base_temp.exists():
            shutil.rmtree(base_temp)
        base_temp.mkdir()

        try:
            for idx, p in enumerate(paths):
                ext = Path(p).suffix
                fname = f"{idx:06d}{ext}"
                shutil.copy(p, base_temp / fname)

            res = embed_anything.embed_image_directory(str(base_temp), embedder=model)

            all_embeddings = []
            if isinstance(res, list):
                for item in res:
                    if hasattr(item, 'embedding'):
                        all_embeddings.append(item.embedding)
        finally:
            if base_temp.exists():
                shutil.rmtree(base_temp)

        return all_embeddings

    @staticmethod
    def _get_model():
        if MODEL_NAME not in model_cache:
            model_cache[MODEL_NAME] = EmbeddingModel.from_pretrained_hf(MODEL_NAME)
        return model_cache[MODEL_NAME]


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


def s1_ingest_qdrant(client, image_paths, embed_client):
    if client.collection_exists("faces"):
        client.delete_collection("faces")

    hnsw_config = models.HnswConfigDiff(m=16, ef_construct=100)
    client.create_collection(
        collection_name="faces",
        vectors_config=VectorParams(size=512, distance=Distance.COSINE, hnsw_config=hnsw_config)
    )

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

    results = execute_values(
        cur,
        "INSERT INTO faces (path, person_id, image_data) VALUES %s RETURNING id",
        list(zip(image_paths, person_ids, batch_images)),
        fetch=True
    )
    all_ids = [r[0] for r in results]

    conn.commit()
    cur.close()
    return all_ids


def s2_ingest_qdrant(conn, client, image_paths, embed_client):
    pg_ids = s2_ingest_dist_common(conn, image_paths)
    if client.collection_exists("faces"):
        client.delete_collection("faces")
    hnsw_config = models.HnswConfigDiff(m=16, ef_construct=100)
    client.create_collection(
        collection_name="faces",
        vectors_config=VectorParams(size=512, distance=Distance.COSINE, hnsw_config=hnsw_config)
    )

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
    cur = conn.cursor()

    images_data = [open(p, "rb").read() for p in query_paths]
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
    cur = conn.cursor()

    images_data = [open(p, "rb").read() for p in query_paths]
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


def serve_s2_qdrant(client, conn, embed_client, query_paths):
    embeddings = embed_client.embed_files(query_paths)
    requests = [
        models.QueryRequest(query=emb, limit=TOP_K, with_payload=True)
        for emb in embeddings
    ]
    if not requests:
        return
    results = client.query_batch_points(collection_name="faces", requests=requests)
    all_pg_ids = []
    for batch_res in results:
        for point in batch_res.points:
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
    if not embeddings:
        return
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
    methods = ['pg_indexed', 'qd_indexed', 'chroma']

    for scenario_idx in [1, 2]:
        scenario_name = f"Scenario {scenario_idx}: {'Paths' if scenario_idx == 1 else 'Blobs'}"
        print(f"\n{'=' * 120}\n{scenario_name.upper()}\n{'=' * 120}")

        # --- Phase 1: Ingestion ---
        print("\n--- Phase 1: Ingestion ---")
        ing_results = []
        final_chroma_path = None

        # Warm-up Ingestion
        print(f"Warming up ingestion for Scenario {scenario_idx}...")
        warmup_paths = get_image_paths(1, 2)
        if not warmup_paths:
            print("Warning: No images found for warmup.")
        else:
            # PG Warmup
            conn = connect_pg()
            fn = s1_ingest_pg_indexed if scenario_idx == 1 else s2_ingest_pg_indexed
            setup_pg_schema(conn)
            fn(conn, warmup_paths)
            truncate_pg_table(conn)
            conn.close()

            # Qdrant Warmup
            qd_client = create_qdrant_client()
            if scenario_idx == 1:
                s1_ingest_qdrant(qd_client, warmup_paths, embed_client)
            else:
                pg_tmp = connect_pg()
                setup_pg_schema(pg_tmp)
                s2_ingest_qdrant(pg_tmp, qd_client, warmup_paths, embed_client)
                pg_tmp.close()
            if qd_client.collection_exists("faces"):
                qd_client.delete_collection("faces")
            qd_client.close()

            # Chroma Warmup
            c_client, c_path = create_chroma_client()
            col = c_client.create_collection("faces", configuration={"hnsw": {"space": "cosine"}})
            if scenario_idx == 1:
                s1_ingest_chroma(col, warmup_paths, embed_client)
            else:
                pg_tmp = connect_pg()
                setup_pg_schema(pg_tmp)
                s2_ingest_chroma(pg_tmp, col, warmup_paths, embed_client)
                pg_tmp.close()
            cleanup_chroma(c_client, c_path)

        # Initialize results storage for ingestion
        results_by_size = {
            size: {'pg_indexed': [], 'qd_indexed': [], 'chroma': []}
            for size in INGESTION_SET_SIZES
        }

        # Outer loop: runs (cyclic execution)
        for run_idx in range(RUNS_INGESTION):
            print(f"\n{'=' * 120}")
            print(f"RUN {run_idx + 1}/{RUNS_INGESTION}")
            print(f"{'=' * 120}")

            # Inner loop: sizes
            for size in INGESTION_SET_SIZES:
                print(f"\n  Size: {size}", flush=True)
                n_people = max(1, size // INGEST_PER_PERSON)
                imgs_per = INGEST_PER_PERSON if size >= INGEST_PER_PERSON else size
                paths = get_image_paths(n_people, imgs_per)

                # Clear Python model cache for this iteration
                model_cache.clear()

                # Setup connections for this iteration
                conn_idx, pid_idx = connect_and_get_pid()
                warmup_pg_connection(conn_idx)

                conn_dist = None
                pid_dist = None
                if scenario_idx == 2:
                    conn_dist, pid_dist = connect_and_get_pid()
                    warmup_pg_connection(conn_dist)

                try:
                    is_last_run = (run_idx == RUNS_INGESTION - 1) and (size == INGESTION_SET_SIZES[-1])

                    # PG Indexed
                    setup_pg_schema(conn_idx)
                    truncate_pg_table(conn_idx)
                    fn = s1_ingest_pg_indexed if scenario_idx == 1 else s2_ingest_pg_indexed
                    elapsed, _, stats = ResourceMonitor.measure(py_pid, pid_idx, lambda: fn(conn_idx, paths))
                    results_by_size[size]['pg_indexed'].append(BenchmarkResult(elapsed, stats))
                    print(f"    PG: {elapsed:.2f}s", flush=True)

                    if is_last_run:
                        pg_persistent_conn = conn_idx
                        # We keep conn_idx if it's the last run of the last size for serving phase

                    # Qdrant Indexed
                    qd_client = create_qdrant_client()
                    if scenario_idx == 1:
                        elapsed, _, stats = ResourceMonitor.measure(py_pid, None,
                                                                    lambda: s1_ingest_qdrant(qd_client, paths,
                                                                                             embed_client),
                                                                    container_name=QDRANT_CONTAINER_NAME)
                    else:
                        setup_pg_schema(conn_dist)
                        elapsed, _, stats = ResourceMonitor.measure(py_pid, pid_dist,
                                                                    lambda: s2_ingest_qdrant(conn_dist, qd_client,
                                                                                             paths, embed_client),
                                                                    container_name=QDRANT_CONTAINER_NAME)
                    results_by_size[size]['qd_indexed'].append(BenchmarkResult(elapsed, stats))
                    print(f"    Qdrant: {elapsed:.2f}s", flush=True)

                    if not is_last_run:
                        qd_client.close()
                    else:
                        qd_persistent_client = qd_client

                    # Chroma
                    c_client, c_path = create_chroma_client()
                    col = c_client.create_collection("faces", configuration={"hnsw": {"space": "cosine"}})
                    if scenario_idx == 1:
                        elapsed, _, stats = ResourceMonitor.measure(py_pid, None,
                                                                    lambda: s1_ingest_chroma(col, paths, embed_client))
                    else:
                        setup_pg_schema(conn_dist)
                        elapsed, _, stats = ResourceMonitor.measure(py_pid, pid_dist,
                                                                    lambda: s2_ingest_chroma(conn_dist, col, paths,
                                                                                             embed_client))
                    results_by_size[size]['chroma'].append(BenchmarkResult(elapsed, stats))
                    print(f"    Chroma: {elapsed:.2f}s", flush=True)

                    if is_last_run:
                        final_chroma_path = c_path
                        del c_client
                        print(f"    [Keeping Chroma DB for Serving Phase]")
                    else:
                        cleanup_chroma(c_client, c_path)
                finally:
                    # Close connections unless needed for serving
                    if not ((run_idx == RUNS_INGESTION - 1) and (size == INGESTION_SET_SIZES[-1])):
                        conn_idx.close()
                    if conn_dist:
                        conn_dist.close()

        # Print aggregated ingestion results
        print(f"\n\n{'=' * 120}")
        print(f"SCENARIO {scenario_idx} INGESTION AGGREGATED RESULTS")
        print(f"{'=' * 120}")
        print_detailed_header()

        ing_results = []
        for size in INGESTION_SET_SIZES:
            print(f"Size: {size}", flush=True)
            results = results_by_size[size]

            print_result("PG Indexed", results['pg_indexed'])
            print_result("QD Indexed", results['qd_indexed'])
            print_result("Chroma", results['chroma'])
            print()

            ing_results.append({
                'size': size,
                'pg_indexed': compute_metrics(size, results['pg_indexed']),
                'qd_indexed': compute_metrics(size, results['qd_indexed']),
                'chroma': compute_metrics(size, results['chroma'])
            })

        save_results_csv(ing_results, OUTPUT_DIR / f"scenario{scenario_idx}_ingestion", timestamp, methods)
        generate_plots(ing_results, OUTPUT_DIR / f"scenario{scenario_idx}_ingestion", timestamp, methods)

        gc.collect()
        time.sleep(2)

        # --- Phase 2: Serving ---
        print("\n--- Phase 2: Serving (against max ingested set) ---")
        c_persistent_client = chromadb.PersistentClient(path=final_chroma_path)
        c_persistent_col = c_persistent_client.get_collection("faces")

        # Warm-up Serving
        print(f"Warming up serving for Scenario {scenario_idx}...")
        warmup_queries = get_image_paths(1, 2, offset=INGEST_PER_PERSON)
        if not warmup_queries:
            print("Warning: No query images found for warmup.")
        else:
            # PG Serving Warmup
            serve_fn = serve_s1_pg if scenario_idx == 1 else serve_s2_pg
            serve_fn(pg_persistent_conn, warmup_queries)

            # Qdrant Serving Warmup
            if scenario_idx == 1:
                serve_s1_qdrant(qd_persistent_client, embed_client, warmup_queries)
            else:
                pg_tmp = connect_pg()
                serve_s2_qdrant(qd_persistent_client, pg_tmp, embed_client, warmup_queries)
                pg_tmp.close()

            # Chroma Serving Warmup
            if scenario_idx == 1:
                serve_s1_chroma(c_persistent_col, embed_client, warmup_queries)
            else:
                pg_tmp = connect_pg()
                serve_s2_chroma(c_persistent_col, pg_tmp, embed_client, warmup_queries)
                pg_tmp.close()

        # Initialize results storage for serving
        results_by_size = {
            size: {'pg_mono_store': [], 'poly_qdrant': [], 'poly_chroma': []}
            for size in SERVING_TEST_SIZES
        }

        # Pre-calculate query paths for all sizes
        n_people_max = INGESTION_SET_SIZES[-1] // INGEST_PER_PERSON
        queries_by_size = {}
        for size in SERVING_TEST_SIZES:
            n_people = max(1, size // QUERY_PER_PERSON)
            imgs_per = QUERY_PER_PERSON if size >= QUERY_PER_PERSON else size
            n_people = min(n_people, n_people_max)
            queries_by_size[size] = get_image_paths(n_people, imgs_per, offset=INGEST_PER_PERSON)[:size]

        # Outer loop: runs (cyclic execution)
        for run_idx in range(RUNS_SERVING):
            print(f"\n{'=' * 120}")
            print(f"RUN {run_idx + 1}/{RUNS_SERVING}")
            print(f"{'=' * 120}")

            # Inner loop: sizes
            for size in SERVING_TEST_SIZES:
                print(f"\n  Size: {size}", flush=True)
                queries = queries_by_size[size]

                # Clear Python model cache for each iteration
                model_cache.clear()

                # PG
                serve_fn = serve_s1_pg if scenario_idx == 1 else serve_s2_pg
                elapsed, _, stats = ResourceMonitor.measure(py_pid, get_pg_pid(pg_persistent_conn),
                                                            lambda: serve_fn(pg_persistent_conn, queries))
                results_by_size[size]['pg_mono_store'].append(BenchmarkResult(elapsed, stats))
                print(f"    PG: {elapsed:.2f}s", flush=True)

                # Qdrant
                if scenario_idx == 1:
                    elapsed, _, stats = ResourceMonitor.measure(py_pid, None,
                                                                lambda: serve_s1_qdrant(qd_persistent_client,
                                                                                        embed_client, queries),
                                                                container_name=QDRANT_CONTAINER_NAME)
                else:
                    pg_tmp = connect_pg()
                    warmup_pg_connection(pg_tmp)
                    elapsed, _, stats = ResourceMonitor.measure(py_pid, get_pg_pid(pg_tmp),
                                                                lambda: serve_s2_qdrant(qd_persistent_client, pg_tmp,
                                                                                        embed_client, queries),
                                                                container_name=QDRANT_CONTAINER_NAME)
                    pg_tmp.close()
                results_by_size[size]['poly_qdrant'].append(BenchmarkResult(elapsed, stats))
                print(f"    Qdrant: {elapsed:.2f}s", flush=True)

                # Chroma
                if scenario_idx == 1:
                    elapsed, _, stats = ResourceMonitor.measure(py_pid, None,
                                                                lambda: serve_s1_chroma(c_persistent_col, embed_client,
                                                                                        queries))
                else:
                    pg_tmp = connect_pg()
                    warmup_pg_connection(pg_tmp)
                    elapsed, _, stats = ResourceMonitor.measure(py_pid, get_pg_pid(pg_tmp),
                                                                lambda: serve_s2_chroma(c_persistent_col, pg_tmp,
                                                                                        embed_client, queries))
                    pg_tmp.close()
                results_by_size[size]['poly_chroma'].append(BenchmarkResult(elapsed, stats))
                print(f"    Chroma: {elapsed:.2f}s", flush=True)

        # Print aggregated serving results
        print(f"\n\n{'=' * 120}")
        print(f"SCENARIO {scenario_idx} SERVING AGGREGATED RESULTS")
        print(f"{'=' * 120}")
        print_detailed_header()

        srv_results = []
        for size in SERVING_TEST_SIZES:
            print(f"Size: {size}", flush=True)
            results = results_by_size[size]

            print_result("PG Mono-Store", results['pg_mono_store'])
            print_result("Qdrant", results['poly_qdrant'])
            print_result("Chroma", results['poly_chroma'])
            print()

            srv_results.append({
                'size': size,
                'pg_mono_store': compute_metrics(size, results['pg_mono_store']),
                'poly_qdrant': compute_metrics(size, results['poly_qdrant']),
                'poly_chroma': compute_metrics(size, results['poly_chroma'])
            })

        serving_methods = ['pg_mono_store', 'poly_qdrant', 'poly_chroma']
        save_results_csv(srv_results, OUTPUT_DIR / f"scenario{scenario_idx}_serving", timestamp, serving_methods)
        generate_plots(srv_results, OUTPUT_DIR / f"scenario{scenario_idx}_serving", timestamp, serving_methods)
        pg_persistent_conn.close();
        qd_persistent_client.close();
        cleanup_chroma(c_persistent_client, final_chroma_path)

        if scenario_idx == 1:
            gc.collect()
            time.sleep(2)


if __name__ == "__main__":
    main()
