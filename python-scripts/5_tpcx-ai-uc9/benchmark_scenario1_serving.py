#!/usr/bin/env python3
"""
Benchmark 5 - TPCx-AI UC9, Scenario 1: Serving (path-based ANN search)
Populate DBs with --db-size images, then query with --sizes batches.
"""
import argparse
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List

import chromadb
import embed_anything
from benchmark_utils import (
    QDRANT_URL, QDRANT_CONTAINER_NAME,
    BenchmarkResult, ResourceMonitor,
    connect_and_get_pid, get_pg_pid, warmup_pg_connection,
    temp_image_dir,
)
from embed_anything import EmbeddingModel
from plot_utils import save_single_run_csv
from psycopg2.extras import execute_values
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct

OUTPUT_DIR = Path(__file__).parent / "output"
DATA_DIR = Path(__file__).parent.parent / "data" / "CUSTOMER_IMAGES"

MODEL_NAME = "openai/clip-vit-base-patch32"
INGEST_PER_PERSON = 64
QUERY_PER_PERSON = 16
TOP_K = 5

model_cache = {}


# =============================================================================
# Image Embedding Client
# =============================================================================

class EmbedAnythingImageClient:
    def embed_files(self, paths: List[str]) -> List[List[float]]:
        model = self._get_model()
        embeddings = []
        with temp_image_dir(paths) as base_temp:
            res = embed_anything.embed_image_directory(str(base_temp), embedder=model)
            if isinstance(res, list):
                for item in res:
                    if hasattr(item, 'embedding'):
                        embeddings.append(item.embedding)
        return embeddings

    @staticmethod
    def _get_model():
        if MODEL_NAME not in model_cache:
            model_cache[MODEL_NAME] = EmbeddingModel.from_pretrained_hf(MODEL_NAME)
        return model_cache[MODEL_NAME]


# =============================================================================
# Data Helpers
# =============================================================================

def get_image_paths(n_people: int, n_images_per_person: int, offset: int = 0) -> List[str]:
    required_total = INGEST_PER_PERSON + QUERY_PER_PERSON
    all_people_dirs = sorted([
        d for d in DATA_DIR.iterdir()
        if d.is_dir() and len(list(d.glob('*'))) >= required_total
    ])
    if len(all_people_dirs) < n_people:
        mult = (n_people // len(all_people_dirs)) + 1
        all_people_dirs = all_people_dirs * mult
    selected = all_people_dirs[:n_people]
    all_paths = []
    for person_dir in selected:
        imgs = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            imgs.extend(sorted(person_dir.glob(ext)))
        all_paths.extend([str(p.absolute()) for p in imgs[offset:offset + n_images_per_person]])
    return all_paths


def get_person_name(path: str) -> str:
    return Path(path).parent.name


# =============================================================================
# PostgreSQL Functions
# =============================================================================

def setup_pg_schema(conn):
    cur = conn.cursor()
    cur.execute("""
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
                    id        SERIAL PRIMARY KEY,
                    person_id INTEGER REFERENCES persons (id),
                    path      TEXT NOT NULL,
                    embedding vector(512)
                );
                """)
    conn.commit()
    cur.close()


def s1_populate_pg(conn, image_paths: List[str]):
    cur = conn.cursor()
    batch_names = [get_person_name(p) for p in image_paths]
    unique_names = list(set(batch_names))
    execute_values(cur, "INSERT INTO persons (name) VALUES %s ON CONFLICT (name) DO NOTHING",
                   [(n,) for n in unique_names])

    with temp_image_dir(image_paths, prefix="temp_pg_ingest_setup") as base_temp:
        sql = """
              INSERT INTO faces (path, person_id, embedding)
              SELECT t.p, p.id, t.e
              FROM unnest(%s::text[], %s::text[],
                          embed_image_directory('embed_anything', %s, %s)) AS t(p, n, e)
                       JOIN persons p ON t.n = p.name
              """
        cur.execute(sql, (image_paths, batch_names, MODEL_NAME, str(base_temp.absolute())))
        conn.commit()
    cur.close()


def s1_ingest_pg_unified(conn, image_paths: List[str]):
    cur = conn.cursor()
    cur.execute("CREATE INDEX ON faces USING hnsw (embedding vector_cosine_ops)"
                " WITH (m=16, ef_construction=100);")
    cur.close()
    s1_populate_pg(conn, image_paths)


def s1_ingest_qdrant(client, image_paths: List[str], embed_client: EmbedAnythingImageClient):
    if client.collection_exists("faces"):
        client.delete_collection("faces")
    hnsw_config = models.HnswConfigDiff(m=16, ef_construct=100)
    client.create_collection("faces",
                             vectors_config=VectorParams(size=512, distance=Distance.COSINE,
                                                         hnsw_config=hnsw_config))
    embeddings = embed_client.embed_files(image_paths)
    points = [
        PointStruct(id=idx, vector=emb, payload={"path": p, "person_name": get_person_name(p)})
        for idx, (emb, p) in enumerate(zip(embeddings, image_paths))
    ]
    if points:
        client.upsert("faces", points, wait=True)


def create_chroma_client_for_ingestion(base_path: str = "./chroma_bench_uc9_s1_svc"):
    db_path = f"{base_path}_{time.time_ns()}"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    client = chromadb.PersistentClient(path=db_path)
    return client, db_path


def cleanup_chroma(client, db_path: str):
    del client
    time.sleep(0.5)
    if os.path.exists(db_path):
        shutil.rmtree(db_path)


def create_qdrant_client():
    return QdrantClient(url=QDRANT_URL)


def s1_ingest_chroma(collection, image_paths: List[str], embed_client: EmbedAnythingImageClient):
    embeddings = embed_client.embed_files(image_paths)
    ids = [str(idx) for idx in range(len(image_paths))]
    metas = [{"path": p, "person_name": get_person_name(p)} for p in image_paths]
    if embeddings:
        collection.add(ids=ids, embeddings=embeddings, metadatas=metas)


# =============================================================================
# Serving Functions
# =============================================================================

def serve_s1_pg_unified(conn, query_paths: List[str]):
    cur = conn.cursor()
    with temp_image_dir(query_paths, "s1_pg_unified_") as base_temp:
        sql = """
            WITH query_embeddings AS (
                SELECT unnest(embed_image_directory('embed_anything', %s, %s)) as embedding
            )
            SELECT q.embedding, f.path
            FROM query_embeddings q,
            LATERAL (
                SELECT path
                FROM faces
                ORDER BY faces.embedding <-> q.embedding
                LIMIT %s
            ) f;
        """
        cur.execute(sql, (MODEL_NAME, str(base_temp.absolute()), TOP_K))
        _ = cur.fetchall()

    conn.commit()
    cur.close()


def serve_s1_qdrant(client, embed_client: EmbedAnythingImageClient, query_paths: List[str]):
    embeddings = embed_client.embed_files(query_paths)
    requests = [models.QueryRequest(query=emb, limit=TOP_K, with_payload=True) for emb in embeddings]
    if requests:
        _ = client.query_batch_points(collection_name="faces", requests=requests)


def serve_s1_chroma(collection, embed_client: EmbedAnythingImageClient, query_paths: List[str]):
    embeddings = embed_client.embed_files(query_paths)
    if embeddings:
        _ = collection.query(query_embeddings=embeddings, n_results=TOP_K)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark 5 UC9 - Scenario 1 Serving')
    parser.add_argument('--sizes', type=int, nargs='+', required=True,
                        help='Query batch sizes to test')
    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--db-size', type=int, required=True,
                        help='Number of images to pre-populate in DB')
    args = parser.parse_args()

    test_sizes = args.sizes
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    db_size = args.db_size
    methods = ['pg_unified', 'poly_qdrant', 'poly_chroma']

    print(f"\nStarting Benchmark 5 UC9 - Scenario 1 Serving")
    print(f"Run ID: {run_id}")
    print(f"DB Size: {db_size}")
    print(f"Query Sizes: {test_sizes}")
    print("Loading model...", flush=True)
    EmbedAnythingImageClient._get_model()
    embed_client = EmbedAnythingImageClient()
    py_pid = os.getpid()

    # Compute ingestion paths
    n_people_db = max(1, db_size // INGEST_PER_PERSON)
    imgs_per_db = INGEST_PER_PERSON if db_size >= INGEST_PER_PERSON else db_size
    ingest_paths = get_image_paths(n_people_db, imgs_per_db)

    # Compute query paths for each size (different people or offset)
    queries_by_size = {}
    for size in test_sizes:
        n_people_q = max(1, size // QUERY_PER_PERSON)
        imgs_per_q = QUERY_PER_PERSON if size >= QUERY_PER_PERSON else size
        n_people_q = min(n_people_q, n_people_db)
        queries_by_size[size] = get_image_paths(n_people_q, imgs_per_q,
                                                offset=INGEST_PER_PERSON)[:size]

    print("Setting up DBs...")
    conn_pg, pg_pid = connect_and_get_pid()
    warmup_pg_connection(conn_pg)
    setup_pg_schema(conn_pg)
    model_cache.clear()
    s1_ingest_pg_unified(conn_pg, ingest_paths)

    qd = create_qdrant_client()
    model_cache.clear()
    s1_ingest_qdrant(qd, ingest_paths, embed_client)

    c_client, c_path = create_chroma_client_for_ingestion()
    c_col = c_client.create_collection("faces", configuration={"hnsw": {"space": "cosine"}})
    model_cache.clear()
    s1_ingest_chroma(c_col, ingest_paths, embed_client)

    # Warm-up queries
    print("Warming up queries...")
    warmup_queries = queries_by_size.get(min(test_sizes), [])[:4]
    if warmup_queries:
        model_cache.clear()
        serve_s1_pg_unified(conn_pg, warmup_queries)
        model_cache.clear()
        serve_s1_qdrant(qd, embed_client, warmup_queries)
        model_cache.clear()
        serve_s1_chroma(c_col, embed_client, warmup_queries)

    # Single run over all sizes
    results_by_size = {size: {m: None for m in methods} for size in test_sizes}

    for size in test_sizes:
        print(f"\nSize: {size}", flush=True)
        queries = queries_by_size[size]

        model_cache.clear()
        elapsed, _, stats = ResourceMonitor.measure(
            py_pid, get_pg_pid(conn_pg),
            lambda: serve_s1_pg_unified(conn_pg, queries))
        results_by_size[size]['pg_unified'] = BenchmarkResult(elapsed, stats)
        print(f"  pg_unified: {elapsed:.2f}s", flush=True)

        model_cache.clear()
        elapsed, _, stats = ResourceMonitor.measure(
            py_pid, None,
            lambda: serve_s1_qdrant(qd, embed_client, queries),
            container_name=QDRANT_CONTAINER_NAME)
        results_by_size[size]['poly_qdrant'] = BenchmarkResult(elapsed, stats)
        print(f"  poly_qdrant: {elapsed:.2f}s", flush=True)

        model_cache.clear()
        elapsed, _, stats = ResourceMonitor.measure(
            py_pid, None,
            lambda: serve_s1_chroma(c_col, embed_client, queries))
        results_by_size[size]['poly_chroma'] = BenchmarkResult(elapsed, stats)
        print(f"  poly_chroma: {elapsed:.2f}s", flush=True)

    # Cleanup
    conn_pg.close()
    if qd.collection_exists("faces"):
        qd.delete_collection("faces")
    qd.close()
    cleanup_chroma(c_client, c_path)

    # Collect metrics
    all_results = []
    for size in test_sizes:
        entry = {'size': size}
        for method in methods:
            r = results_by_size[size][method]
            entry[method] = {
                'time_s': r.time_s,
                'throughput': size / r.time_s if r.time_s > 0 else 0,
                'py_cpu': r.stats.py_cpu,
                'py_mem_delta': r.stats.py_delta_mb,
                'py_mem_peak': r.stats.py_peak_mb,
                'pg_cpu': r.stats.pg_cpu,
                'pg_mem_delta': r.stats.pg_delta_mb,
                'pg_mem_peak': r.stats.pg_peak_mb,
                'qd_cpu': r.stats.qd_cpu,
                'qd_mem_delta': r.stats.qd_delta_mb,
                'qd_mem_peak': r.stats.qd_peak_mb,
                'sys_cpu': r.stats.sys_cpu,
                'sys_mem': r.stats.sys_mem_mb,
            }
        all_results.append(entry)

    output_dir = OUTPUT_DIR / "scenario1_serving"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_single_run_csv(all_results, output_dir, run_id, methods)
    print("Run completed!")


if __name__ == "__main__":
    main()
