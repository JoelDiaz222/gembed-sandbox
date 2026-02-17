#!/usr/bin/env python3
"""
Benchmark 5 - TPCx-AI UC9, Scenario 1: Ingestion (paths stored in PG)
Embeds images and stores (path + embedding) in PG, Qdrant, and Chroma.
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
    connect_and_get_pid, warmup_pg_connection,
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

model_cache = {}


# =============================================================================
# Image Embedding Client
# =============================================================================

class EmbedAnythingImageClient:
    def embed_files(self, paths: List[str]) -> List[List[float]]:
        model = self._get_model()
        base_temp = Path("temp_bench_imgs")
        if base_temp.exists():
            shutil.rmtree(base_temp)
        base_temp.mkdir()
        try:
            for idx, p in enumerate(paths):
                ext = Path(p).suffix
                shutil.copy(p, base_temp / f"{idx:06d}{ext}")
            res = embed_anything.embed_image_directory(str(base_temp), embedder=model)
            embeddings = []
            if isinstance(res, list):
                for item in res:
                    if hasattr(item, 'embedding'):
                        embeddings.append(item.embedding)
        finally:
            if base_temp.exists():
                shutil.rmtree(base_temp)
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


def truncate_pg_table(conn):
    cur = conn.cursor()
    cur.execute("TRUNCATE faces, persons RESTART IDENTITY")
    conn.commit()
    cur.close()


def s1_populate_pg(conn, image_paths: List[str]):
    cur = conn.cursor()
    batch_names = [get_person_name(p) for p in image_paths]
    batch_images = []
    for p in image_paths:
        with open(p, "rb") as f:
            batch_images.append(f.read())
    unique_names = list(set(batch_names))
    execute_values(cur, "INSERT INTO persons (name) VALUES %s ON CONFLICT (name) DO NOTHING",
                   [(n,) for n in unique_names])
    sql = """
          INSERT INTO faces (path, person_id, embedding)
          SELECT t.p, p.id, t.e
          FROM unnest(%s::text[], %s::text[],
                      embed_images('embed_anything', %s, %s::bytea[])) AS t(p, n, e)
                   JOIN persons p ON t.n = p.name \
          """
    cur.execute(sql, (image_paths, batch_names, MODEL_NAME, batch_images))
    conn.commit()
    cur.close()


def s1_ingest_pg_indexed(conn, image_paths: List[str]):
    cur = conn.cursor()
    cur.execute("CREATE INDEX ON faces USING hnsw (embedding vector_cosine_ops)"
                " WITH (m=16, ef_construction=100);")
    cur.close()
    s1_populate_pg(conn, image_paths)


# =============================================================================
# Vector DB Functions
# =============================================================================

def create_chroma_client(base_path: str = "./chroma_bench_uc9_s1"):
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


def s1_ingest_chroma(collection, image_paths: List[str], embed_client: EmbedAnythingImageClient):
    embeddings = embed_client.embed_files(image_paths)
    ids = [str(idx) for idx in range(len(image_paths))]
    metas = [{"path": p, "person_name": get_person_name(p)} for p in image_paths]
    if embeddings:
        collection.add(ids=ids, embeddings=embeddings, metadatas=metas)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark 5 UC9 - Scenario 1 Ingestion')
    parser.add_argument('--sizes', type=int, nargs='+', required=True)
    parser.add_argument('--run-id', type=str, default=None)
    args = parser.parse_args()

    test_sizes = args.sizes
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    methods = ['pg_indexed', 'qd_indexed', 'chroma']

    print(f"\nStarting Benchmark 5 UC9 - Scenario 1 Ingestion")
    print(f"Run ID: {run_id}")
    print(f"Sizes: {test_sizes}")
    print("Loading model...", flush=True)
    EmbedAnythingImageClient._get_model()
    embed_client = EmbedAnythingImageClient()
    py_pid = os.getpid()

    # Pre-compute paths for all sizes
    paths_by_size = {}
    for size in test_sizes:
        n_people = max(1, size // INGEST_PER_PERSON)
        imgs_per = INGEST_PER_PERSON if size >= INGEST_PER_PERSON else size
        paths_by_size[size] = get_image_paths(n_people, imgs_per)

    # Warm-up
    print("Warming up...")
    warmup_paths = get_image_paths(1, 2)
    if warmup_paths:
        conn_w, _ = connect_and_get_pid()
        warmup_pg_connection(conn_w)
        setup_pg_schema(conn_w)
        s1_ingest_pg_indexed(conn_w, warmup_paths)
        truncate_pg_table(conn_w)
        conn_w.close()

        qd_w = create_qdrant_client()
        s1_ingest_qdrant(qd_w, warmup_paths, embed_client)
        if qd_w.collection_exists("faces"):
            qd_w.delete_collection("faces")
        qd_w.close()

        c_w, c_path_w = create_chroma_client()
        col_w = c_w.create_collection("faces", configuration={"hnsw": {"space": "cosine"}})
        s1_ingest_chroma(col_w, warmup_paths, embed_client)
        cleanup_chroma(c_w, c_path_w)

    # Single run over all sizes
    results_by_size = {size: {m: None for m in methods} for size in test_sizes}

    for size in test_sizes:
        print(f"\nSize: {size}", flush=True)
        paths = paths_by_size[size]
        model_cache.clear()

        conn, pg_pid = connect_and_get_pid()
        warmup_pg_connection(conn)
        setup_pg_schema(conn)

        try:
            # PG Indexed
            elapsed, _, stats = ResourceMonitor.measure(
                py_pid, pg_pid,
                lambda: s1_ingest_pg_indexed(conn, paths))
            results_by_size[size]['pg_indexed'] = BenchmarkResult(elapsed, stats)
            print(f"  pg_indexed: {elapsed:.2f}s", flush=True)
            conn.close()

            # Qdrant
            qd = create_qdrant_client()
            model_cache.clear()
            try:
                elapsed, _, stats = ResourceMonitor.measure(
                    py_pid, None,
                    lambda: s1_ingest_qdrant(qd, paths, embed_client),
                    container_name=QDRANT_CONTAINER_NAME)
                results_by_size[size]['qd_indexed'] = BenchmarkResult(elapsed, stats)
                print(f"  qd_indexed: {elapsed:.2f}s", flush=True)
            finally:
                if qd.collection_exists("faces"):
                    qd.delete_collection("faces")
                qd.close()

            # Chroma
            c_client, c_path = create_chroma_client()
            col = c_client.create_collection("faces", configuration={"hnsw": {"space": "cosine"}})
            model_cache.clear()
            try:
                elapsed, _, stats = ResourceMonitor.measure(
                    py_pid, None,
                    lambda: s1_ingest_chroma(col, paths, embed_client))
                results_by_size[size]['chroma'] = BenchmarkResult(elapsed, stats)
                print(f"  chroma: {elapsed:.2f}s", flush=True)
            finally:
                cleanup_chroma(c_client, c_path)
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            raise

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

    output_dir = OUTPUT_DIR / "scenario1_ingestion"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_single_run_csv(all_results, output_dir, run_id, methods)
    print("Run completed!")


if __name__ == "__main__":
    main()
