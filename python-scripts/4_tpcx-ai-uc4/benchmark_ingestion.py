#!/usr/bin/env python3
"""
Benchmark 4 - TPCx-AI UC4: Ingestion Benchmark
Inserts review texts + generates embeddings across PG, Qdrant, and Chroma.
"""
import argparse
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import chromadb
from benchmark_utils import (
    QDRANT_URL, QDRANT_CONTAINER_NAME, EMBED_ANYTHING_MODEL,
    BenchmarkResult, ResourceMonitor,
    EmbedAnythingDirectClient, EmbeddingWrapper,
    connect_and_get_pid, warmup_pg_connection,
)
from plot_utils import save_single_run_csv
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct

OUTPUT_DIR = Path(__file__).parent / "output"


# =============================================================================
# Client helpers
# =============================================================================

def create_chroma_client(base_path: str = "./chroma_bench_uc4"):
    db_path = f"{base_path}_{time.time_ns()}"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    client = chromadb.PersistentClient(path=db_path)
    return client, db_path


def cleanup_chroma(client, db_path: str):
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
    cur.execute("""
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
                """)
    cur.close()


def populate_pg_database(conn, ingestion_data: List[Tuple]):
    cur = conn.cursor()
    texts = [b[0] for b in ingestion_data]
    spams = [b[1] for b in ingestion_data]
    cur.execute("""
                INSERT INTO reviews (text, spam, embedding)
                SELECT t, s, e
                FROM unnest(%s::text[], %s::boolean[],
                            embed_texts('embed_anything', %s, %s::text[])) AS i(t, s, e);
                """, (texts, spams, EMBED_ANYTHING_MODEL, texts))
    cur.close()


def setup_pg_indexed(conn, ingestion_data: List[Tuple]):
    """Create HNSW index BEFORE embedding generation, then insert."""
    setup_pg_schema(conn)
    cur = conn.cursor()
    cur.execute("CREATE INDEX ON reviews USING hnsw (embedding vector_cosine_ops)"
                " WITH (m=16, ef_construction=100);")
    cur.close()
    populate_pg_database(conn, ingestion_data)
    conn.commit()


def setup_qdrant_common(client, embed_client, ingestion_data: List[Tuple]):
    if client.collection_exists("reviews"):
        client.delete_collection("reviews")
    hnsw_config = models.HnswConfigDiff(m=16, ef_construct=100)
    client.create_collection(
        "reviews",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE, hnsw_config=hnsw_config))
    texts = [t for t, s in ingestion_data]
    embeddings = embed_client.embed(texts)
    points = [PointStruct(id=j, vector=embeddings[j], payload={"text": t, "spam": s})
              for j, (t, s) in enumerate(ingestion_data)]
    client.upsert("reviews", points, wait=True)


def setup_chroma(client, embed_client, ingestion_data: List[Tuple]):
    configuration = {"hnsw": {"space": "cosine", "max_neighbors": 16, "ef_construction": 100}}
    emb_obj = EmbeddingWrapper(embed_client.embed)
    collection = client.create_collection("reviews", embedding_function=emb_obj,
                                          configuration=configuration)
    texts = [t for t, s in ingestion_data]
    embeddings = embed_client.embed(texts)
    spams = [{"spam": s} for t, s in ingestion_data]
    ids = [str(j) for j in range(len(ingestion_data))]
    collection.add(ids=ids, embeddings=embeddings, metadatas=spams, documents=texts)
    return collection


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark 4 UC4 - Ingestion')
    parser.add_argument('--sizes', type=int, nargs='+', required=True)
    parser.add_argument('--run-id', type=str, default=None)
    args = parser.parse_args()

    test_sizes = args.sizes
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    methods = ['pg_indexed', 'qd_indexed', 'chroma']

    print(f"\nStarting Benchmark 4 UC4 - Ingestion")
    print(f"Run ID: {run_id}")
    print(f"Sizes: {test_sizes}")

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.loader import get_reviews_with_labels

    max_size = max(test_sizes)
    print(f"Loading {max_size} reviews...")
    full_data_raw = get_reviews_with_labels(max_size, shuffle=False, legitimate_only=False)
    full_data = [(text, bool(spam)) for text, spam in full_data_raw]

    embed_client = EmbedAnythingDirectClient()
    py_pid = os.getpid()

    # Warm-up
    print("Warming up...")
    warmup_data = full_data[:8]
    conn_w, _ = connect_and_get_pid()
    warmup_pg_connection(conn_w)
    setup_pg_indexed(conn_w, warmup_data)
    conn_w.close()

    qd_w = create_qdrant_client()
    setup_qdrant_common(qd_w, embed_client, warmup_data)
    if qd_w.collection_exists("reviews"):
        qd_w.delete_collection("reviews")
    qd_w.close()

    c_w, c_path_w = create_chroma_client()
    setup_chroma(c_w, embed_client, warmup_data)
    cleanup_chroma(c_w, c_path_w)

    # Single run over all sizes
    results_by_size = {size: {m: None for m in methods} for size in test_sizes}

    for size in test_sizes:
        print(f"\nSize: {size}", flush=True)
        ingestion_data = full_data[:size]

        conn, pg_pid = connect_and_get_pid()
        warmup_pg_connection(conn)

        try:
            # PG Indexed
            elapsed, _, stats = ResourceMonitor.measure(
                py_pid, pg_pid,
                lambda: setup_pg_indexed(conn, ingestion_data))
            results_by_size[size]['pg_indexed'] = BenchmarkResult(elapsed, stats)
            print(f"  pg_indexed: {elapsed:.2f}s", flush=True)
            conn.close()

            # Qdrant
            qd = create_qdrant_client()
            try:
                elapsed, _, stats = ResourceMonitor.measure(
                    py_pid, None,
                    lambda: setup_qdrant_common(qd, embed_client, ingestion_data),
                    container_name=QDRANT_CONTAINER_NAME)
                results_by_size[size]['qd_indexed'] = BenchmarkResult(elapsed, stats)
                print(f"  qd_indexed: {elapsed:.2f}s", flush=True)
            finally:
                if qd.collection_exists("reviews"):
                    qd.delete_collection("reviews")
                qd.close()

            # Chroma
            c_client, c_path = create_chroma_client()
            try:
                elapsed, _, stats = ResourceMonitor.measure(
                    py_pid, None,
                    lambda: setup_chroma(c_client, embed_client, ingestion_data))
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

    output_dir = OUTPUT_DIR / "ingestion"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_single_run_csv(all_results, output_dir, run_id, methods)
    print("Run completed!")


if __name__ == "__main__":
    main()
