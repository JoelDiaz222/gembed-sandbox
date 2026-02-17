#!/usr/bin/env python3
"""
Benchmark 4 - TPCx-AI UC4: Serving Benchmark
Populate DBs with --db-size reviews, then serve ANN look-ups for --sizes query batches.
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

def create_chroma_client(base_path: str = "./chroma_bench_uc4_svc"):
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
# Schema / Population Functions
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
# Serving Functions
# =============================================================================

def serve_pg(conn, input_texts: List[str]):
    cur = conn.cursor()
    sql = """
          WITH predictions AS (SELECT i.ord,
                                      (count(*) FILTER (WHERE t.spam) >= 3)::boolean AS predicted_spam
                               FROM unnest(%s::text[],
                                           embed_texts('embed_anything', %s, %s::text[]))
                                        WITH ORDINALITY AS i(txt, embedding, ord)
                                        CROSS JOIN LATERAL (
                                   SELECT spam
                                   FROM reviews t
                                   ORDER BY t.embedding <-> i.embedding
                                   LIMIT 5
                                   ) t
                               GROUP BY i.ord)
          SELECT predicted_spam
          FROM predictions
          ORDER BY ord; \
          """
    cur.execute(sql, (input_texts, EMBED_ANYTHING_MODEL, input_texts))
    result = cur.fetchall()
    conn.commit()
    cur.close()
    return result


def serve_chroma(collection, embed_client, input_texts: List[str]):
    embs = embed_client.embed(input_texts)
    results = collection.query(query_embeddings=embs, n_results=5, include=["metadatas"])
    predictions = []
    for metas in results['metadatas']:
        spam_votes = sum(1 for m in metas if m.get('spam'))
        predictions.append(spam_votes >= 3)
    return predictions


def serve_qdrant(client, embed_client, input_texts: List[str]):
    embs = embed_client.embed(input_texts)
    requests = [models.QueryRequest(query=emb, limit=5, with_payload=True) for emb in embs]
    results = client.query_batch_points(collection_name="reviews", requests=requests)
    predictions = []
    for response in results:
        spam_votes = sum(1 for p in response.points if p.payload.get('spam'))
        predictions.append(spam_votes >= 3)
    return predictions


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark 4 UC4 - Serving')
    parser.add_argument('--sizes', type=int, nargs='+', required=True,
                        help='Query batch sizes to test')
    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--db-size', type=int, required=True,
                        help='Number of records to pre-populate in DB')
    args = parser.parse_args()

    test_sizes = args.sizes
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    db_size = args.db_size
    methods = ['pg', 'qdrant', 'chroma']

    print(f"\nStarting Benchmark 4 UC4 - Serving")
    print(f"Run ID: {run_id}")
    print(f"DB Size: {db_size}")
    print(f"Query Sizes: {test_sizes}")

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.loader import get_reviews_with_labels

    total_needed = db_size + max(test_sizes)
    print(f"Loading {total_needed} reviews...")
    full_data_raw = get_reviews_with_labels(total_needed, shuffle=False, legitimate_only=False)
    full_data = [(text, bool(spam)) for text, spam in full_data_raw]

    db_data = full_data[:db_size]
    test_pool = full_data[db_size:]

    embed_client = EmbedAnythingDirectClient()
    py_pid = os.getpid()

    # Warm-up serving
    print("Setting up DBs...")
    conn_pg, pg_pid = connect_and_get_pid()
    warmup_pg_connection(conn_pg)
    setup_pg_indexed(conn_pg, db_data)

    qd = create_qdrant_client()
    setup_qdrant_common(qd, embed_client, db_data)

    c_client, c_path = create_chroma_client()
    c_col = setup_chroma(c_client, embed_client, db_data)

    # Warm-up queries
    print("Warming up queries...")
    warmup_texts = [t for t, s in test_pool[:8]]
    serve_pg(conn_pg, warmup_texts)
    serve_qdrant(qd, embed_client, warmup_texts)
    serve_chroma(c_col, embed_client, warmup_texts)

    # Single run over all sizes
    results_by_size = {size: {m: None for m in methods} for size in test_sizes}

    for size in test_sizes:
        print(f"\nSize: {size}", flush=True)
        input_texts = [t for t, s in test_pool[:size]]

        elapsed, _, stats = ResourceMonitor.measure(
            py_pid, pg_pid,
            lambda: serve_pg(conn_pg, input_texts))
        results_by_size[size]['pg'] = BenchmarkResult(elapsed, stats)
        print(f"  pg: {elapsed:.2f}s", flush=True)

        elapsed, _, stats = ResourceMonitor.measure(
            py_pid, None,
            lambda: serve_qdrant(qd, embed_client, input_texts),
            container_name=QDRANT_CONTAINER_NAME)
        results_by_size[size]['qdrant'] = BenchmarkResult(elapsed, stats)
        print(f"  qdrant: {elapsed:.2f}s", flush=True)

        elapsed, _, stats = ResourceMonitor.measure(
            py_pid, None,
            lambda: serve_chroma(c_col, embed_client, input_texts))
        results_by_size[size]['chroma'] = BenchmarkResult(elapsed, stats)
        print(f"  chroma: {elapsed:.2f}s", flush=True)

    # Cleanup
    conn_pg.close()
    if qd.collection_exists("reviews"):
        qd.delete_collection("reviews")
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

    output_dir = OUTPUT_DIR / "serving"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_single_run_csv(all_results, output_dir, run_id, methods)
    print("Run completed!")


if __name__ == "__main__":
    main()
