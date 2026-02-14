import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import List

import chromadb
from benchmark_utils import (
    DB_CONFIG, QDRANT_URL, QDRANT_CONTAINER_NAME, EMBED_ANYTHING_MODEL,
    BenchmarkResult, ResourceMonitor,
    EmbedAnythingDirectClient, EmbeddingWrapper,
    safe_stdev, calc_iqr, compute_metrics,
    connect_pg, get_pg_pid, warmup_pg_connection,
)
from plot_utils import save_results_csv, generate_plots
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct

# Configuration
INGESTION_SET_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
RUNS_INGESTION = 5

SERVING_TEST_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
RUNS_SERVING = 5

OUTPUT_DIR = Path(__file__).parent / "output"


# =============================================================================
# Client & Connectors
# =============================================================================

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

    texts = [b[0] for b in ingestion_data]
    spams = [b[1] for b in ingestion_data]

    cur.execute('''
                INSERT INTO reviews (text, spam, embedding)
                SELECT t, s, e
                FROM unnest(%s::text[], %s::boolean[], embed_texts('embed_anything', %s, %s::text[])) AS i(t, s, e);
                ''', (texts, spams, EMBED_ANYTHING_MODEL, texts))

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

    if deferred:
        client.update_collection("reviews", optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0))

    texts = [t for t, s in ingestion_data]
    embeddings = embed_client.embed(texts)
    points = [PointStruct(id=j, vector=embeddings[j], payload={"text": t, "spam": s})
              for j, (t, s) in enumerate(ingestion_data)]
    client.upsert("reviews", points, wait=True)

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
    ids = [str(j) for j in range(len(ingestion_data))]
    collection.add(ids=ids, embeddings=embeddings, metadatas=spams, documents=texts)
    return collection


# =============================================================================
# Serving Functions
# =============================================================================

def serve_pg(conn, input_texts):
    cur = conn.cursor()
    
    # SQL to get majority vote
    sql = '''
          WITH predictions AS (SELECT i.ord, (count(*) filter (where t.spam) >= 3)::boolean as predicted_spam
                               FROM unnest(%s::text[],
                                           embed_texts('embed_anything', %s, %s::text[])) WITH ORDINALITY AS i(txt, embedding, ord)
                                        CROSS JOIN LATERAL (
                                   SELECT spam
                                   FROM reviews t
                                   ORDER BY t.embedding <-> i.embedding
                                   LIMIT 5
                                   ) t
                               GROUP BY i.ord)
          SELECT predicted_spam
          FROM predictions
          ORDER BY ord;
          '''
    cur.execute(sql, (input_texts, EMBED_ANYTHING_MODEL, input_texts))
    all_predictions = cur.fetchall()
        
    conn.commit()
    cur.close()
    return all_predictions


def serve_chroma(collection, embed_client, input_texts):
    predictions = []
    embs = embed_client.embed(input_texts)
    results = collection.query(query_embeddings=embs, n_results=5, include=["metadatas"])

    for metas in results['metadatas']:
        spam_votes = sum(1 for m in metas if m.get('spam'))
        predictions.append(spam_votes >= 3)
    return predictions


def serve_qdrant(client, embed_client, input_texts):
    predictions = []
    embs = embed_client.embed(input_texts)

    # Use Batch Search API to send all queries in one request
    requests = [
        models.QueryRequest(query=emb, limit=5, with_payload=True)
        for emb in embs
    ]
    results = client.query_batch_points(collection_name="reviews", requests=requests)

    # results is a list of QueryResponse
    for response in results:
        spam_votes = sum(1 for p in response.points if p.payload.get('spam'))
        predictions.append(spam_votes >= 3)
    return predictions


# =============================================================================
# Reporting
# =============================================================================

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

    from data.loader import get_reviews_with_labels

    full_data_raw = get_reviews_with_labels(max_ingestion + max(SERVING_TEST_SIZES), shuffle=True,
                                            legitimate_only=False)
    full_data = [(text, bool(spam)) for text, spam in full_data_raw]

    # Reserve test pool from the end of the loaded data
    test_pool = full_data[max_ingestion:]

    # Phase 1: Ingestion
    embed_client = EmbedAnythingDirectClient()
    final_chroma_path = None

    all_ingestion_metrics = []

    # Warm-up Ingestion
    print("Warming up ingestion...")
    warmup_data = full_data[:8]
    
    # PG Warmup
    conn = connect_pg()
    setup_pg_indexed(conn, warmup_data)
    conn.close()
    
    conn = connect_pg()
    setup_pg_deferred(conn, warmup_data)
    conn.close()
    
    # Qdrant Warmup
    qd_client = create_qdrant_client()
    setup_qdrant_common(qd_client, embed_client, warmup_data, deferred=False)
    setup_qdrant_common(qd_client, embed_client, warmup_data, deferred=True)
    if qd_client.collection_exists("reviews"):
        qd_client.delete_collection("reviews")
    qd_client.close()
    
    # Chroma Warmup
    c_client, c_path = create_chroma_client()
    setup_chroma(c_client, embed_client, warmup_data)
    cleanup_chroma(c_client, c_path)

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
        
        # Open connections for this size
        conn_idx = connect_pg()
        pid_idx = get_pg_pid(conn_idx)
        warmup_pg_connection(conn_idx)
        
        conn_def = connect_pg()
        pid_def = get_pg_pid(conn_def)
        warmup_pg_connection(conn_def)

        try:
            for run_idx in range(RUNS_INGESTION):
                # Postgres Indexed
                elapsed, _, stats = ResourceMonitor.measure(py_pid, pid_idx,
                                                         lambda: setup_pg_indexed(conn_idx, current_ingestion_data))
                results_pg_indexed.append(BenchmarkResult(elapsed, stats))

                # Postgres Deferred
                elapsed, _, stats = ResourceMonitor.measure(py_pid, pid_def,
                                                         lambda: setup_pg_deferred(conn_def, current_ingestion_data))
                results_pg_deferred.append(BenchmarkResult(elapsed, stats))

                # Qdrant Indexed
                qd_client = create_qdrant_client()
                elapsed, _, stats = ResourceMonitor.measure(py_pid, None, lambda: setup_qdrant_common(qd_client, embed_client,
                                                                                                   current_ingestion_data,
                                                                                                   deferred=False),
                                                           container_name=QDRANT_CONTAINER_NAME)
                results_qd_indexed.append(BenchmarkResult(elapsed, stats))
                qd_client.close()

                # Qdrant Deferred
                qd_client = create_qdrant_client()
                elapsed, _, stats = ResourceMonitor.measure(py_pid, None, lambda: setup_qdrant_common(qd_client, embed_client,
                                                                                                   current_ingestion_data,
                                                                                                   deferred=True),
                                                           container_name=QDRANT_CONTAINER_NAME)
                results_qd_deferred.append(BenchmarkResult(elapsed, stats))
                qd_client.close()

                # Chroma
                c_client, c_path = create_chroma_client()
                elapsed, _, stats = ResourceMonitor.measure(py_pid, None, lambda: setup_chroma(c_client, embed_client,
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
        finally:
            conn_idx.close()
            conn_def.close()

        # Print results for this size
        print_result("PG Indexed", results_pg_indexed)
        print_result("PG Deferred", results_pg_deferred)
        print_result("QD Indexed", results_qd_indexed)
        print_result("QD Deferred", results_qd_deferred)
        print_result("Chroma", results_chroma)

        # Collect metrics
        all_ingestion_metrics.append({
            'size': ingestion_size,
            'pg_indexed': compute_metrics(ingestion_size, results_pg_indexed),
            'pg_deferred': compute_metrics(ingestion_size, results_pg_deferred),
            'qd_indexed': compute_metrics(ingestion_size, results_qd_indexed),
            'qd_deferred': compute_metrics(ingestion_size, results_qd_deferred),
            'chroma': compute_metrics(ingestion_size, results_chroma),
        })

    # Save Ingestion Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ingestion_methods = ['pg_indexed', 'pg_deferred', 'qd_indexed', 'qd_deferred', 'chroma']
    ingestion_labels = {
        'pg_indexed': 'PG Indexed', 'pg_deferred': 'PG Deferred',
        'qd_indexed': 'Qdrant Indexed', 'qd_deferred': 'Qdrant Deferred',
        'chroma': 'Chroma'
    }

    save_results_csv(all_ingestion_metrics, OUTPUT_DIR / "ingestion", timestamp, ingestion_methods)
    generate_plots(all_ingestion_metrics, OUTPUT_DIR / "ingestion", timestamp, ingestion_methods)

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

    all_serving_metrics = []

    print_detailed_header()

    # Serving Warm-up
    print("Warming up serving...")
    warmup_inputs = [t for t, l in test_pool[:8]]
    serve_pg(pg_conn, warmup_inputs)
    serve_qdrant(qd_client, embed_client, warmup_inputs)
    serve_chroma(c_collection, embed_client, warmup_inputs)

    for size in SERVING_TEST_SIZES:
        print(f"Test Size: {size}")
        test_inputs = [t for t, l in test_pool[:size]]

        m_pg = []
        for _ in range(RUNS_SERVING):
            elapsed, _, stats = ResourceMonitor.measure(py_pid, pg_pid, lambda: serve_pg(pg_conn, test_inputs))
            m_pg.append(BenchmarkResult(elapsed, stats))

        m_qd = []
        for _ in range(RUNS_SERVING):
            elapsed, _, stats = ResourceMonitor.measure(py_pid, None,
                                                     lambda: serve_qdrant(qd_client, embed_client, test_inputs),
                                                     container_name=QDRANT_CONTAINER_NAME)
            m_qd.append(BenchmarkResult(elapsed, stats))

        m_ch = []
        for _ in range(RUNS_SERVING):
            elapsed, _, stats = ResourceMonitor.measure(py_pid, None,
                                                     lambda: serve_chroma(c_collection, embed_client, test_inputs))
            m_ch.append(BenchmarkResult(elapsed, stats))

        print_result("PostgreSQL", m_pg)
        print_result("Qdrant", m_qd)
        print_result("Chroma", m_ch)
        print()

        res = {
            'size': size,
            'pg': compute_metrics(size, m_pg),
            'qdrant': compute_metrics(size, m_qd),
            'chroma': compute_metrics(size, m_ch)
        }
        all_serving_metrics.append(res)

    # Save Serving Results
    serving_methods = ['pg', 'qdrant', 'chroma']

    save_results_csv(all_serving_metrics, OUTPUT_DIR / "serving", timestamp, serving_methods)
    generate_plots(all_serving_metrics, OUTPUT_DIR / "serving", timestamp, serving_methods)

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
