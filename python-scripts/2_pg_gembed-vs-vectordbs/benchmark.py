import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Callable, List

import chromadb
from benchmark_utils import (
    QDRANT_URL, QDRANT_CONTAINER_NAME, EMBED_ANYTHING_MODEL,
    BenchmarkResult, ResourceMonitor,
    EmbedAnythingDirectClient, EmbeddingWrapper,
    safe_stdev, calc_iqr, compute_metrics,
    connect_and_get_pid, warmup_pg_connection, cleanup_chroma,
)
from data.loader import get_review_texts
from plot_utils import save_results_csv, generate_plots
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct

# Configuration
TEST_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
RUNS_PER_SIZE = 5

OUTPUT_DIR = Path(__file__).parent / "output"


# =============================================================================
# PostgreSQL Functions
# =============================================================================

def setup_pg_schema(conn):
    """Initialize PostgreSQL schema."""
    cur = conn.cursor()
    cur.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE EXTENSION IF NOT EXISTS pg_gembed;
                DROP TABLE IF EXISTS embeddings_test;
                CREATE TABLE embeddings_test
                (
                    id        SERIAL PRIMARY KEY,
                    text      TEXT,
                    embedding vector(384)
                );
                """)
    conn.commit()
    cur.close()


def populate_pg_database(conn, texts: List[str], provider: str):
    """Insert data using internal generation."""
    cur = conn.cursor()
    sql = """
          WITH input_data AS (SELECT %s::text[] AS texts)
          INSERT
          INTO embeddings_test (text, embedding)
          SELECT t, e
          FROM input_data,
               unnest(texts, embed_texts(%s, %s, texts)) AS x(t, e)
          """
    cur.execute(sql, (texts, provider, EMBED_ANYTHING_MODEL))
    conn.commit()
    cur.close()


def setup_pg_indexed(conn, texts: List[str], provider: str):
    """Index exists BEFORE embedding generation."""
    setup_pg_schema(conn)
    cur = conn.cursor()
    cur.execute(
        "CREATE INDEX ON embeddings_test USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=100);")
    conn.commit()
    cur.close()
    populate_pg_database(conn, texts, provider)


def setup_pg_deferred(conn, texts: List[str], provider: str):
    """Index created AFTER embedding generation."""
    setup_pg_schema(conn)
    populate_pg_database(conn, texts, provider)
    cur = conn.cursor()
    cur.execute(
        "CREATE INDEX ON embeddings_test USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=100);")
    conn.commit()
    cur.close()


# =============================================================================
# ChromaDB Functions
# =============================================================================

def create_chroma_client(base_path: str = "./chroma_bench", embed_fn: Callable = None):
    """Create a fresh ChromaDB persistent client."""
    db_path = f"{base_path}_{time.time_ns()}"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    client = chromadb.PersistentClient(path=db_path)

    configuration = {
        "hnsw": {
            "space": "cosine",
            "max_neighbors": 16,
            "ef_construction": 100
        }
    }

    emb_obj = EmbeddingWrapper(embed_fn)
    collection = client.create_collection("bench", embedding_function=emb_obj, configuration=configuration)
    return client, collection, db_path


def benchmark_chroma(collection, texts: List[str], embed_fn: Callable):
    """Benchmark ChromaDB with external embeddings."""
    embeddings = embed_fn(texts)
    collection.add(
        ids=[f"id_{j}" for j in range(len(texts))],
        embeddings=embeddings,
        documents=texts
    )


# =============================================================================
# Qdrant Functions
# =============================================================================

def setup_qdrant(client, texts: List[str], embed_fn: Callable, deferred: bool):
    """Initialize Qdrant and ingest data."""
    if client.collection_exists("bench"):
        client.delete_collection("bench")

    hnsw_config = models.HnswConfigDiff(m=16, ef_construct=100)
    client.create_collection(
        collection_name="bench",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE, hnsw_config=hnsw_config)
    )

    if deferred:
        client.update_collection("bench", optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0))

    embeddings = embed_fn(texts)
    points = [PointStruct(id=j, vector=embeddings[j], payload={"text": texts[j]}) for j in range(len(texts))]
    client.upsert(collection_name="bench", points=points, wait=True)

    if deferred:
        client.update_collection("bench", optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000))


def cleanup_qdrant(client):
    """Clean up Qdrant client."""
    if client.collection_exists("bench"):
        client.delete_collection(collection_name="bench")
    client.close()
    time.sleep(0.1)


# =============================================================================
# Benchmark Runner Functions
# =============================================================================

def run_pg_method(texts: List[str], provider: str, strategy: str, runs: int) -> List[BenchmarkResult]:
    """Run PG benchmark with a fresh connection per method/size."""
    py_pid = os.getpid()
    conn, pg_pid = connect_and_get_pid()

    try:
        # Warm up the PG connection (loads pg_gembed, JIT, etc.)
        warmup_pg_connection(conn, provider)

        # Method-specific warm-up with small data
        warmup_texts = get_review_texts(8, shuffle=False)
        fn = setup_pg_indexed if strategy == "indexed" else setup_pg_deferred
        fn(conn, warmup_texts, provider)

        # Run benchmark iterations
        results = []
        for _ in range(runs):
            elapsed, _, stats = ResourceMonitor.measure(py_pid, pg_pid, lambda: fn(conn, texts, provider))
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))

        return results
    finally:
        conn.close()


def run_chroma_method(texts: List[str], embed_client: EmbedAnythingDirectClient, runs: int) -> List[BenchmarkResult]:
    """Run ChromaDB benchmark."""
    py_pid = os.getpid()

    # Warm up
    warmup_texts = get_review_texts(8, shuffle=False)
    client, collection, db_path = create_chroma_client(embed_fn=embed_client.embed)
    try:
        benchmark_chroma(collection, warmup_texts, embed_client.embed)
    finally:
        cleanup_chroma(client, db_path)

    results = []
    for _ in range(runs):
        client, collection, db_path = create_chroma_client(embed_fn=embed_client.embed)
        try:
            elapsed, _, stats = ResourceMonitor.measure(py_pid, None,
                                                        lambda: benchmark_chroma(collection, texts, embed_client.embed))
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        finally:
            cleanup_chroma(client, db_path)
    return results


def run_qdrant_method(texts: List[str], embed_client: EmbedAnythingDirectClient, runs: int, deferred: bool) -> List[
    BenchmarkResult]:
    """Run Qdrant benchmark."""
    py_pid = os.getpid()

    # Warm up
    warmup_texts = get_review_texts(8, shuffle=False)
    client = QdrantClient(url=QDRANT_URL)
    try:
        setup_qdrant(client, warmup_texts, embed_client.embed, deferred)
    finally:
        cleanup_qdrant(client)

    results = []
    for _ in range(runs):
        client = QdrantClient(url=QDRANT_URL)
        try:
            elapsed, _, stats = ResourceMonitor.measure(py_pid, None,
                                                        lambda: setup_qdrant(client, texts, embed_client.embed,
                                                                             deferred),
                                                        container_name=QDRANT_CONTAINER_NAME)
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        finally:
            cleanup_qdrant(client)
    return results


# =============================================================================
# Output Functions
# =============================================================================

def print_header():
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
    py_deltas = [r.stats.py_delta_mb for r in results];
    py_peaks = [r.stats.py_peak_mb for r in results];
    py_cpus = [r.stats.py_cpu for r in results]
    pg_deltas = [r.stats.pg_delta_mb for r in results];
    pg_peaks = [r.stats.pg_peak_mb for r in results];
    pg_cpus = [r.stats.pg_cpu for r in results]
    qd_deltas = [r.stats.qd_delta_mb for r in results];
    qd_peaks = [r.stats.qd_peak_mb for r in results];
    qd_cpus = [r.stats.qd_cpu for r in results]
    sys_mems = [r.stats.sys_mem_mb for r in results];
    sys_cpus = [r.stats.sys_cpu for r in results]

    def fmt(vals, p=1): return f"{mean(vals):.{p}f}±{safe_stdev(vals):.{p}f}"

    def fmt_med(vals, p=1): return f"{median(vals):.{p}f}±{calc_iqr(vals):.{p}f}"

    row_fmt = "  {label:<12}{med:>7} | {time:>14} | {pyd:>13} | {pyp:>13} | {pyc:>13} | {pgd:>13} | {pgp:>13} | {pgc:>13} | {qdd:>13} | {qdp:>13} | {qdc:>13} | {sysm:>13} | {sysc:>13}"
    print(
        row_fmt.format(label=label, med='', time=fmt(times, 3), pyd=fmt(py_deltas), pyp=fmt(py_peaks), pyc=fmt(py_cpus),
                       pgd=fmt(pg_deltas), pgp=fmt(pg_peaks), pgc=fmt(pg_cpus), qdd=fmt(qd_deltas), qdp=fmt(qd_peaks),
                       qdc=fmt(qd_cpus), sysm=fmt(sys_mems, 0), sysc=fmt(sys_cpus)), flush=True)
    print(
        row_fmt.format(label=label, med=' (med)', time=fmt_med(times, 3), pyd=fmt_med(py_deltas), pyp=fmt_med(py_peaks),
                       pyc=fmt_med(py_cpus), pgd=fmt_med(pg_deltas), pgp=fmt_med(pg_peaks), pgc=fmt_med(pg_cpus),
                       qdd=fmt_med(qd_deltas), qdp=fmt_med(qd_peaks), qdc=fmt_med(qd_cpus), sysm=fmt_med(sys_mems, 0),
                       sysc=fmt_med(sys_cpus)), flush=True)


def main():
    print(f"\n{'=' * 105}\nBENCHMARK 2: PG GEMBED VS VECTOR DBS (INDEXED VS DEFERRED)\n{'=' * 105}\n")
    embed_client = EmbedAnythingDirectClient()
    print_header()
    all_results = []

    for size in TEST_SIZES:
        texts = get_review_texts(size, shuffle=False)
        print(f"Size: {size}", flush=True)

        res_pg_local_idx = run_pg_method(texts, "embed_anything", "indexed", RUNS_PER_SIZE)
        res_pg_local_def = run_pg_method(texts, "embed_anything", "deferred", RUNS_PER_SIZE)
        res_pg_grpc_idx = run_pg_method(texts, "grpc", "indexed", RUNS_PER_SIZE)
        res_pg_grpc_def = run_pg_method(texts, "grpc", "deferred", RUNS_PER_SIZE)
        res_qd_idx = run_qdrant_method(texts, embed_client, RUNS_PER_SIZE, deferred=False)
        res_qd_def = run_qdrant_method(texts, embed_client, RUNS_PER_SIZE, deferred=True)
        res_ch = run_chroma_method(texts, embed_client, RUNS_PER_SIZE)

        print_result("PG Local Indexed", res_pg_local_idx)
        print_result("PG Local Deferred", res_pg_local_def)
        print_result("PG gRPC Indexed", res_pg_grpc_idx)
        print_result("PG gRPC Deferred", res_pg_grpc_def)
        print_result("QD Indexed", res_qd_idx)
        print_result("QD Deferred", res_qd_def)
        print_result("Chroma", res_ch)
        print()

        all_results.append({
            'size': size,
            'pg_local_indexed': compute_metrics(size, res_pg_local_idx),
            'pg_local_deferred': compute_metrics(size, res_pg_local_def),
            'pg_grpc_indexed': compute_metrics(size, res_pg_grpc_idx),
            'pg_grpc_deferred': compute_metrics(size, res_pg_grpc_def),
            'qd_indexed': compute_metrics(size, res_qd_idx),
            'qd_deferred': compute_metrics(size, res_qd_def),
            'chroma': compute_metrics(size, res_ch),
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    methods = ['pg_local_indexed', 'pg_local_deferred', 'pg_grpc_indexed', 'pg_grpc_deferred', 'qd_indexed',
               'qd_deferred', 'chroma']
    save_results_csv(all_results, OUTPUT_DIR, timestamp, methods)
    generate_plots(all_results, OUTPUT_DIR, timestamp, methods)


if __name__ == "__main__":
    main()
