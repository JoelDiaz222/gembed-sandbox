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
    connect_and_get_pid, warmup_pg_connection,
)
from plot_utils import save_results_csv, generate_plots
from psycopg2 import extras
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct

# Configuration
TEST_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
RUNS_PER_SIZE = 5

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output"


# =============================================================================
# Data Generation
# =============================================================================

def generate_products(n: int) -> List[dict]:
    """Generate realistic product data with reviews from TPCx-AI dataset."""
    from data.loader import get_review_texts

    categories_pool = [
        "Electronics", "Home & Garden", "Sports", "Books", "Clothing",
        "Toys", "Health", "Automotive", "Food", "Office"
    ]

    adjectives = ["Premium", "Professional", "Essential", "Advanced", "Classic"]

    # Load real reviews from TPCx-AI dataset
    # Estimate ~3.5 reviews per product on average
    total_reviews_needed = int(n * 3.5) + 100
    reviews_pool = get_review_texts(total_reviews_needed, shuffle=False)
    review_idx = 0

    products = []
    for i in range(n):
        adj = adjectives[i % len(adjectives)]
        category_main = categories_pool[i % len(categories_pool)]

        product = {
            'name': f"{adj} {category_main} Item {i}",
            'description': f"High-quality {category_main.lower()} product featuring "
                           f"durable construction and modern design. Suitable for both "
                           f"professional and home use.",
            'price': 19.99 + (i % 10) * 10,
            'stock': 50 + (i * 7) % 200,
            'reviews': [],
            'categories': []
        }

        # Assign 2-5 real reviews per product from TPCx-AI dataset
        num_reviews = 2 + (i % 4)
        for j in range(num_reviews):
            product['reviews'].append({
                'rating': 3 + (j % 3),
                'text': reviews_pool[review_idx % len(reviews_pool)]
            })
            review_idx += 1

        # Assign 1-3 categories
        num_cats = 1 + (i % 3)
        product['categories'] = [
            categories_pool[(i + k) % len(categories_pool)]
            for k in range(num_cats)
        ]

        products.append(product)

    return products


def build_embedding_context(product: dict) -> str:
    """Build the text context for embedding (simulating JOIN in SQL)."""
    reviews_text = " | ".join([r['text'] for r in product['reviews'][:5]])
    categories_text = ", ".join(product['categories'])

    return (
        f"{product['name']}. {product['description']}. "
        f"Price: ${product['price']:.2f}. "
        f"Reviews: {reviews_text or 'No reviews'}. "
        f"Categories: {categories_text or 'Uncategorized'}"
    )


# =============================================================================
# PostgreSQL Functions
# =============================================================================

def setup_pg_database(conn):
    """Initialize PostgreSQL schema with product tables and HNSW index."""
    cur = conn.cursor()
    cur.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE EXTENSION IF NOT EXISTS pg_gembed;

                DROP TABLE IF EXISTS product_categories CASCADE;
                DROP TABLE IF EXISTS reviews CASCADE;
                DROP TABLE IF EXISTS products CASCADE;

                CREATE TABLE products
                (
                    product_id  SERIAL PRIMARY KEY,
                    name        TEXT NOT NULL,
                    description TEXT NOT NULL,
                    price       DECIMAL(10, 2),
                    stock_count INTEGER DEFAULT 0,
                    embedding   vector(384)
                );

                CREATE TABLE reviews
                (
                    review_id   SERIAL PRIMARY KEY,
                    product_id  INTEGER REFERENCES products (product_id),
                    rating      INTEGER CHECK (rating BETWEEN 1 AND 5),
                    review_text TEXT,
                    created_at  TIMESTAMP DEFAULT NOW()
                );

                CREATE TABLE product_categories
                (
                    product_id    INTEGER REFERENCES products (product_id),
                    category_name TEXT,
                    PRIMARY KEY (product_id, category_name)
                );

                CREATE INDEX ON products
                    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 100);
                """)
    conn.commit()
    cur.close()


def truncate_pg_tables(conn):
    """Clear all PostgreSQL tables and reset embeddings."""
    cur = conn.cursor()
    cur.execute("TRUNCATE products, reviews, product_categories CASCADE;")
    conn.commit()
    cur.close()


def clear_embeddings(conn):
    """Clear embeddings but keep relational data."""
    cur = conn.cursor()
    cur.execute("UPDATE products SET embedding = NULL;")
    conn.commit()
    cur.close()


def batch_insert_products(cur, products: List[dict]) -> List[int]:
    """Efficiently batch insert products and their dependencies."""
    # Prepare data for products
    product_values = [
        (p['name'], p['description'], p['price'], p['stock'])
        for p in products
    ]

    # Insert products and get IDs
    query = """
            INSERT INTO products (name, description, price, stock_count)
            VALUES
            %s
        RETURNING product_id \
            """
    results = extras.execute_values(cur, query, product_values, fetch=True)
    product_ids = [row[0] for row in results]

    # Prepare data for reviews and categories
    review_values = []
    category_values = []

    for pid, p in zip(product_ids, products):
        for review in p['reviews']:
            review_values.append((pid, review['rating'], review['text']))
        for category in p['categories']:
            category_values.append((pid, category))

    # Batch insert reviews
    if review_values:
        extras.execute_values(
            cur,
            "INSERT INTO reviews (product_id, rating, review_text) VALUES %s",
            review_values
        )

    # Batch insert categories
    if category_values:
        extras.execute_values(
            cur,
            "INSERT INTO product_categories (product_id, category_name) VALUES %s",
            category_values
        )

    return product_ids


def insert_product_data(conn, products: List[dict]) -> List[int]:
    """Insert product relational data (without embeddings)."""
    cur = conn.cursor()
    product_ids = batch_insert_products(cur, products)
    conn.commit()
    cur.close()
    return product_ids


# =============================================================================
# Vector DB Functions (ChromaDB + Qdrant)
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
    collection = client.create_collection("products", embedding_function=emb_obj, configuration=configuration)

    return client, collection, db_path


def cleanup_chroma(client, db_path: str):
    """Clean up ChromaDB client and files."""
    del client
    time.sleep(0.2)
    if os.path.exists(db_path):
        shutil.rmtree(db_path)


def create_qdrant_client():
    """Create a fresh Qdrant client."""
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists("products"):
        client.delete_collection("products")

    # Configure HNSW index parameters
    hnsw_config = models.HnswConfigDiff(
        m=16,
        ef_construct=100,
    )

    client.create_collection(
        collection_name="products",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE, hnsw_config=hnsw_config)
    )

    return client


def cleanup_qdrant(client):
    """Clean up Qdrant client and files."""
    if client.collection_exists("products"):
        client.delete_collection(collection_name="products")

    client.close()
    time.sleep(0.1)


# =============================================================================
# SCENARIO 2 FUNCTIONS: Pre-existing Data (embedding generation only)
# =============================================================================

def scenario2_unified(conn):
    """Scenario 2 - Unified: Generate embeddings for pre-existing data in batches of 2048."""
    cur = conn.cursor()

    # Fetch context data first
    cur.execute("""
                SELECT p.product_id,
                       p.name || '. ' || p.description ||
                       '. Price: $' || p.price::text ||
                       '. Reviews: ' || COALESCE(
                               (SELECT string_agg(r.review_text, ' | ' ORDER BY r.created_at DESC)
                                FROM (SELECT review_text, created_at
                                      FROM reviews
                                      WHERE product_id = p.product_id
                                      LIMIT 5) r),
                               'No reviews'
                                        ) ||
                       '. Categories: ' || COALESCE(
                               (SELECT string_agg(category_name, ', ')
                                FROM product_categories
                                WHERE product_id = p.product_id),
                               'Uncategorized'
                                           ) AS full_text
                FROM products p
                ORDER BY p.product_id;
                """)

    rows = cur.fetchall()
    product_ids = [row[0] for row in rows]
    full_texts = [row[1] for row in rows]

    cur.execute("""
                WITH embeddings AS (SELECT id, embedding
                                    FROM embed_texts_with_ids(
                                            'embed_anything',
                                            %s,
                                            %s::int[],
                                            %s::text[]
                                         ))
                UPDATE products p
                SET embedding = e.embedding
                FROM embeddings e
                WHERE p.product_id = e.id;
                """, (EMBED_ANYTHING_MODEL, product_ids, full_texts))

    conn.commit()
    cur.close()


def scenario2_distributed_chroma(conn, embed_client: EmbedAnythingDirectClient,
                                 chroma_collection):
    """Scenario 2 - Distributed (Chroma): Fetch data, embed, store in ChromaDB."""
    cur = conn.cursor()

    # Fetch product data with reviews and categories
    cur.execute("""
                SELECT p.product_id,
                       p.name,
                       p.description,
                       p.price,
                       COALESCE((SELECT string_agg(r.review_text, ' | ' ORDER BY r.created_at DESC)
                                 FROM (SELECT review_text, created_at
                                       FROM reviews
                                       WHERE product_id = p.product_id
                                       LIMIT 5) r), 'No reviews'),
                       COALESCE((SELECT string_agg(category_name, ', ')
                                 FROM product_categories
                                 WHERE product_id = p.product_id), 'Uncategorized')
                FROM products p
                ORDER BY p.product_id
                """)

    rows = cur.fetchall()
    product_ids = []
    documents = []

    for row in rows:
        pid, name, desc, price, reviews, categories = row
        product_ids.append(pid)
        doc = f"{name}. {desc}. Price: ${price}. Reviews: {reviews}. Categories: {categories}"
        documents.append(doc)

    # Generate embeddings and store
    embeddings = embed_client.embed(documents)
    chroma_collection.add(
        ids=[str(pid) for pid in product_ids],
        embeddings=embeddings,
        documents=documents
    )

    cur.close()


def scenario2_distributed_qdrant(conn, embed_client: EmbedAnythingDirectClient,
                                 qdrant_client: QdrantClient):
    """Scenario 2 - Distributed (Qdrant): Fetch data, embed, store in Qdrant."""
    cur = conn.cursor()

    # Fetch product data
    cur.execute("""
                SELECT p.product_id,
                       p.name,
                       p.description,
                       p.price,
                       COALESCE((SELECT string_agg(r.review_text, ' | ' ORDER BY r.created_at DESC)
                                 FROM (SELECT review_text, created_at
                                       FROM reviews
                                       WHERE product_id = p.product_id
                                       LIMIT 5) r), 'No reviews'),
                       COALESCE((SELECT string_agg(category_name, ', ')
                                 FROM product_categories
                                 WHERE product_id = p.product_id), 'Uncategorized')
                FROM products p
                ORDER BY p.product_id
                """)

    rows = cur.fetchall()
    product_ids = []
    documents = []

    for row in rows:
        pid, name, desc, price, reviews, categories = row
        product_ids.append(pid)
        doc = f"{name}. {desc}. Price: ${price}. Reviews: {reviews}. Categories: {categories}"
        documents.append(doc)

    # Generate embeddings and store
    embeddings = embed_client.embed(documents)
    points = [
        PointStruct(id=pid, vector=embedding, payload={"text": doc})
        for pid, embedding, doc in zip(product_ids, embeddings, documents)
    ]
    qdrant_client.upsert(collection_name="products", points=points, wait=True)

    cur.close()


# =============================================================================
# SCENARIO 1 FUNCTIONS: Cold Start (insert + embedding generation)
# =============================================================================

def scenario1_unified(conn, products: List[dict]):
    """Scenario 1 - Unified: Insert data and generate embeddings in PostgreSQL."""
    cur = conn.cursor()

    # Insert relational data
    batch_insert_products(cur, products)

    # Fetch context data to generate embeddings
    cur.execute("""
                SELECT p.product_id,
                       p.name || '. ' || p.description ||
                       '. Price: $' || p.price::text ||
                       '. Reviews: ' || COALESCE(
                               (SELECT string_agg(r.review_text, ' | ' ORDER BY r.created_at DESC)
                                FROM (SELECT review_text, created_at
                                      FROM reviews
                                      WHERE product_id = p.product_id
                                      LIMIT 5) r),
                               'No reviews'
                                        ) ||
                       '. Categories: ' || COALESCE(
                               (SELECT string_agg(category_name, ', ')
                                FROM product_categories
                                WHERE product_id = p.product_id),
                               'Uncategorized'
                                           ) AS full_text
                FROM products p
                ORDER BY p.product_id;
                """)

    rows = cur.fetchall()
    product_ids = [row[0] for row in rows]
    full_texts = [row[1] for row in rows]

    cur.execute("""
                WITH embeddings AS (SELECT id, embedding
                                    FROM embed_texts_with_ids(
                                            'embed_anything',
                                            %s,
                                            %s::int[],
                                            %s::text[]
                                         ))
                UPDATE products p
                SET embedding = e.embedding
                FROM embeddings e
                WHERE p.product_id = e.id;
                """, (EMBED_ANYTHING_MODEL, product_ids, full_texts))

    conn.commit()
    cur.close()


def scenario1_distributed_chroma(conn, products: List[dict],
                                 embed_client: EmbedAnythingDirectClient,
                                 chroma_collection):
    """Scenario 1 - Distributed (Chroma): Insert data into PG, embed from app data, store in Chroma."""
    cur = conn.cursor()

    # Insert relational data
    product_ids = batch_insert_products(cur, products)
    conn.commit()

    # Build context from app-level data
    documents = [build_embedding_context(p) for p in products]

    # Generate embeddings and store
    embeddings = embed_client.embed(documents)
    chroma_collection.add(
        ids=[str(pid) for pid in product_ids],
        embeddings=embeddings,
        documents=documents
    )

    cur.close()


def scenario1_distributed_qdrant(conn, products: List[dict],
                                 embed_client: EmbedAnythingDirectClient,
                                 qdrant_client: QdrantClient):
    """Scenario 1 - Distributed (Qdrant): Insert data into PG, embed from app data, store in Qdrant."""
    cur = conn.cursor()

    # Insert relational data
    product_ids = batch_insert_products(cur, products)
    conn.commit()

    # Build context from app-level data
    documents = [build_embedding_context(p) for p in products]

    # Generate embeddings and store
    embeddings = embed_client.embed(documents)
    points = [
        PointStruct(id=pid, vector=embedding, payload={"text": doc})
        for pid, embedding, doc in zip(product_ids, embeddings, documents)
    ]
    qdrant_client.upsert(collection_name="products", points=points, wait=True)

    cur.close()


# =============================================================================
# Benchmark Runners
# =============================================================================

def run_scenario1_unified(products: List[dict], runs: int) -> List[BenchmarkResult]:
    """Run scenario 1 unified: cold start, insert + embed in PostgreSQL."""
    conn, pg_pid = connect_and_get_pid()
    py_pid = os.getpid()

    try:
        # Warm up the PG connection (loads pg_gembed, JIT, etc.)
        warmup_pg_connection(conn)

        # Method-specific warm-up
        warmup_products = generate_products(8)
        truncate_pg_tables(conn)
        scenario1_unified(conn, warmup_products)

        # Run benchmark
        results = []
        for _ in range(runs):
            truncate_pg_tables(conn)
            elapsed, _, stats = ResourceMonitor.measure(
                py_pid, pg_pid,
                lambda: scenario1_unified(conn, products)
            )
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))

        return results
    finally:
        conn.close()


def run_scenario1_distributed_chroma(products: List[dict],
                                     embed_client: EmbedAnythingDirectClient,
                                     runs: int) -> List[BenchmarkResult]:
    """Run scenario 1 distributed (Chroma)."""
    py_pid = os.getpid()
    conn, pg_pid = connect_and_get_pid()

    try:
        # Warm up the PG connection
        warmup_pg_connection(conn)

        # Method-specific warm-up
        warmup_products = generate_products(8)
        client, collection, db_path = create_chroma_client(embed_fn=embed_client.embed)
        try:
            truncate_pg_tables(conn)
            scenario1_distributed_chroma(conn, warmup_products, embed_client, collection)
        finally:
            cleanup_chroma(client, db_path)

        # Run benchmark
        results = []
        for _ in range(runs):
            client, collection, db_path = create_chroma_client(embed_fn=embed_client.embed)
            try:
                truncate_pg_tables(conn)
                elapsed, _, stats = ResourceMonitor.measure(
                    py_pid, pg_pid,
                    lambda: scenario1_distributed_chroma(conn, products, embed_client, collection)
                )
                results.append(BenchmarkResult(time_s=elapsed, stats=stats))
            finally:
                cleanup_chroma(client, db_path)

        return results
    finally:
        conn.close()


def run_scenario1_distributed_qdrant(products: List[dict],
                                     embed_client: EmbedAnythingDirectClient,
                                     runs: int) -> List[BenchmarkResult]:
    """Run scenario 1 distributed (Qdrant)."""
    py_pid = os.getpid()
    conn, pg_pid = connect_and_get_pid()

    try:
        # Warm up the PG connection
        warmup_pg_connection(conn)

        # Method-specific warm-up
        warmup_products = generate_products(8)
        client = create_qdrant_client()
        try:
            truncate_pg_tables(conn)
            scenario1_distributed_qdrant(conn, warmup_products, embed_client, client)
        finally:
            cleanup_qdrant(client)

        # Run benchmark
        results = []
        for _ in range(runs):
            client = create_qdrant_client()
            try:
                truncate_pg_tables(conn)
                elapsed, _, stats = ResourceMonitor.measure(
                    py_pid, pg_pid,
                    lambda: scenario1_distributed_qdrant(conn, products, embed_client, client),
                    container_name=QDRANT_CONTAINER_NAME
                )
                results.append(BenchmarkResult(time_s=elapsed, stats=stats))
            finally:
                cleanup_qdrant(client)

        return results
    finally:
        conn.close()


def run_scenario2_unified(products: List[dict], runs: int) -> List[BenchmarkResult]:
    """Run scenario 2 unified: pre-existing data."""
    conn, pg_pid = connect_and_get_pid()
    py_pid = os.getpid()

    try:
        # Setup
        truncate_pg_tables(conn)
        insert_product_data(conn, products)

        # Warm up the PG connection
        warmup_pg_connection(conn)
        scenario2_unified(conn)
        clear_embeddings(conn)

        # Run benchmark
        results = []
        for _ in range(runs):
            clear_embeddings(conn)
            elapsed, _, stats = ResourceMonitor.measure(
                py_pid, pg_pid,
                lambda: scenario2_unified(conn)
            )
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))

        return results
    finally:
        conn.close()


def run_scenario2_distributed_chroma(products: List[dict],
                                     embed_client: EmbedAnythingDirectClient,
                                     runs: int) -> List[BenchmarkResult]:
    """Run scenario 2 distributed (Chroma)."""
    py_pid = os.getpid()
    conn, pg_pid = connect_and_get_pid()

    try:
        # Setup
        truncate_pg_tables(conn)
        insert_product_data(conn, products)

        # Warm up the PG connection
        warmup_pg_connection(conn)
        client, collection, db_path = create_chroma_client(embed_fn=embed_client.embed)
        try:
            scenario2_distributed_chroma(conn, embed_client, collection)
        finally:
            cleanup_chroma(client, db_path)

        # Run benchmark
        results = []
        for _ in range(runs):
            client, collection, db_path = create_chroma_client(embed_fn=embed_client.embed)
            try:
                elapsed, _, stats = ResourceMonitor.measure(
                    py_pid, pg_pid,
                    lambda: scenario2_distributed_chroma(conn, embed_client, collection)
                )
                results.append(BenchmarkResult(time_s=elapsed, stats=stats))
            finally:
                cleanup_chroma(client, db_path)

        return results
    finally:
        conn.close()


def run_scenario2_distributed_qdrant(products: List[dict],
                                     embed_client: EmbedAnythingDirectClient,
                                     runs: int) -> List[BenchmarkResult]:
    """Run scenario 2 distributed (Qdrant)."""
    py_pid = os.getpid()
    conn, pg_pid = connect_and_get_pid()

    try:
        # Setup
        truncate_pg_tables(conn)
        insert_product_data(conn, products)

        # Warm up the PG connection
        warmup_pg_connection(conn)
        client = create_qdrant_client()
        try:
            scenario2_distributed_qdrant(conn, embed_client, client)
        finally:
            cleanup_qdrant(client)

        # Run benchmark
        results = []
        for _ in range(runs):
            client = create_qdrant_client()
            try:
                elapsed, _, stats = ResourceMonitor.measure(
                    py_pid, pg_pid,
                    lambda: scenario2_distributed_qdrant(conn, embed_client, client),
                    container_name=QDRANT_CONTAINER_NAME
                )
                results.append(BenchmarkResult(time_s=elapsed, stats=stats))
            finally:
                cleanup_qdrant(client)

        return results
    finally:
        conn.close()


# =============================================================================
# Output Functions
# =============================================================================

def print_header(title):
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def print_detailed_header():
    """Print detailed benchmark results header."""
    lbl_w, time_w, col_w, med_w = 24, 14, 13, 7

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
        "  {label:<24}{med:>7} | {time:>14} | "
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
# Main Entry Point
# =============================================================================

def main():
    # Initialize PostgreSQL schema
    conn, _ = connect_and_get_pid()
    setup_pg_database(conn)
    conn.close()

    # Initialize embedding client
    embed_client = EmbedAnythingDirectClient()

    all_results_s1 = []
    all_results_s2 = []

    # ==========================================================================
    # Scenario 1: Cold Start (insert + embedding generation)
    # ==========================================================================
    print_header("SCENARIO 1: Cold Start (insert + embedding generation)")
    print_detailed_header()

    for size in TEST_SIZES:
        products = generate_products(size)
        print(f"Size: {size}", flush=True)

        unified_results = run_scenario1_unified(products, RUNS_PER_SIZE)
        chroma_results = run_scenario1_distributed_chroma(products, embed_client, RUNS_PER_SIZE)
        qdrant_results = run_scenario1_distributed_qdrant(products, embed_client, RUNS_PER_SIZE)

        print_result("Unified", unified_results)
        print_result("Distributed (ChromaDB)", chroma_results)
        print_result("Distributed (Qdrant)", qdrant_results)
        print()

        all_results_s1.append({
            'size': size,
            'unified': compute_metrics(size, unified_results),
            'dist_chroma': compute_metrics(size, chroma_results),
            'dist_qdrant': compute_metrics(size, qdrant_results),
        })

    # ==========================================================================
    # Scenario 2: Pre-existing Data (embedding generation only)
    # ==========================================================================
    print_header("SCENARIO 2: Pre-existing Data (embedding generation only)")
    print_detailed_header()

    for size in TEST_SIZES:
        products = generate_products(size)
        print(f"Size: {size}", flush=True)

        unified_results = run_scenario2_unified(products, RUNS_PER_SIZE)
        chroma_results = run_scenario2_distributed_chroma(products, embed_client, RUNS_PER_SIZE)
        qdrant_results = run_scenario2_distributed_qdrant(products, embed_client, RUNS_PER_SIZE)

        print_result("Unified", unified_results)
        print_result("Distributed (ChromaDB)", chroma_results)
        print_result("Distributed (Qdrant)", qdrant_results)
        print()

        all_results_s2.append({
            'size': size,
            'unified': compute_metrics(size, unified_results),
            'dist_chroma': compute_metrics(size, chroma_results),
            'dist_qdrant': compute_metrics(size, qdrant_results),
        })

    # Save results to CSV and generate plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    methods = ['unified', 'dist_chroma', 'dist_qdrant']

    # Scenario 1
    s1_dir = OUTPUT_DIR / "scenario1"
    save_results_csv(all_results_s1, s1_dir, timestamp, methods)
    generate_plots(all_results_s1, s1_dir, timestamp, methods)

    # Scenario 2
    s2_dir = OUTPUT_DIR / "scenario2"
    save_results_csv(all_results_s2, s2_dir, timestamp, methods)
    generate_plots(all_results_s2, s2_dir, timestamp, methods)


if __name__ == "__main__":
    main()
