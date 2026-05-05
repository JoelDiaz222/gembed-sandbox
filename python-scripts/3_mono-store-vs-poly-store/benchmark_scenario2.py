#!/usr/bin/env python3
"""
Benchmark 3 - Scenario 2: Mono-Store vs Poly-Store, Pre-existing Data
Data already exists in PG; only generates/stores embeddings.
"""
import argparse
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, List

import chromadb
import numpy as np
from pgvector.psycopg2 import register_vector
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct
from utils.benchmark_utils import (
    QDRANT_URL, QDRANT_CONTAINER_NAME, EMBED_ANYTHING_MODEL,
    BenchmarkResult, ResourceMonitor,
    EmbedAnythingDirectClient, EmbeddingWrapper,
    connect_and_get_pid, warmup_pg_connection, clear_model_cache,
)
from utils.plot_utils import save_single_run_csv

OUTPUT_DIR = Path(__file__).parent / "output"
CHROMA_MAX_SIZE = 4096

PRODUCT_TEXT_QUERY = """
                     WITH review_texts AS (SELECT product_id,
                                                  string_agg(review_text, ' | ' ORDER BY review_id) AS reviews_text
                                           FROM (SELECT product_id,
                                                        review_id,
                                                        review_text,
                                                        row_number() OVER (PARTITION BY product_id ORDER BY review_id) AS rn
                                                 FROM review) r
                                           WHERE rn <= 5
                                           GROUP BY product_id),
                          category_texts AS (SELECT product_id,
                                                    string_agg(category_name, ', ' ORDER BY category_name) AS categories_text
                                             FROM product_category
                                             GROUP BY product_id)
                     SELECT p.product_id,
                            p.name || '. ' || p.description ||
                            '. Price: $' || to_char(p.price, 'FM999999990.00') ||
                            '. Reviews: ' || COALESCE(r.reviews_text, 'No reviews') ||
                            '. Categories: ' || COALESCE(c.categories_text, 'Uncategorized') AS full_text
                     FROM product p
                              LEFT JOIN review_texts r ON r.product_id = p.product_id
                              LEFT JOIN category_texts c ON c.product_id = p.product_id
                     ORDER BY p.product_id
                     """


# =============================================================================
# Data Generation
# =============================================================================

def generate_products(n: int) -> List[dict]:
    from data.loader import get_review_texts
    categories_pool = ["Electronics", "Home & Garden", "Sports", "Books", "Clothing",
                       "Toys", "Health", "Automotive", "Food", "Office"]
    adjectives = ["Premium", "Professional", "Essential", "Advanced", "Classic"]
    total_reviews_needed = int(n * 3.5) + 100
    reviews_pool = get_review_texts(total_reviews_needed, shuffle=True)
    review_idx = 0
    products = []
    for i in range(n):
        adj = adjectives[i % len(adjectives)]
        category_main = categories_pool[i % len(categories_pool)]
        product = {
            'name': f"{adj} {category_main} Item {i}",
            'description': f"High-quality {category_main.lower()} product featuring "
                           f"durable construction and modern design.",
            'price': 19.99 + (i % 10) * 10,
            'stock': 50 + (i * 7) % 200,
            'reviews': [],
            'categories': [],
        }
        num_reviews = 2 + (i % 4)
        for j in range(num_reviews):
            product['reviews'].append({'rating': 3 + (j % 3),
                                       'text': reviews_pool[review_idx % len(reviews_pool)]})
            review_idx += 1
        num_cats = 1 + (i % 3)
        product['categories'] = [categories_pool[(i + k) % len(categories_pool)] for k in range(num_cats)]
        products.append(product)
    return products


# =============================================================================
# PostgreSQL Functions
# =============================================================================

def setup_pg_database(conn):
    cur = conn.cursor()
    cur.execute("""
                CREATE
                EXTENSION IF NOT EXISTS vector;
                CREATE
                EXTENSION IF NOT EXISTS pg_gembed;
                DROP TABLE IF EXISTS product_category CASCADE;
                DROP TABLE IF EXISTS review CASCADE;
                DROP TABLE IF EXISTS product CASCADE;
                CREATE TABLE product
                (
                    product_id  SERIAL PRIMARY KEY,
                    name        TEXT NOT NULL,
                    description TEXT NOT NULL,
                    price       DECIMAL(10, 2),
                    stock_count INTEGER DEFAULT 0,
                    embedding   vector(384)
                );
                CREATE TABLE review
                (
                    review_id   SERIAL PRIMARY KEY,
                    product_id  INTEGER REFERENCES product (product_id),
                    rating      INTEGER CHECK (rating BETWEEN 1 AND 5),
                    review_text TEXT,
                    created_at  TIMESTAMP DEFAULT NOW()
                );
                CREATE TABLE product_category
                (
                    product_id    INTEGER REFERENCES product (product_id),
                    category_name TEXT,
                    PRIMARY KEY (product_id, category_name)
                );
                """)
    cur.close()


def clear_embeddings(conn):
    cur = conn.cursor()
    cur.execute("UPDATE product SET embedding = NULL;")
    cur.close()


def batch_insert_products(cur, products: List[dict]) -> List[int]:
    names = [p['name'] for p in products]
    descriptions = [p['description'] for p in products]
    prices = [p['price'] for p in products]
    stocks = [p['stock'] for p in products]
    review_p_indices, review_ratings, review_texts = [], [], []
    for i, p in enumerate(products):
        for r in p['reviews']:
            review_p_indices.append(i + 1)
            review_ratings.append(r['rating'])
            review_texts.append(r['text'])
    category_p_indices, category_names = [], []
    for i, p in enumerate(products):
        for c in p['categories']:
            category_p_indices.append(i + 1)
            category_names.append(c)
    cur.execute("""
                WITH input_products AS (SELECT name, description, price, stock, ord
                                        FROM unnest(%s::text[], %s::text[], %s::numeric[], %s::int[])
                                                 WITH ORDINALITY AS i(name, description, price, stock, ord)),
                     inserted_products AS (
                INSERT
                INTO product (name, description, price, stock_count)
                SELECT name, description, price, stock
                FROM input_products RETURNING product_id, name),
                     product_mapping AS (
                SELECT p.product_id, i.ord
                FROM inserted_products p
                    JOIN input_products i
                ON p.name = i.name),
                    inserted_reviews AS (
                INSERT
                INTO review (product_id, rating, review_text)
                SELECT m.product_id, r.rating, r.text
                FROM unnest(%s:: int [], %s:: int [], %s::text[]) AS r(p_idx, rating, text)
                    JOIN product_mapping m
                ON r.p_idx = m.ord),
                    inserted_categories AS (
                INSERT
                INTO product_category (product_id, category_name)
                SELECT m.product_id, c.cat_name
                FROM unnest(%s:: int [], %s::text[]) AS c (p_idx, cat_name)
                    JOIN product_mapping m
                ON c.p_idx = m.ord)
                SELECT product_id
                FROM product_mapping
                ORDER BY ord;
                """, (names, descriptions, prices, stocks,
                      review_p_indices, review_ratings, review_texts,
                      category_p_indices, category_names))
    return [row[0] for row in cur.fetchall()]


def insert_product_data(conn, products: List[dict]) -> List[int]:
    cur = conn.cursor()
    product_ids = batch_insert_products(cur, products)
    cur.close()
    return product_ids


# =============================================================================
# Vector DB Clients
# =============================================================================

def create_chroma_client(base_path: str = "./chroma_bench3_s2", embed_fn: Callable = None):
    db_path = f"{base_path}_{time.time_ns()}"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    client = chromadb.PersistentClient(path=db_path)
    configuration = {"hnsw": {"space": "cosine", "max_neighbors": 16, "ef_construction": 100}}
    emb_obj = EmbeddingWrapper(embed_fn)
    collection = client.create_collection("product", embedding_function=emb_obj, configuration=configuration)
    return client, collection, db_path


def cleanup_chroma(client, db_path: str):
    del client
    time.sleep(0.2)
    if os.path.exists(db_path):
        shutil.rmtree(db_path)


def create_qdrant_client():
    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists("product"):
        client.delete_collection("product")
    hnsw_config = models.HnswConfigDiff(m=16, ef_construct=100)
    client.create_collection("product",
                             vectors_config=VectorParams(size=384, distance=Distance.COSINE,
                                                         hnsw_config=hnsw_config))
    client.update_collection("product", optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0))
    return client


def cleanup_qdrant(client):
    if client.collection_exists("product"):
        client.delete_collection("product")
    client.close()
    time.sleep(0.1)


# =============================================================================
# Scenario 2 Benchmark Functions
# =============================================================================

def scenario2_mono_store(conn, create_index: bool = True):
    """Generate embeddings for pre-existing data in PG."""
    cur = conn.cursor()
    cur.execute(f"""
                WITH product_data AS ({PRODUCT_TEXT_QUERY}),
                     embeddings AS (SELECT id, embedding
                                    FROM unnest(
                                            (SELECT array_agg(product_id ORDER BY product_id) FROM product_data)::int[],
                                            embed_texts(
                                                    'embed_anything', %s,
                                                    (SELECT array_agg(full_text ORDER BY product_id) FROM product_data)::text[]
                                            )
                                         ) AS e(id, embedding))
                UPDATE product p
                SET embedding = e.embedding FROM embeddings e
                WHERE p.product_id = e.id;
                """, (EMBED_ANYTHING_MODEL,))
    if create_index:
        cur.execute(
            "CREATE INDEX ON product USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 100);")
    cur.close()


def scenario2_mono_direct(conn, embed_client):
    """Fetch text from PG, generate embeddings from app, store in PG."""
    cur = conn.cursor()
    cur.execute(PRODUCT_TEXT_QUERY)
    rows = cur.fetchall()
    product_ids, documents = [], []
    for row in rows:
        pid, doc = row
        product_ids.append(pid)
        documents.append(doc)

    embeddings = embed_client.embed(documents)

    cur.execute("""
                UPDATE product p
                SET embedding = e.embedding FROM unnest(%s:: int []
                  , %s::vector[]) AS e(id
                  , embedding)
                WHERE p.product_id = e.id;
                """, (product_ids, [np.array(e) for e in embeddings]))

    cur.execute(
        "CREATE INDEX ON product USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 100);")
    cur.close()


def scenario2_poly_store_chroma(conn, embed_client, chroma_collection):
    """Fetch data from PG, embed, store in ChromaDB."""
    cur = conn.cursor()
    cur.execute(PRODUCT_TEXT_QUERY)
    rows = cur.fetchall()
    product_ids, documents = [], []
    for row in rows:
        pid, doc = row
        product_ids.append(pid)
        documents.append(doc)
    embeddings = embed_client.embed(documents)
    chroma_collection.add(ids=[str(pid) for pid in product_ids], embeddings=embeddings)
    cur.close()


def scenario2_poly_store_qdrant(conn, embed_client, qdrant_client):
    """Fetch data from PG, embed, store in Qdrant."""
    cur = conn.cursor()
    cur.execute(PRODUCT_TEXT_QUERY)
    rows = cur.fetchall()
    product_ids, documents = [], []
    for row in rows:
        pid, doc = row
        product_ids.append(pid)
        documents.append(doc)
    embeddings = embed_client.embed(documents)
    points = [PointStruct(id=pid, vector=emb)
              for pid, emb in zip(product_ids, embeddings)]
    qdrant_client.upsert(collection_name="product", points=points, wait=True)
    qdrant_client.update_collection("product", optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000))
    cur.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Benchmark 3 - Scenario 2: Pre-existing Data')
    parser.add_argument('--sizes', type=int, nargs='+', required=True)
    parser.add_argument('--run-id', type=str, default=None)
    args = parser.parse_args()

    test_sizes = args.sizes
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    methods = ['mono_pg_unified_no_index', 'mono_pg_unified_deferred',
               'mono_pg_direct_deferred', 'poly_chroma', 'poly_qdrant_deferred']

    print(f"\nStarting Benchmark 3 - Scenario 2: Pre-existing Data")
    print(f"Run ID: {run_id}")
    print(f"Sizes: {test_sizes}")

    embed_client = EmbedAnythingDirectClient()
    py_pid = os.getpid()

    # Warm-up
    print("Warming up...")
    warmup_products = generate_products(8)
    conn_w, _ = connect_and_get_pid()
    register_vector(conn_w)
    warmup_pg_connection(conn_w)
    setup_pg_database(conn_w)
    insert_product_data(conn_w, warmup_products)
    scenario2_mono_store(conn_w)
    clear_embeddings(conn_w)
    scenario2_mono_direct(conn_w, embed_client)
    clear_embeddings(conn_w)
    conn_w.commit()
    clear_model_cache()
    conn_w.close()

    # Pre-generate test data
    test_products = {size: generate_products(size) for size in test_sizes}

    # Single run over all sizes
    results_by_size = {size: {m: None for m in methods} for size in test_sizes}

    for size in test_sizes:
        print(f"\nSize: {size}", flush=True)
        products = test_products[size]

        # Mono-Store No Index
        conn_mono_no_index, pg_pid_mono_no_index = connect_and_get_pid()
        warmup_pg_connection(conn_mono_no_index)
        setup_pg_database(conn_mono_no_index)
        insert_product_data(conn_mono_no_index, products)
        conn_mono_no_index.commit()
        try:
            elapsed, _, stats = ResourceMonitor.measure(
                py_pid, pg_pid_mono_no_index,
                lambda: scenario2_mono_store(conn_mono_no_index, create_index=False))
            conn_mono_no_index.commit()
            results_by_size[size]['mono_pg_unified_no_index'] = BenchmarkResult(elapsed, stats)
            print(f"  mono_pg_unified_no_index: {elapsed:.2f}s", flush=True)
            clear_model_cache()
        finally:
            conn_mono_no_index.close()

        # Mono-Store
        conn_mono, pg_pid_mono = connect_and_get_pid()
        warmup_pg_connection(conn_mono)
        setup_pg_database(conn_mono)
        insert_product_data(conn_mono, products)
        conn_mono.commit()
        try:
            elapsed, _, stats = ResourceMonitor.measure(
                py_pid, pg_pid_mono,
                lambda: scenario2_mono_store(conn_mono))
            conn_mono.commit()
            results_by_size[size]['mono_pg_unified_deferred'] = BenchmarkResult(elapsed, stats)
            print(f"  mono_pg_unified_deferred: {elapsed:.2f}s", flush=True)
            clear_model_cache()
        finally:
            conn_mono.close()

        # Mono-Store Direct
        conn_direct, pg_pid_direct = connect_and_get_pid()
        register_vector(conn_direct)
        warmup_pg_connection(conn_direct)
        setup_pg_database(conn_direct)
        insert_product_data(conn_direct, products)
        conn_direct.commit()
        embed_client.embed(['warmup'])
        try:
            elapsed, _, stats = ResourceMonitor.measure(
                py_pid, pg_pid_direct,
                lambda: scenario2_mono_direct(conn_direct, embed_client))
            conn_direct.commit()
            results_by_size[size]['mono_pg_direct_deferred'] = BenchmarkResult(elapsed, stats)
            print(f"  mono_pg_direct_deferred: {elapsed:.2f}s", flush=True)
            clear_model_cache()
        finally:
            conn_direct.close()

        # Poly-Store Chroma
        if size <= CHROMA_MAX_SIZE:
            conn_chroma, pg_pid_chroma = connect_and_get_pid()
            warmup_pg_connection(conn_chroma)
            setup_pg_database(conn_chroma)
            insert_product_data(conn_chroma, products)
            conn_chroma.commit()
            embed_client.embed(['warmup'])
            try:
                client_c, col_c, path_c = create_chroma_client(embed_fn=embed_client.embed)
                try:
                    elapsed, _, stats = ResourceMonitor.measure(
                        py_pid, pg_pid_chroma,
                        lambda: scenario2_poly_store_chroma(conn_chroma, embed_client, col_c))
                    conn_chroma.commit()
                    results_by_size[size]['poly_chroma'] = BenchmarkResult(elapsed, stats)
                    print(f"  poly_chroma: {elapsed:.2f}s", flush=True)
                    clear_model_cache()
                finally:
                    cleanup_chroma(client_c, path_c)
            finally:
                conn_chroma.close()
        else:
            print(f"  poly_chroma: skipped (cap {CHROMA_MAX_SIZE})", flush=True)

        # Poly-Store Qdrant
        conn_qdrant, pg_pid_qdrant = connect_and_get_pid()
        warmup_pg_connection(conn_qdrant)
        setup_pg_database(conn_qdrant)
        insert_product_data(conn_qdrant, products)
        conn_qdrant.commit()
        embed_client.embed(['warmup'])
        try:
            qd = create_qdrant_client()
            try:
                elapsed, _, stats = ResourceMonitor.measure(
                    py_pid, pg_pid_qdrant,
                    lambda: scenario2_poly_store_qdrant(conn_qdrant, embed_client, qd))
                conn_qdrant.commit()
                results_by_size[size]['poly_qdrant_deferred'] = BenchmarkResult(elapsed, stats)
                print(f"  poly_qdrant_deferred: {elapsed:.2f}s", flush=True)
                clear_model_cache()
            finally:
                cleanup_qdrant(qd)
        finally:
            conn_qdrant.close()

    # Collect metrics
    all_results = []
    for size in test_sizes:
        entry = {'size': size}
        for method in methods:
            r = results_by_size[size][method]
            if r is None:
                continue
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

    output_dir = OUTPUT_DIR / "scenario2"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_single_run_csv(all_results, output_dir, run_id, methods)
    print("Run completed!")


if __name__ == "__main__":
    main()
