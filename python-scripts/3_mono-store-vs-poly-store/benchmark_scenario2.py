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
# Data Generation
# =============================================================================

def generate_products(n: int) -> List[dict]:
    from data.loader import get_review_texts
    categories_pool = ["Electronics", "Home & Garden", "Sports", "Books", "Clothing",
                       "Toys", "Health", "Automotive", "Food", "Office"]
    adjectives = ["Premium", "Professional", "Essential", "Advanced", "Classic"]
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
                """)
    conn.commit()
    cur.close()


def clear_embeddings(conn):
    cur = conn.cursor()
    cur.execute("UPDATE products SET embedding = NULL;")
    conn.commit()
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
                         INSERT INTO products (name, description, price, stock_count)
                             SELECT name, description, price, stock FROM input_products
                             RETURNING product_id, name),
                     product_mapping AS (SELECT p.product_id, i.ord
                                         FROM inserted_products p
                                                  JOIN input_products i ON p.name = i.name),
                     inserted_reviews AS (
                         INSERT INTO reviews (product_id, rating, review_text)
                             SELECT m.product_id, r.rating, r.text
                             FROM unnest(%s::int[], %s::int[], %s::text[]) AS r(p_idx, rating, text)
                                      JOIN product_mapping m ON r.p_idx = m.ord),
                     inserted_categories AS (
                         INSERT INTO product_categories (product_id, category_name)
                             SELECT m.product_id, c.cat_name
                             FROM unnest(%s::int[], %s::text[]) AS c(p_idx, cat_name)
                                      JOIN product_mapping m ON c.p_idx = m.ord)
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
    conn.commit()
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
    collection = client.create_collection("products", embedding_function=emb_obj, configuration=configuration)
    return client, collection, db_path


def cleanup_chroma(client, db_path: str):
    del client
    time.sleep(0.2)
    if os.path.exists(db_path):
        shutil.rmtree(db_path)


def create_qdrant_client():
    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists("products"):
        client.delete_collection("products")
    hnsw_config = models.HnswConfigDiff(m=16, ef_construct=100)
    client.create_collection("products",
                             vectors_config=VectorParams(size=384, distance=Distance.COSINE,
                                                         hnsw_config=hnsw_config))
    return client


def cleanup_qdrant(client):
    if client.collection_exists("products"):
        client.delete_collection("products")
    client.close()
    time.sleep(0.1)


# =============================================================================
# Scenario 2 Benchmark Functions
# =============================================================================

def scenario2_mono_store(conn):
    """Generate embeddings for pre-existing data in PG."""
    cur = conn.cursor()
    cur.execute("""
                WITH product_data AS (SELECT p.product_id,
                                             p.name || '. ' || p.description ||
                                             '. Price: $' || p.price::text ||
                                             '. Reviews: ' || COALESCE(
                                                     (SELECT string_agg(r.review_text, ' | ' ORDER BY r.created_at DESC)
                                                      FROM (SELECT review_text, created_at
                                                            FROM reviews
                                                            WHERE product_id = p.product_id
                                                            LIMIT 5) r), 'No reviews') ||
                                             '. Categories: ' || COALESCE(
                                                     (SELECT string_agg(category_name, ', ')
                                                      FROM product_categories
                                                      WHERE product_id = p.product_id), 'Uncategorized') AS full_text
                                      FROM products p),
                     embeddings AS (SELECT id, embedding
                                    FROM embed_texts_with_ids(
                                            'embed_anything', %s,
                                            (SELECT array_agg(product_id ORDER BY product_id) FROM product_data)::int[],
                                            (SELECT array_agg(full_text ORDER BY product_id) FROM product_data)::text[]))
                UPDATE products p
                SET embedding = e.embedding
                FROM embeddings e
                WHERE p.product_id = e.id;
                """, (EMBED_ANYTHING_MODEL,))
    cur.execute("CREATE INDEX ON products USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 100);")
    conn.commit()
    cur.close()


def scenario2_poly_store_chroma(conn, embed_client, chroma_collection):
    """Fetch data from PG, embed, store in ChromaDB."""
    cur = conn.cursor()
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
    product_ids, documents = [], []
    for row in rows:
        pid, name, desc, price, reviews, categories = row
        product_ids.append(pid)
        documents.append(f"{name}. {desc}. Price: ${price}. Reviews: {reviews}. Categories: {categories}")
    embeddings = embed_client.embed(documents)
    chroma_collection.add(ids=[str(pid) for pid in product_ids], embeddings=embeddings, documents=documents)
    cur.close()


def scenario2_poly_store_qdrant(conn, embed_client, qdrant_client):
    """Fetch data from PG, embed, store in Qdrant."""
    cur = conn.cursor()
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
    product_ids, documents = [], []
    for row in rows:
        pid, name, desc, price, reviews, categories = row
        product_ids.append(pid)
        documents.append(f"{name}. {desc}. Price: ${price}. Reviews: {reviews}. Categories: {categories}")
    embeddings = embed_client.embed(documents)
    points = [PointStruct(id=pid, vector=emb, payload={"text": doc})
              for pid, emb, doc in zip(product_ids, embeddings, documents)]
    qdrant_client.upsert(collection_name="products", points=points, wait=True)
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
    methods = ['pg_mono_deferred', 'poly_chroma', 'poly_qdrant']

    print(f"\nStarting Benchmark 3 - Scenario 2: Pre-existing Data")
    print(f"Run ID: {run_id}")
    print(f"Sizes: {test_sizes}")

    embed_client = EmbedAnythingDirectClient()
    py_pid = os.getpid()

    # Warm-up
    print("Warming up...")
    warmup_products = generate_products(8)
    conn_w, _ = connect_and_get_pid()
    warmup_pg_connection(conn_w)
    setup_pg_database(conn_w)
    insert_product_data(conn_w, warmup_products)
    scenario2_mono_store(conn_w)
    clear_embeddings(conn_w)
    conn_w.close()

    # Pre-generate test data
    test_products = {size: generate_products(size) for size in test_sizes}

    # Single run over all sizes
    results_by_size = {size: {m: None for m in methods} for size in test_sizes}

    for size in test_sizes:
        print(f"\nSize: {size}", flush=True)
        products = test_products[size]

        conn_mono, pg_pid_mono = connect_and_get_pid()
        warmup_pg_connection(conn_mono)
        setup_pg_database(conn_mono)
        insert_product_data(conn_mono, products)

        conn_chroma, pg_pid_chroma = connect_and_get_pid()
        warmup_pg_connection(conn_chroma)
        setup_pg_database(conn_chroma)
        insert_product_data(conn_chroma, products)

        conn_qdrant, pg_pid_qdrant = connect_and_get_pid()
        warmup_pg_connection(conn_qdrant)
        setup_pg_database(conn_qdrant)
        insert_product_data(conn_qdrant, products)

        try:
            # Mono-Store
            elapsed, _, stats = ResourceMonitor.measure(
                py_pid, pg_pid_mono,
                lambda: scenario2_mono_store(conn_mono))
            results_by_size[size]['pg_mono_deferred'] = BenchmarkResult(elapsed, stats)
            print(f"  pg_mono_deferred: {elapsed:.2f}s", flush=True)

            # Poly-Store Chroma
            client_c, col_c, path_c = create_chroma_client(embed_fn=embed_client.embed)
            try:
                elapsed, _, stats = ResourceMonitor.measure(
                    py_pid, pg_pid_chroma,
                    lambda: scenario2_poly_store_chroma(conn_chroma, embed_client, col_c))
                results_by_size[size]['poly_chroma'] = BenchmarkResult(elapsed, stats)
                print(f"  poly_chroma: {elapsed:.2f}s", flush=True)
            finally:
                cleanup_chroma(client_c, path_c)

            # Poly-Store Qdrant
            qd = create_qdrant_client()
            try:
                elapsed, _, stats = ResourceMonitor.measure(
                    py_pid, pg_pid_qdrant,
                    lambda: scenario2_poly_store_qdrant(conn_qdrant, embed_client, qd))
                results_by_size[size]['poly_qdrant'] = BenchmarkResult(elapsed, stats)
                print(f"  poly_qdrant: {elapsed:.2f}s", flush=True)
            finally:
                cleanup_qdrant(qd)
        finally:
            conn_mono.close()
            conn_chroma.close()
            conn_qdrant.close()

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

    output_dir = OUTPUT_DIR / "scenario2"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_single_run_csv(all_results, output_dir, run_id, methods)
    print("Run completed!")


if __name__ == "__main__":
    main()
