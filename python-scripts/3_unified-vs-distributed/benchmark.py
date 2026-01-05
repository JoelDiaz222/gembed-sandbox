import csv
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Callable, List

import chromadb
import embed_anything
import matplotlib.pyplot as plt
import psutil
import psycopg2
from embed_anything import EmbeddingModel, WhichModel

# Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'joeldiaz',
    'user': 'joeldiaz',
}

TEST_SIZES = [16, 32, 64, 128, 256, 512]
EMBED_ANYTHING_MODEL = "Qdrant/all-MiniLM-L6-v2-onnx"
RUNS_PER_SIZE = 3

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output"

# Global model cache
model_cache = {}


@dataclass
class ResourceStats:
    delta_mb: float
    peak_mb: float
    cpu_usage: float
    sys_peak_mb: float
    sys_cpu_usage: float


@dataclass
class BenchmarkResult:
    time_s: float
    stats: ResourceStats


class ResourceMonitor:
    def __init__(self, pid: int):
        self.pid = pid
        self.process = psutil.Process(pid)

        try:
            mem_info = self.process.memory_full_info()
            self.baseline = mem_info.uss if hasattr(mem_info, 'uss') else mem_info.rss
        except (psutil.AccessDenied, AttributeError):
            mem_info = self.process.memory_info()
            self.baseline = mem_info.rss

        self.process.cpu_percent()
        time.sleep(0.1)

    @staticmethod
    def measure(pid: int, func: Callable):
        """Measure resource usage during function execution."""
        monitor = ResourceMonitor(pid)
        result = func()

        try:
            mem_info = monitor.process.memory_full_info()
            peak = mem_info.uss if hasattr(mem_info, 'uss') else mem_info.rss
        except (psutil.AccessDenied, AttributeError):
            mem_info = monitor.process.memory_info()
            peak = mem_info.rss

        cpu_usage = monitor.process.cpu_percent()

        baseline_mb = monitor.baseline / (1024 * 1024)
        peak_mb = peak / (1024 * 1024)
        delta_mb = peak_mb - baseline_mb

        sys_mem = psutil.virtual_memory()
        sys_peak_mb = sys_mem.used / (1024 * 1024)
        sys_cpu_usage = psutil.cpu_percent()

        stats = ResourceStats(
            delta_mb=delta_mb,
            peak_mb=peak_mb,
            cpu_usage=cpu_usage,
            sys_peak_mb=sys_peak_mb,
            sys_cpu_usage=sys_cpu_usage
        )

        return result, stats


class EmbedAnythingDirectClient:
    """Direct Python client for EmbedAnything."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings directly via Python."""
        model = self._get_model(EMBED_ANYTHING_MODEL)
        data = embed_anything.embed_query(texts, embedder=model)
        return [item.embedding for item in data]

    @staticmethod
    def _get_model(model_name: str):
        """Get or load model from cache."""
        if model_name not in model_cache:
            model_cache[model_name] = EmbeddingModel.from_pretrained_onnx(
                WhichModel.Bert,
                hf_model_id=model_name
            )
        return model_cache[model_name]


# --- Data Generation ---

def generate_products(n: int) -> List[dict]:
    """Generate realistic product data with reviews and categories."""
    categories_pool = [
        "Electronics", "Home & Garden", "Sports", "Books", "Clothing",
        "Toys", "Health", "Automotive", "Food", "Office"
    ]

    adjectives = ["Premium", "Professional", "Essential", "Advanced", "Classic"]
    review_templates = [
        "Great product, exactly what I needed for {}.",
        "Works well for {}. Good value for money.",
        "Perfect for {} applications. Highly recommend.",
        "Decent quality for {}. Met my expectations.",
        "Excellent {} solution. Fast shipping too.",
    ]

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

        # Generate 2-5 reviews per product
        num_reviews = 2 + (i % 4)
        for j in range(num_reviews):
            use_case = ["daily tasks", "work projects", "home improvement",
                        "outdoor activities", "creative projects"][j % 5]
            product['reviews'].append({
                'rating': 3 + (j % 3),
                'text': review_templates[j % len(review_templates)].format(use_case)
            })

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


# --- PostgreSQL Functions ---

def connect_and_get_pid():
    """Connect to PostgreSQL and get backend PID."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("SELECT pg_backend_pid();")
    pid = cur.fetchone()[0]
    cur.close()
    return conn, pid


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


def insert_product_data(conn, products: List[dict]) -> List[int]:
    """Insert product relational data (without embeddings)."""
    cur = conn.cursor()
    product_ids = []

    for p in products:
        cur.execute(
            """INSERT INTO products (name, description, price, stock_count)
               VALUES (%s, %s, %s, %s)
               RETURNING product_id""",
            (p['name'], p['description'], p['price'], p['stock'])
        )
        pid = cur.fetchone()[0]
        product_ids.append(pid)

        for review in p['reviews']:
            cur.execute(
                """INSERT INTO reviews (product_id, rating, review_text)
                   VALUES (%s, %s, %s)""",
                (pid, review['rating'], review['text'])
            )

        for category in p['categories']:
            cur.execute(
                """INSERT INTO product_categories (product_id, category_name)
                   VALUES (%s, %s)""",
                (pid, category)
            )

    conn.commit()
    cur.close()
    return product_ids


# --- ChromaDB Functions ---

def create_chroma_client(base_path: str = "./chroma_bench"):
    """Create a fresh ChromaDB persistent client."""
    db_path = f"{base_path}_{time.time_ns()}"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    client = chromadb.PersistentClient(path=db_path)
    collection = client.create_collection(
        "products",
        metadata={"hnsw:space": "cosine"}
    )
    return client, collection, db_path


def cleanup_chroma(client, db_path: str):
    """Clean up ChromaDB client and files."""
    del client
    time.sleep(0.2)
    if os.path.exists(db_path):
        shutil.rmtree(db_path)


# =============================================================================
# SCENARIO 2 FUNCTIONS: Pre-existing Data (embedding generation only)
# =============================================================================

def scenario2_unified(conn) -> float:
    """
    Scenario 2 - Unified: Generate embeddings for pre-existing data.
    Uses SQL JOIN + pg_gembed. Data already exists in PostgreSQL.
    """
    cur = conn.cursor()
    start = time.perf_counter()

    cur.execute("""
                WITH context AS (SELECT p.product_id,
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
                                 FROM products p),
                     embeddings AS (SELECT sentence_id, embedding
                                    FROM embed_texts_with_ids(
                                            'embed_anything',
                                            %s,
                                            (SELECT array_agg(product_id ORDER BY product_id) FROM context),
                                            (SELECT array_agg(full_text ORDER BY product_id) FROM context)
                                         ))
                UPDATE products p
                SET embedding = e.embedding
                FROM embeddings e
                WHERE p.product_id = e.sentence_id;
                """, (EMBED_ANYTHING_MODEL,))

    conn.commit()
    elapsed = time.perf_counter() - start
    cur.close()
    return elapsed


def scenario2_distributed(conn, embed_client: EmbedAnythingDirectClient,
                          chroma_collection) -> float:
    """
    Scenario 2 - Distributed: Fetch data from PostgreSQL, embed, store in ChromaDB.
    Data already exists in PostgreSQL, need to fetch and build context.
    """
    cur = conn.cursor()
    start = time.perf_counter()

    # Fetch product data with reviews and categories (simulating the JOIN)
    cur.execute("""
                SELECT p.product_id,
                       p.name,
                       p.description,
                       p.price,
                       COALESCE(
                               (SELECT string_agg(r.review_text, ' | ' ORDER BY r.created_at DESC)
                                FROM (SELECT review_text, created_at
                                      FROM reviews
                                      WHERE product_id = p.product_id
                                      LIMIT 5) r),
                               'No reviews'
                       ) AS reviews,
                       COALESCE(
                               (SELECT string_agg(category_name, ', ')
                                FROM product_categories
                                WHERE product_id = p.product_id),
                               'Uncategorized'
                       ) AS categories
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

    # Generate embeddings
    embeddings = embed_client.embed(documents)

    # Store in ChromaDB
    chroma_collection.add(
        ids=[str(pid) for pid in product_ids],
        embeddings=embeddings,
        documents=documents
    )

    elapsed = time.perf_counter() - start
    cur.close()
    return elapsed


# =============================================================================
# SCENARIO 1 FUNCTIONS: Cold Start (insert + embedding generation)
# =============================================================================

def scenario1_unified(conn, products: List[dict]) -> float:
    """
    Scenario 1 - Unified: Insert data and generate embeddings in PostgreSQL.
    Uses SQL JOIN to build context after insertion.
    """
    cur = conn.cursor()
    start = time.perf_counter()

    # Insert relational data
    for p in products:
        cur.execute(
            """INSERT INTO products (name, description, price, stock_count)
               VALUES (%s, %s, %s, %s)
               RETURNING product_id""",
            (p['name'], p['description'], p['price'], p['stock'])
        )
        pid = cur.fetchone()[0]

        for review in p['reviews']:
            cur.execute(
                """INSERT INTO reviews (product_id, rating, review_text)
                   VALUES (%s, %s, %s)""",
                (pid, review['rating'], review['text'])
            )

        for category in p['categories']:
            cur.execute(
                """INSERT INTO product_categories (product_id, category_name)
                   VALUES (%s, %s)""",
                (pid, category)
            )

    # Generate embeddings using SQL JOIN + pg_gembed
    cur.execute("""
                WITH context AS (SELECT p.product_id,
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
                                 FROM products p),
                     embeddings AS (SELECT sentence_id, embedding
                                    FROM embed_texts_with_ids(
                                            'embed_anything',
                                            %s,
                                            (SELECT array_agg(product_id ORDER BY product_id) FROM context),
                                            (SELECT array_agg(full_text ORDER BY product_id) FROM context)
                                         ))
                UPDATE products p
                SET embedding = e.embedding
                FROM embeddings e
                WHERE p.product_id = e.sentence_id;
                """, (EMBED_ANYTHING_MODEL,))

    conn.commit()
    elapsed = time.perf_counter() - start
    cur.close()
    return elapsed


def scenario1_distributed(conn, products: List[dict],
                          embed_client: EmbedAnythingDirectClient,
                          chroma_collection) -> float:
    """
    Scenario 1 - Distributed: Insert data into PostgreSQL, embed from app data,
    store vectors in ChromaDB. No DB fetch needed since data is in memory.
    """
    cur = conn.cursor()
    start = time.perf_counter()

    # Insert relational data into PostgreSQL
    product_ids = []
    for p in products:
        cur.execute(
            """INSERT INTO products (name, description, price, stock_count)
               VALUES (%s, %s, %s, %s)
               RETURNING product_id""",
            (p['name'], p['description'], p['price'], p['stock'])
        )
        pid = cur.fetchone()[0]
        product_ids.append(pid)

        for review in p['reviews']:
            cur.execute(
                """INSERT INTO reviews (product_id, rating, review_text)
                   VALUES (%s, %s, %s)""",
                (pid, review['rating'], review['text'])
            )

        for category in p['categories']:
            cur.execute(
                """INSERT INTO product_categories (product_id, category_name)
                   VALUES (%s, %s)""",
                (pid, category)
            )

    conn.commit()

    # Build context from app-level data (no DB fetch needed)
    documents = [build_embedding_context(p) for p in products]

    # Generate embeddings
    embeddings = embed_client.embed(documents)

    # Store in ChromaDB
    chroma_collection.add(
        ids=[str(pid) for pid in product_ids],
        embeddings=embeddings,
        documents=documents
    )

    elapsed = time.perf_counter() - start
    cur.close()
    return elapsed


# =============================================================================
# Benchmark Runners
# =============================================================================

def run_scenario1_unified(products: List[dict], runs: int) -> List[BenchmarkResult]:
    """Run scenario 1 unified: cold start, insert + embed in PostgreSQL."""
    conn, pid = connect_and_get_pid()

    try:
        # Warm up
        warmup_products = generate_products(8)
        truncate_pg_tables(conn)
        scenario1_unified(conn, warmup_products)

        # Run benchmark
        results = []
        for _ in range(runs):
            truncate_pg_tables(conn)
            elapsed, stats = ResourceMonitor.measure(
                pid,
                lambda: scenario1_unified(conn, products)
            )
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))

        return results
    finally:
        conn.close()


def run_scenario1_distributed(products: List[dict],
                              embed_client: EmbedAnythingDirectClient,
                              runs: int) -> List[BenchmarkResult]:
    """Run scenario 1 distributed: cold start, insert + embed from app data."""
    pid = os.getpid()

    # Warm up
    warmup_products = generate_products(8)
    conn, _ = connect_and_get_pid()
    client, collection, db_path = create_chroma_client()
    try:
        truncate_pg_tables(conn)
        scenario1_distributed(conn, warmup_products, embed_client, collection)
    finally:
        conn.close()
        cleanup_chroma(client, db_path)

    # Run benchmark
    results = []
    for _ in range(runs):
        conn, _ = connect_and_get_pid()
        client, collection, db_path = create_chroma_client()
        try:
            truncate_pg_tables(conn)
            elapsed, stats = ResourceMonitor.measure(
                pid,
                lambda: scenario1_distributed(conn, products, embed_client, collection)
            )
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        finally:
            conn.close()
            cleanup_chroma(client, db_path)

    return results


def run_scenario2_unified(products: List[dict], runs: int) -> List[BenchmarkResult]:
    """Run scenario 2 unified: pre-existing data, generate embeddings only."""
    conn, pid = connect_and_get_pid()

    try:
        # Setup: insert data first (not timed)
        truncate_pg_tables(conn)
        insert_product_data(conn, products)

        # Warm up
        scenario2_unified(conn)
        clear_embeddings(conn)

        # Run benchmark
        results = []
        for _ in range(runs):
            clear_embeddings(conn)
            elapsed, stats = ResourceMonitor.measure(
                pid,
                lambda: scenario2_unified(conn)
            )
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))

        return results
    finally:
        conn.close()


def run_scenario2_distributed(products: List[dict],
                              embed_client: EmbedAnythingDirectClient,
                              runs: int) -> List[BenchmarkResult]:
    """Run scenario 2 distributed: pre-existing data, fetch + embed + store."""
    pid = os.getpid()

    # Setup: insert data into PostgreSQL first (not timed)
    conn, _ = connect_and_get_pid()
    truncate_pg_tables(conn)
    insert_product_data(conn, products)
    conn.close()

    # Warm up
    conn, _ = connect_and_get_pid()
    client, collection, db_path = create_chroma_client()
    try:
        scenario2_distributed(conn, embed_client, collection)
    finally:
        conn.close()
        cleanup_chroma(client, db_path)

    # Run benchmark
    results = []
    for _ in range(runs):
        conn, _ = connect_and_get_pid()
        client, collection, db_path = create_chroma_client()
        try:
            elapsed, stats = ResourceMonitor.measure(
                pid,
                lambda: scenario2_distributed(conn, embed_client, collection)
            )
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        finally:
            conn.close()
            cleanup_chroma(client, db_path)

    return results


# =============================================================================
# Output Functions
# =============================================================================

def print_header():
    """Print benchmark results header."""
    print(f"{'':14} | {'Time (s)':>9} | {'Δ Mem (MB)':>10} | {'Peak (MB)':>10} | "
          f"{'CPU (%)':>9} | {'Sys Mem (MB)':>12} | {'Sys CPU (%)':>11}")
    print("=" * 95)


def print_result(label: str, results: List[BenchmarkResult]):
    """Print aggregated results from multiple runs."""
    times = [r.time_s for r in results]
    delta_mems = [r.stats.delta_mb for r in results]
    peak_mems = [r.stats.peak_mb for r in results]
    cpus = [r.stats.cpu_usage for r in results]
    sys_peak_mems = [r.stats.sys_peak_mb for r in results]
    sys_cpus = [r.stats.sys_cpu_usage for r in results]

    print(f"  {label:12} | {mean(times):>9.3f} | {mean(delta_mems):>10.1f} | "
          f"{mean(peak_mems):>10.1f} | {mean(cpus):>9.1f} | "
          f"{mean(sys_peak_mems):>12.0f} | {mean(sys_cpus):>11.1f}")


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
    print("=" * 95)
    print("SCENARIO 1: Cold Start (insert + embedding generation)")
    print("=" * 95)
    print_header()

    for size in TEST_SIZES:
        products = generate_products(size)
        print(f"Size: {size}")

        unified_results = run_scenario1_unified(products, RUNS_PER_SIZE)
        distributed_results = run_scenario1_distributed(products, embed_client, RUNS_PER_SIZE)

        print_result("Unified", unified_results)
        print_result("Distributed", distributed_results)
        print()

        all_results_s1.append({
            'size': size,
            'unified': {
                'throughput': size / mean([r.time_s for r in unified_results]),
                'time_s': mean([r.time_s for r in unified_results]),
                'cpu': mean([r.stats.cpu_usage for r in unified_results]),
                'mem_delta': mean([r.stats.delta_mb for r in unified_results]),
                'mem_peak': mean([r.stats.peak_mb for r in unified_results]),
                'sys_cpu': mean([r.stats.sys_cpu_usage for r in unified_results]),
                'sys_mem': mean([r.stats.sys_peak_mb for r in unified_results]),
            },
            'distributed': {
                'throughput': size / mean([r.time_s for r in distributed_results]),
                'time_s': mean([r.time_s for r in distributed_results]),
                'cpu': mean([r.stats.cpu_usage for r in distributed_results]),
                'mem_delta': mean([r.stats.delta_mb for r in distributed_results]),
                'mem_peak': mean([r.stats.peak_mb for r in distributed_results]),
                'sys_cpu': mean([r.stats.sys_cpu_usage for r in distributed_results]),
                'sys_mem': mean([r.stats.sys_peak_mb for r in distributed_results]),
            },
        })

    # ==========================================================================
    # Scenario 2: Pre-existing Data (embedding generation only)
    # ==========================================================================
    print("=" * 95)
    print("SCENARIO 2: Pre-existing Data (embedding generation only)")
    print("=" * 95)
    print_header()

    for size in TEST_SIZES:
        products = generate_products(size)
        print(f"Size: {size}")

        unified_results = run_scenario2_unified(products, RUNS_PER_SIZE)
        distributed_results = run_scenario2_distributed(products, embed_client, RUNS_PER_SIZE)

        print_result("Unified", unified_results)
        print_result("Distributed", distributed_results)
        print()

        all_results_s2.append({
            'size': size,
            'unified': {
                'throughput': size / mean([r.time_s for r in unified_results]),
                'time_s': mean([r.time_s for r in unified_results]),
                'cpu': mean([r.stats.cpu_usage for r in unified_results]),
                'mem_delta': mean([r.stats.delta_mb for r in unified_results]),
                'mem_peak': mean([r.stats.peak_mb for r in unified_results]),
                'sys_cpu': mean([r.stats.sys_cpu_usage for r in unified_results]),
                'sys_mem': mean([r.stats.sys_peak_mb for r in unified_results]),
            },
            'distributed': {
                'throughput': size / mean([r.time_s for r in distributed_results]),
                'time_s': mean([r.time_s for r in distributed_results]),
                'cpu': mean([r.stats.cpu_usage for r in distributed_results]),
                'mem_delta': mean([r.stats.delta_mb for r in distributed_results]),
                'mem_peak': mean([r.stats.peak_mb for r in distributed_results]),
                'sys_cpu': mean([r.stats.sys_cpu_usage for r in distributed_results]),
                'sys_mem': mean([r.stats.sys_peak_mb for r in distributed_results]),
            },
        })

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("=" * 95)
    print("\nAverage Throughput (products/sec):")

    print("\n  Scenario 1 - Cold Start:")
    print(f"    Unified (PG+pg_gembed):      {mean([r['unified']['throughput'] for r in all_results_s1]):.2f}")
    print(f"    Distributed (PG+ChromaDB):   {mean([r['distributed']['throughput'] for r in all_results_s1]):.2f}")

    print("\n  Scenario 2 - Pre-existing Data:")
    print(f"    Unified (PG+pg_gembed):      {mean([r['unified']['throughput'] for r in all_results_s2]):.2f}")
    print(f"    Distributed (PG+ChromaDB):   {mean([r['distributed']['throughput'] for r in all_results_s2]):.2f}")

    # Save results to CSV and generate plots
    save_results_csv(all_results_s1, all_results_s2)
    generate_plots(all_results_s1, all_results_s2)


def save_results_csv(all_results_s1: List[dict], all_results_s2: List[dict]):
    """Save benchmark results to CSV files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    methods = ['unified', 'distributed']
    metrics = ['throughput', 'time_s', 'cpu', 'mem_delta', 'mem_peak', 'sys_cpu', 'sys_mem']

    # Scenario 1 CSV
    csv_path_s1 = OUTPUT_DIR / f"scenario1_cold_start_{timestamp}.csv"
    with open(csv_path_s1, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['size']
        for method in methods:
            for metric in metrics:
                header.append(f"{method}_{metric}")
        writer.writerow(header)
        for r in all_results_s1:
            row = [r['size']]
            for method in methods:
                for metric in metrics:
                    row.append(r[method][metric])
            writer.writerow(row)

    # Scenario 2 CSV
    csv_path_s2 = OUTPUT_DIR / f"scenario2_preexisting_{timestamp}.csv"
    with open(csv_path_s2, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)  # Same header
        for r in all_results_s2:
            row = [r['size']]
            for method in methods:
                for metric in metrics:
                    row.append(r[method][metric])
            writer.writerow(row)

    print(f"\nResults saved to:")
    print(f"  {csv_path_s1}")
    print(f"  {csv_path_s2}")


def generate_plots(all_results_s1: List[dict], all_results_s2: List[dict]):
    """Generate comparison plots for throughput, CPU, and memory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sizes_s1 = [r['size'] for r in all_results_s1]
    sizes_s2 = [r['size'] for r in all_results_s2]
    methods = ['unified', 'distributed']
    labels = ['Unified (PG)', 'Distributed (PG + ChromaDB)']
    colors = ['#2ecc71', '#e74c3c']
    markers = ['o', 's']

    def plot_metric(results, sizes, metric, ylabel, title_suffix, filename):
        plt.figure(figsize=(10, 6))
        for method, label, color, marker in zip(methods, labels, colors, markers):
            plt.plot(sizes, [r[method][metric] for r in results],
                     f'{marker}-', label=label, linewidth=2, color=color)
        plt.xlabel('Number of Products')
        plt.ylabel(ylabel)
        plt.title(f'{title_suffix}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log', base=2)
        plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
        plt.close()

    # Scenario 1 plots
    plot_metric(all_results_s1, sizes_s1, 'throughput', 'Throughput (products/sec)',
                'Scenario 1 Cold Start: Throughput', f"s1_throughput_{timestamp}.png")
    plot_metric(all_results_s1, sizes_s1, 'cpu', 'Process CPU Usage (%)',
                'Scenario 1 Cold Start: Process CPU\n(Unified: PostgreSQL, Distributed: Python)',
                f"s1_cpu_process_{timestamp}.png")
    plot_metric(all_results_s1, sizes_s1, 'mem_delta', 'Process Memory Delta (MB)',
                'Scenario 1 Cold Start: Process Memory Change\n(Unified: PostgreSQL, Distributed: Python)',
                f"s1_memory_delta_{timestamp}.png")
    plot_metric(all_results_s1, sizes_s1, 'mem_peak', 'Process Peak Memory (MB)',
                'Scenario 1 Cold Start: Process Peak Memory\n(Unified: PostgreSQL, Distributed: Python)',
                f"s1_memory_peak_{timestamp}.png")
    plot_metric(all_results_s1, sizes_s1, 'sys_cpu', 'System CPU Usage (%)',
                'Scenario 1 Cold Start: System CPU', f"s1_cpu_system_{timestamp}.png")
    plot_metric(all_results_s1, sizes_s1, 'sys_mem', 'System Memory Used (MB)',
                'Scenario 1 Cold Start: System Memory', f"s1_memory_system_{timestamp}.png")

    # Scenario 2 plots
    plot_metric(all_results_s2, sizes_s2, 'throughput', 'Throughput (products/sec)',
                'Scenario 2 Pre-existing: Throughput', f"s2_throughput_{timestamp}.png")
    plot_metric(all_results_s2, sizes_s2, 'cpu', 'Process CPU Usage (%)',
                'Scenario 2 Pre-existing: Process CPU\n(Unified: PostgreSQL, Distributed: Python)',
                f"s2_cpu_process_{timestamp}.png")
    plot_metric(all_results_s2, sizes_s2, 'mem_delta', 'Process Memory Delta (MB)',
                'Scenario 2 Pre-existing: Process Memory Change\n(Unified: PostgreSQL, Distributed: Python)',
                f"s2_memory_delta_{timestamp}.png")
    plot_metric(all_results_s2, sizes_s2, 'mem_peak', 'Process Peak Memory (MB)',
                'Scenario 2 Pre-existing: Process Peak Memory\n(Unified: PostgreSQL, Distributed: Python)',
                f"s2_memory_peak_{timestamp}.png")
    plot_metric(all_results_s2, sizes_s2, 'sys_cpu', 'System CPU Usage (%)',
                'Scenario 2 Pre-existing: System CPU', f"s2_cpu_system_{timestamp}.png")
    plot_metric(all_results_s2, sizes_s2, 'sys_mem', 'System Memory Used (MB)',
                'Scenario 2 Pre-existing: System Memory', f"s2_memory_system_{timestamp}.png")

    # Summary bar charts (2x6 grid: 2 scenarios x 6 metrics)
    # Note: Process metrics track PostgreSQL for Unified, Python for Distributed
    fig, axes = plt.subplots(2, 6, figsize=(20, 8))
    metric_labels = ['Throughput\n(prod/sec)', 'Proc CPU* (%)', 'Proc Mem* Δ (MB)', 'Proc Peak* (MB)', 'Sys CPU (%)',
                     'Sys Mem (MB)']
    metric_keys = ['throughput', 'cpu', 'mem_delta', 'mem_peak', 'sys_cpu', 'sys_mem']

    for col, (metric, mlabel) in enumerate(zip(metric_keys, metric_labels)):
        # Scenario 1
        avgs = [mean([r[m][metric] for r in all_results_s1]) for m in methods]
        bars = axes[0, col].bar(labels, avgs, color=colors)
        axes[0, col].set_title(f'S1: {mlabel}')
        axes[0, col].bar_label(bars, fmt='%.1f')
        axes[0, col].tick_params(axis='x', rotation=15)

        # Scenario 2
        avgs = [mean([r[m][metric] for r in all_results_s2]) for m in methods]
        bars = axes[1, col].bar(labels, avgs, color=colors)
        axes[1, col].set_title(f'S2: {mlabel}')
        axes[1, col].bar_label(bars, fmt='%.1f')
        axes[1, col].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"summary_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
