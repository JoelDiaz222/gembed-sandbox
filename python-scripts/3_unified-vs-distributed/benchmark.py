import csv
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev, median, quantiles
from typing import Callable, List

import chromadb
import docker
import embed_anything
import matplotlib.pyplot as plt
import psutil
import psycopg2
from embed_anything import EmbeddingModel, WhichModel
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct

# Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'joeldiaz',
    'user': 'joeldiaz',
}

QDRANT_URL = "http://localhost:6333"
QDRANT_CONTAINER_NAME = "qdrant"

TEST_SIZES = [16, 32, 64, 128, 256, 512]
EMBED_ANYTHING_MODEL = "Qdrant/all-MiniLM-L6-v2-onnx"
RUNS_PER_SIZE = 5

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output"

# Global model cache
model_cache = {}


# Simple wrapper matching Chroma's EmbeddingFunction interface
class EmbeddingWrapper:
    def __init__(self, fn: Callable):
        self._fn = fn

    def __call__(self, input):
        return self._fn(list(input))


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ResourceStats:
    """Resource statistics for Python, PostgreSQL, and Qdrant processes."""
    py_delta_mb: float
    py_peak_mb: float
    py_cpu: float
    pg_delta_mb: float
    pg_peak_mb: float
    pg_cpu: float
    qd_delta_mb: float
    qd_peak_mb: float
    qd_cpu: float
    sys_mem_mb: float
    sys_cpu: float


@dataclass
class BenchmarkResult:
    time_s: float
    stats: ResourceStats


# =============================================================================
# Statistics Functions
# =============================================================================

def safe_stdev(values: List[float]) -> float:
    """Calculate standard deviation, returning 0 for lists with <2 elements."""
    return stdev(values) if len(values) > 1 else 0.0


def calc_iqr(values: List[float]) -> float:
    """Calculate interquartile range (Q3 - Q1)."""
    if len(values) < 4:
        return 0.0
    q = quantiles(values, n=4)
    return q[2] - q[0]  # Q3 - Q1


# =============================================================================
# Resource Monitoring
# =============================================================================

class ResourceMonitor:
    def __init__(self, py_pid: int, pg_pid: int = None, qd_name: str = QDRANT_CONTAINER_NAME):
        self.py_process = psutil.Process(py_pid)
        self.pg_process = psutil.Process(pg_pid) if pg_pid else None

        # Docker initialization
        self.docker_client = docker.from_env()
        try:
            self.container = self.docker_client.containers.get(qd_name)
        except Exception:
            self.container = None

        # Baselines
        self.py_baseline = self._get_py_mem()
        self.pg_baseline = self._get_pg_mem()
        self.qd_baseline_stats = self._get_qd_stats() if self.container else None

        # Warm up CPU counters
        self.py_process.cpu_percent()
        if self.pg_process: self.pg_process.cpu_percent()
        time.sleep(0.1)

    def _get_py_mem(self):
        m = self.py_process.memory_full_info()
        return m.uss if hasattr(m, 'uss') else m.rss

    def _get_pg_mem(self):
        if not self.pg_process: return 0
        try:
            m = self.pg_process.memory_full_info()
            return m.uss if hasattr(m, 'uss') else m.rss
        except:
            return 0

    def _get_qd_stats(self):
        """Returns a snapshot of Docker stats."""
        return self.container.stats(stream=False)

    def _calculate_qd_cpu(self, start_stats, end_stats):
        """Calculates CPU percentage similar to 'docker stats'."""
        cpu_delta = end_stats['cpu_stats']['cpu_usage']['total_usage'] - \
                    start_stats['cpu_stats']['cpu_usage']['total_usage']
        system_delta = end_stats['cpu_stats']['system_cpu_usage'] - \
                       start_stats['cpu_stats']['system_cpu_usage']

        if system_delta > 0.0 and cpu_delta > 0.0:
            # We multiply by number of cores to get a 0-100% per-core scaled value
            cpus = end_stats['cpu_stats'].get('online_cpus', psutil.cpu_count())
            return (cpu_delta / system_delta) * cpus * 100.0
        return 0.0

    @staticmethod
    def measure(py_pid: int, pg_pid: int, func: Callable):
        monitor = ResourceMonitor(py_pid, pg_pid)
        result = func()

        # Gather final stats
        py_peak = monitor._get_py_mem()
        py_cpu = monitor.py_process.cpu_percent()

        pg_peak = monitor._get_pg_mem()
        pg_cpu = monitor.pg_process.cpu_percent() if monitor.pg_process else 0.0

        qd_delta_mb = 0.0
        qd_peak_mb = 0.0
        qd_cpu = 0.0

        if monitor.container:
            end_qd_stats = monitor._get_qd_stats()
            qd_peak_raw = end_qd_stats['memory_stats']['usage']
            qd_peak_mb = qd_peak_raw / (1024 * 1024)

            baseline_raw = monitor.qd_baseline_stats['memory_stats']['usage']
            qd_delta_mb = qd_peak_mb - (baseline_raw / (1024 * 1024))
            qd_cpu = monitor._calculate_qd_cpu(monitor.qd_baseline_stats, end_qd_stats)

        sys_v = psutil.virtual_memory()

        stats = ResourceStats(
            py_delta_mb=(py_peak - monitor.py_baseline) / 1e6,
            py_peak_mb=py_peak / 1e6,
            py_cpu=py_cpu,
            pg_delta_mb=(pg_peak - monitor.pg_baseline) / 1e6,
            pg_peak_mb=pg_peak / 1e6,
            pg_cpu=pg_cpu,
            qd_delta_mb=qd_delta_mb,
            qd_peak_mb=qd_peak_mb,
            qd_cpu=qd_cpu,
            sys_mem_mb=sys_v.used / 1e6,
            sys_cpu=psutil.cpu_percent()
        )
        return result, stats


# =============================================================================
# Embedding Client
# =============================================================================

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
    reviews_pool = get_review_texts(total_reviews_needed, shuffle=True)
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

def scenario2_unified(conn) -> float:
    """Scenario 2 - Unified: Generate embeddings for pre-existing data."""
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


def scenario2_distributed_chroma(conn, embed_client: EmbedAnythingDirectClient,
                                 chroma_collection) -> float:
    """Scenario 2 - Distributed (Chroma): Fetch data, embed, store in ChromaDB."""
    cur = conn.cursor()
    start = time.perf_counter()

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


def scenario2_distributed_qdrant(conn, embed_client: EmbedAnythingDirectClient,
                                 qdrant_client: QdrantClient) -> float:
    """Scenario 2 - Distributed (Qdrant): Fetch data, embed, store in Qdrant."""
    cur = conn.cursor()
    start = time.perf_counter()

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

    # Generate embeddings
    embeddings = embed_client.embed(documents)

    # Store in Qdrant
    points = [
        PointStruct(id=pid, vector=embedding, payload={"text": doc})
        for pid, embedding, doc in zip(product_ids, embeddings, documents)
    ]
    qdrant_client.upsert(collection_name="products", points=points, wait=True)

    elapsed = time.perf_counter() - start
    cur.close()
    return elapsed


# =============================================================================
# SCENARIO 1 FUNCTIONS: Cold Start (insert + embedding generation)
# =============================================================================

def scenario1_unified(conn, products: List[dict]) -> float:
    """Scenario 1 - Unified: Insert data and generate embeddings in PostgreSQL."""
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
            cur.execute("""INSERT INTO reviews (product_id, rating, review_text)
                           VALUES (%s, %s, %s)""", (pid, review['rating'], review['text']))
        for category in p['categories']:
            cur.execute("""INSERT INTO product_categories (product_id, category_name)
                           VALUES (%s, %s)""", (pid, category))

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


def scenario1_distributed_chroma(conn, products: List[dict],
                                 embed_client: EmbedAnythingDirectClient,
                                 chroma_collection) -> float:
    """Scenario 1 - Distributed (Chroma): Insert data into PG, embed from app data, store in Chroma."""
    cur = conn.cursor()
    start = time.perf_counter()

    # Insert relational data
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
            cur.execute("""INSERT INTO reviews (product_id, rating, review_text)
                           VALUES (%s, %s, %s)""", (pid, review['rating'], review['text']))
        for category in p['categories']:
            cur.execute("""INSERT INTO product_categories (product_id, category_name)
                           VALUES (%s, %s)""", (pid, category))

    conn.commit()

    # Build context from app-level data
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


def scenario1_distributed_qdrant(conn, products: List[dict],
                                 embed_client: EmbedAnythingDirectClient,
                                 qdrant_client: QdrantClient) -> float:
    """Scenario 1 - Distributed (Qdrant): Insert data into PG, embed from app data, store in Qdrant."""
    cur = conn.cursor()
    start = time.perf_counter()

    # Insert relational data
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
            cur.execute("""INSERT INTO reviews (product_id, rating, review_text)
                           VALUES (%s, %s, %s)""", (pid, review['rating'], review['text']))
        for category in p['categories']:
            cur.execute("""INSERT INTO product_categories (product_id, category_name)
                           VALUES (%s, %s)""", (pid, category))

    conn.commit()

    # Build context from app-level data
    documents = [build_embedding_context(p) for p in products]

    # Generate embeddings
    embeddings = embed_client.embed(documents)

    # Store in Qdrant
    points = [
        PointStruct(id=pid, vector=embedding, payload={"text": doc})
        for pid, embedding, doc in zip(product_ids, embeddings, documents)
    ]
    qdrant_client.upsert(collection_name="products", points=points, wait=True)

    elapsed = time.perf_counter() - start
    cur.close()
    return elapsed


# =============================================================================
# Benchmark Runners
# =============================================================================

def run_scenario1_unified(products: List[dict], runs: int) -> List[BenchmarkResult]:
    """Run scenario 1 unified: cold start, insert + embed in PostgreSQL."""
    conn, pg_pid = connect_and_get_pid()
    py_pid = os.getpid()

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

    # Warm up
    warmup_products = generate_products(8)
    conn, pg_pid = connect_and_get_pid()
    client, collection, db_path = create_chroma_client(embed_fn=embed_client.embed)
    try:
        truncate_pg_tables(conn)
        scenario1_distributed_chroma(conn, warmup_products, embed_client, collection)
    finally:
        conn.close()
        cleanup_chroma(client, db_path)

    # Run benchmark
    results = []
    for _ in range(runs):
        conn, pg_pid = connect_and_get_pid()
        client, collection, db_path = create_chroma_client(embed_fn=embed_client.embed)
        try:
            truncate_pg_tables(conn)
            elapsed, stats = ResourceMonitor.measure(
                py_pid, pg_pid,
                lambda: scenario1_distributed_chroma(conn, products, embed_client, collection)
            )
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        finally:
            conn.close()
            cleanup_chroma(client, db_path)

    return results


def run_scenario1_distributed_qdrant(products: List[dict],
                                     embed_client: EmbedAnythingDirectClient,
                                     runs: int) -> List[BenchmarkResult]:
    """Run scenario 1 distributed (Qdrant)."""
    py_pid = os.getpid()

    # Warm up
    warmup_products = generate_products(8)
    conn, pg_pid = connect_and_get_pid()
    client = create_qdrant_client()
    try:
        truncate_pg_tables(conn)
        scenario1_distributed_qdrant(conn, warmup_products, embed_client, client)
    finally:
        conn.close()
        cleanup_qdrant(client)

    # Run benchmark
    results = []
    for _ in range(runs):
        conn, pg_pid = connect_and_get_pid()
        client = create_qdrant_client()
        try:
            truncate_pg_tables(conn)
            elapsed, stats = ResourceMonitor.measure(
                py_pid, pg_pid,
                lambda: scenario1_distributed_qdrant(conn, products, embed_client, client)
            )
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        finally:
            conn.close()
            cleanup_qdrant(client)

    return results


def run_scenario2_unified(products: List[dict], runs: int) -> List[BenchmarkResult]:
    """Run scenario 2 unified: pre-existing data."""
    conn, pg_pid = connect_and_get_pid()
    py_pid = os.getpid()

    try:
        # Setup
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

    # Setup
    conn, pg_pid = connect_and_get_pid()
    truncate_pg_tables(conn)
    insert_product_data(conn, products)
    conn.close()

    # Warm up
    conn, _ = connect_and_get_pid()
    client, collection, db_path = create_chroma_client(embed_fn=embed_client.embed)
    try:
        scenario2_distributed_chroma(conn, embed_client, collection)
    finally:
        conn.close()
        cleanup_chroma(client, db_path)

    # Run benchmark
    results = []
    for _ in range(runs):
        conn, pg_pid = connect_and_get_pid()
        client, collection, db_path = create_chroma_client(embed_fn=embed_client.embed)
        try:
            elapsed, stats = ResourceMonitor.measure(
                py_pid, pg_pid,
                lambda: scenario2_distributed_chroma(conn, embed_client, collection)
            )
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        finally:
            conn.close()
            cleanup_chroma(client, db_path)

    return results


def run_scenario2_distributed_qdrant(products: List[dict],
                                     embed_client: EmbedAnythingDirectClient,
                                     runs: int) -> List[BenchmarkResult]:
    """Run scenario 2 distributed (Qdrant)."""
    py_pid = os.getpid()

    # Setup
    conn, pg_pid = connect_and_get_pid()
    truncate_pg_tables(conn)
    insert_product_data(conn, products)
    conn.close()

    # Warm up
    conn, _ = connect_and_get_pid()
    client = create_qdrant_client()
    try:
        scenario2_distributed_qdrant(conn, embed_client, client)
    finally:
        conn.close()
        cleanup_qdrant(client)

    # Run benchmark
    results = []
    for _ in range(runs):
        conn, pg_pid = connect_and_get_pid()
        client = create_qdrant_client()
        try:
            elapsed, stats = ResourceMonitor.measure(
                py_pid, pg_pid,
                lambda: scenario2_distributed_qdrant(conn, embed_client, client)
            )
            results.append(BenchmarkResult(time_s=elapsed, stats=stats))
        finally:
            conn.close()
            cleanup_qdrant(client)

    return results


# =============================================================================
# Output Functions
# =============================================================================

def print_header():
    """Print benchmark results header."""
    lbl_w = 12
    time_w = 12
    col_w = 11
    med_w = 7

    header = (
            "  " +
            f"{'':{lbl_w}}{'':{med_w}} | {'Time (s) μ±σ':>{time_w}} | {'Py Δ MB μ±σ':>{col_w}} | {'Py Peak μ±σ':>{col_w}} | {'Py CPU μ±σ':>{col_w}} | "
            f"{'PG Δ MB μ±σ':>{col_w}} | {'PG Peak μ±σ':>{col_w}} | {'PG CPU μ±σ':>{col_w}} | {'QD Δ MB μ±σ':>{col_w}} | {'QD CPU μ±σ':>{col_w}}"
    )
    print("Benchmark Results:", flush=True)
    print(header, flush=True)
    print("=" * len(header), flush=True)


def print_result(label: str, results: List[BenchmarkResult]):
    """Print aggregated results."""
    times = [r.time_s for r in results]
    py_delta = [r.stats.py_delta_mb for r in results]
    py_peak = [r.stats.py_peak_mb for r in results]
    py_cpu = [r.stats.py_cpu for r in results]
    pg_delta = [r.stats.pg_delta_mb for r in results]
    pg_peak = [r.stats.pg_peak_mb for r in results]
    pg_cpu = [r.stats.pg_cpu for r in results]
    qd_delta = [r.stats.qd_delta_mb for r in results]
    qd_cpu = [r.stats.qd_cpu for r in results]

    def fmt_mean(values: List[float], precision: int = 1) -> str:
        avg = mean(values)
        std = safe_stdev(values)
        if precision == 3:
            return f"{avg:.3f}±{std:.3f}"
        return f"{avg:.1f}±{std:.1f}"

    def fmt_median_iqr(values: List[float], precision: int = 1) -> str:
        med = median(values)
        iqr = calc_iqr(values)
        if precision == 3:
            return f"{med:.3f}±{iqr:.3f}"
        return f"{med:.1f}±{iqr:.1f}"

    row_fmt = (
        "  {label:<12}{med:>7} | {time:>12} | {pyd:>11} | {pyp:>11} | {pyc:>11} | {pgd:>11} | {pgp:>11} | {pgc:>11} | {qdd:>11} | {qdc:>11}"
    )
    print(row_fmt.format(
        label=label,
        med='',
        time=fmt_mean(times, 3),
        pyd=fmt_mean(py_delta),
        pyp=fmt_mean(py_peak),
        pyc=fmt_mean(py_cpu),
        pgd=fmt_mean(pg_delta),
        pgp=fmt_mean(pg_peak),
        pgc=fmt_mean(pg_cpu),
        qdd=fmt_mean(qd_delta),
        qdc=fmt_mean(qd_cpu),
    ), flush=True)
    print(row_fmt.format(
        label=label,
        med=' (med)',
        time=fmt_median_iqr(times, 3),
        pyd=fmt_median_iqr(py_delta),
        pyp=fmt_median_iqr(py_peak),
        pyc=fmt_median_iqr(py_cpu),
        pgd=fmt_median_iqr(pg_delta),
        pgp=fmt_median_iqr(pg_peak),
        pgc=fmt_median_iqr(pg_cpu),
        qdd=fmt_median_iqr(qd_delta),
        qdc=fmt_median_iqr(qd_cpu),
    ), flush=True)


def compute_metrics(size: int, results: List[BenchmarkResult]) -> dict:
    """Compute mean/std and median/IQR for all metrics from benchmark results."""
    times = [r.time_s for r in results]
    py_cpu = [r.stats.py_cpu for r in results]
    py_delta = [r.stats.py_delta_mb for r in results]
    py_peak = [r.stats.py_peak_mb for r in results]
    pg_cpu = [r.stats.pg_cpu for r in results]
    pg_delta = [r.stats.pg_delta_mb for r in results]
    pg_peak = [r.stats.pg_peak_mb for r in results]
    qd_cpu = [r.stats.qd_cpu for r in results]
    qd_delta = [r.stats.qd_delta_mb for r in results]
    qd_peak = [r.stats.qd_peak_mb for r in results]
    sys_cpu = [r.stats.sys_cpu for r in results]
    sys_mem = [r.stats.sys_mem_mb for r in results]

    return {
        # Throughput
        'throughput': size / mean(times),
        'throughput_std': size / mean(times) * safe_stdev(times) / mean(times) if len(times) > 1 else 0,
        'throughput_median': size / median(times),
        'throughput_iqr': size / median(times) * calc_iqr(times) / median(times) if len(times) >= 4 else 0,
        # Time
        'time_s': mean(times),
        'time_s_std': safe_stdev(times),
        'time_s_median': median(times),
        'time_s_iqr': calc_iqr(times),
        # Python
        'py_cpu': mean(py_cpu),
        'py_cpu_std': safe_stdev(py_cpu),
        'py_cpu_median': median(py_cpu),
        'py_cpu_iqr': calc_iqr(py_cpu),
        'py_mem_delta': mean(py_delta),
        'py_mem_delta_std': safe_stdev(py_delta),
        'py_mem_peak': mean(py_peak),
        'py_mem_peak_std': safe_stdev(py_peak),
        'py_mem_peak_median': median(py_peak),
        'py_mem_peak_iqr': calc_iqr(py_peak),
        # PG
        'pg_cpu': mean(pg_cpu),
        'pg_cpu_std': safe_stdev(pg_cpu),
        'pg_cpu_median': median(pg_cpu),
        'pg_cpu_iqr': calc_iqr(pg_cpu),
        'pg_mem_delta': mean(pg_delta),
        'pg_mem_delta_std': safe_stdev(pg_delta),
        'pg_mem_peak': mean(pg_peak),
        'pg_mem_peak_std': safe_stdev(pg_peak),
        'pg_mem_peak_median': median(pg_peak),
        'pg_mem_peak_iqr': calc_iqr(pg_peak),
        # Qdrant
        'qd_cpu': mean(qd_cpu),
        'qd_cpu_std': safe_stdev(qd_cpu),
        'qd_cpu_median': median(qd_cpu),
        'qd_cpu_iqr': calc_iqr(qd_cpu),
        'qd_mem_delta': mean(qd_delta),
        'qd_mem_delta_std': safe_stdev(qd_delta),
        'qd_mem_peak': mean(qd_peak),
        'qd_mem_peak_std': safe_stdev(qd_peak),
        'qd_mem_peak_median': median(qd_peak),
        'qd_mem_peak_iqr': calc_iqr(qd_peak),
        # System
        'sys_cpu': mean(sys_cpu),
        'sys_cpu_std': safe_stdev(sys_cpu),
        'sys_cpu_median': median(sys_cpu),
        'sys_cpu_iqr': calc_iqr(sys_cpu),
        'sys_mem': mean(sys_mem),
        'sys_mem_std': safe_stdev(sys_mem),
        'sys_mem_median': median(sys_mem),
        'sys_mem_iqr': calc_iqr(sys_mem),
    }


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
    print("=" * 120)
    print("SCENARIO 1: Cold Start (insert + embedding generation)")
    print("=" * 120)
    print_header()

    for size in TEST_SIZES:
        products = generate_products(size)
        print(f"Size: {size}", flush=True)

        unified_results = run_scenario1_unified(products, RUNS_PER_SIZE)
        chroma_results = run_scenario1_distributed_chroma(products, embed_client, RUNS_PER_SIZE)
        qdrant_results = run_scenario1_distributed_qdrant(products, embed_client, RUNS_PER_SIZE)

        print_result("Unified", unified_results)
        print_result("Chroma", chroma_results)
        print_result("Qdrant", qdrant_results)
        print()

        all_results_s1.append({
            'size': size,
            'unified': compute_metrics(size, unified_results),
            'chroma': compute_metrics(size, chroma_results),
            'qdrant': compute_metrics(size, qdrant_results),
        })

    # ==========================================================================
    # Scenario 2: Pre-existing Data (embedding generation only)
    # ==========================================================================
    print("=" * 120)
    print("SCENARIO 2: Pre-existing Data (embedding generation only)")
    print("=" * 120)
    print_header()

    for size in TEST_SIZES:
        products = generate_products(size)
        print(f"Size: {size}", flush=True)

        unified_results = run_scenario2_unified(products, RUNS_PER_SIZE)
        chroma_results = run_scenario2_distributed_chroma(products, embed_client, RUNS_PER_SIZE)
        qdrant_results = run_scenario2_distributed_qdrant(products, embed_client, RUNS_PER_SIZE)

        print_result("Unified", unified_results)
        print_result("Chroma", chroma_results)
        print_result("Qdrant", qdrant_results)
        print()

        all_results_s2.append({
            'size': size,
            'unified': compute_metrics(size, unified_results),
            'chroma': compute_metrics(size, chroma_results),
            'qdrant': compute_metrics(size, qdrant_results),
        })

    # Save results to CSV and generate plots
    save_results_csv(all_results_s1, all_results_s2)
    generate_plots(all_results_s1, all_results_s2)


def save_results_csv(all_results_s1: List[dict], all_results_s2: List[dict]):
    """Save benchmark results to CSV files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    methods = ['unified', 'chroma', 'qdrant']
    metrics = ['throughput', 'time_s',
               'py_cpu', 'py_mem_delta', 'py_mem_peak',
               'pg_cpu', 'pg_mem_delta', 'pg_mem_peak',
               'qd_cpu', 'qd_mem_delta', 'qd_mem_peak',
               'sys_cpu', 'sys_mem']

    # Build header
    header = ['size']
    for method in methods:
        for metric in metrics:
            header.append(f"{method}_{metric}")
            header.append(f"{method}_{metric}_std")
            header.append(f"{method}_{metric}_median")
            header.append(f"{method}_{metric}_iqr")

    for scenario, results in [('scenario1_cold_start', all_results_s1), ('scenario2_preexisting', all_results_s2)]:
        csv_path = OUTPUT_DIR / f"{scenario}_{timestamp}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for r in results:
                row = [r['size']]
                for method in methods:
                    for metric in metrics:
                        val = r[method].get(metric)
                        row.append(val)
                        row.append(r[method].get(f"{metric}_std", 0))
                        row.append(r[method].get(f"{metric}_median", 0))
                        row.append(r[method].get(f"{metric}_iqr", 0))
                writer.writerow(row)
        print(f"Saved {csv_path}")


def generate_plots(all_results_s1: List[dict], all_results_s2: List[dict]):
    """Generate comparison plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sizes_s1 = [r['size'] for r in all_results_s1]
    sizes_s2 = [r['size'] for r in all_results_s2]

    methods = ['unified', 'chroma', 'qdrant']
    labels = ['Unified (PG)', 'Distributed (Chroma)', 'Distributed (Qdrant)']
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    markers = ['o', 's', '^']

    def plot_metric(results, sizes, metric, ylabel, title_suffix, filename):
        plt.figure(figsize=(10, 6))
        for method, label, color, marker in zip(methods, labels, colors, markers):
            y_vals = [r[method].get(f'{metric}_median', r[method].get(metric, 0)) for r in results]
            y_errs = [r[method].get(f'{metric}_iqr', r[method].get(f'{metric}_std', 0)) for r in results]
            plt.errorbar(sizes, y_vals, yerr=y_errs, fmt=f'{marker}-', label=label,
                         linewidth=2, color=color, capsize=3, capthick=1)
        plt.xlabel('Number of Products')
        plt.ylabel(ylabel)
        plt.title(f'{title_suffix} (Median ± IQR)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xscale('log', base=2)
        plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
        plt.close()

    # Generate plots for both scenarios
    scenarios = [
        (all_results_s1, sizes_s1, 's1', 'Scenario 1 Cold Start'),
        (all_results_s2, sizes_s2, 's2', 'Scenario 2 Pre-existing')
    ]

    for results, sizes, prefix, title in scenarios:
        plot_metric(results, sizes, 'throughput', 'Throughput (products/sec)',
                    f'{title}: Throughput', f"{prefix}_throughput_{timestamp}.png")
        plot_metric(results, sizes, 'py_cpu', 'Python Process CPU (%)',
                    f'{title}: Python Process CPU', f"{prefix}_py_cpu_{timestamp}.png")
        plot_metric(results, sizes, 'pg_cpu', 'PostgreSQL Process CPU (%)',
                    f'{title}: PostgreSQL CPU', f"{prefix}_pg_cpu_{timestamp}.png")
        plot_metric(results, sizes, 'qd_cpu', 'Qdrant CPU (%)',
                    f'{title}: Qdrant CPU', f"{prefix}_qd_cpu_{timestamp}.png")
        plot_metric(results, sizes, 'sys_mem', 'System Memory (MB)',
                    f'{title}: System Memory', f"{prefix}_memory_system_{timestamp}.png")

    # Summary bar charts
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    metric_labels = ['Throughput\n(prod/sec)', 'Py CPU (%)', 'PG CPU (%)', 'QD CPU (%)', 'Time (s)']
    metric_keys = ['throughput', 'py_cpu', 'pg_cpu', 'qd_cpu', 'time_s']
    x_pos = range(len(methods))

    for col, (metric, mlabel) in enumerate(zip(metric_keys, metric_labels)):
        # Scenario 1
        avgs = [median([r[m].get(f"{metric}_median", r[m].get(metric, 0)) for r in all_results_s1]) for m in methods]
        stds = [median([r[m].get(f"{metric}_iqr", r[m].get(f"{metric}_std", 0)) for r in all_results_s1]) for m in
                methods]
        axes[0, col].bar(x_pos, avgs, yerr=stds, color=colors, capsize=3)
        axes[0, col].set_xticks(x_pos)
        axes[0, col].set_xticklabels(labels, rotation=45, ha='right')
        axes[0, col].set_title(f'S1: {mlabel}')

        # Scenario 2
        avgs = [median([r[m].get(f"{metric}_median", r[m].get(metric, 0)) for r in all_results_s2]) for m in methods]
        stds = [median([r[m].get(f"{metric}_iqr", r[m].get(f"{metric}_std", 0)) for r in all_results_s2]) for m in
                methods]
        axes[1, col].bar(x_pos, avgs, yerr=stds, color=colors, capsize=3)
        axes[1, col].set_xticks(x_pos)
        axes[1, col].set_xticklabels(labels, rotation=45, ha='right')
        axes[1, col].set_title(f'S2: {mlabel}')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"summary_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
