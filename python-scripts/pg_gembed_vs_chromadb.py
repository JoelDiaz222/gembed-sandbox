import os
import random
import shutil
import time
from dataclasses import dataclass
from statistics import mean, stdev
from typing import List, Dict, Tuple

import chromadb
import psycopg2
from fastembed import TextEmbedding

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "joeldiaz",
    "user": "joeldiaz"
}

COLD_START_SIZES = [50, 100, 250, 500]
AGGREGATION_SIZES = [50, 100, 250, 500]
BASIC_EMBEDDING_SIZES = [50, 100, 250, 500]
UPDATE_SIZES = [50, 100, 250, 500]
UPDATE_ITERATIONS = 50

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PG_GEMBED_MODEL = "Qdrant/all-MiniLM-L6-v2-onnx"

RUNS_PER_SCENARIO = 5  # Multiple runs for variance


@dataclass
class BenchmarkResult:
    scenario: str
    system: str
    size: int
    times_sec: List[float]
    mean_time: float
    std_dev: float
    mean_throughput: float


results = []


class SharedResources:
    def __init__(self):
        self.db_conn = None
        self.embedding_model = None
        self.chroma_client = None
        self.chroma_path = "./chroma_benchmark"

    def get_db_connection(self):
        if self.db_conn is None or self.db_conn.closed:
            self.db_conn = psycopg2.connect(**DB_CONFIG)
        return self.db_conn

    def get_embedding_model(self):
        if self.embedding_model is None:
            self.embedding_model = TextEmbedding(MODEL_NAME)
        return self.embedding_model

    def get_chroma_client(self):
        if self.chroma_client is None:
            if os.path.exists(self.chroma_path):
                shutil.rmtree(self.chroma_path, ignore_errors=True)
                time.sleep(0.1)
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        return self.chroma_client

    def cleanup(self):
        if self.db_conn and not self.db_conn.closed:
            self.db_conn.close()
        if self.chroma_client:
            del self.chroma_client
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path, ignore_errors=True)


def setup_database():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_gembed;")

    cur.execute("DROP TABLE IF EXISTS reviews CASCADE;")
    cur.execute("DROP TABLE IF EXISTS product_categories CASCADE;")
    cur.execute("DROP TABLE IF EXISTS products CASCADE;")

    cur.execute("""
                CREATE TABLE products
                (
                    product_id   SERIAL PRIMARY KEY,
                    name         TEXT NOT NULL,
                    description  TEXT NOT NULL,
                    stock_count  INTEGER   DEFAULT 100,
                    last_updated TIMESTAMP DEFAULT NOW(),
                    embedding    vector(384)
                );
                """)

    cur.execute("""
                CREATE TABLE reviews
                (
                    review_id   SERIAL PRIMARY KEY,
                    product_id  INTEGER REFERENCES products (product_id),
                    rating      INTEGER CHECK (rating BETWEEN 1 AND 5),
                    review_text TEXT,
                    created_at  TIMESTAMP DEFAULT NOW()
                );
                """)

    cur.execute("""
                CREATE TABLE product_categories
                (
                    product_id    INTEGER REFERENCES products (product_id),
                    category_name TEXT,
                    PRIMARY KEY (product_id, category_name)
                );
                """)

    cur.execute("""
                CREATE INDEX products_embedding_idx ON products
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 100);
                """)

    conn.commit()
    cur.close()
    conn.close()
    print("✓ Database initialized\n")


def generate_products(n: int) -> List[Dict]:
    products = []
    for i in range(n):
        num_reviews = 3 + (i % 6)
        reviews = [
            {
                'rating': 3 + (j % 3),
                'text': f"Review {j}: This product is {'excellent' if j % 2 == 0 else 'good'} quality."
            }
            for j in range(num_reviews)
        ]

        num_categories = 1 + (i % 3)
        categories = [f"Category_{(i + k) % 10}" for k in range(num_categories)]

        products.append({
            'name': f"Product {i}",
            'description': f"Professional grade item {i} with advanced features and durability",
            'stock': random.randint(0, 200),
            'reviews': reviews,
            'categories': categories
        })

    return products


def insert_products(products: List[Dict], conn=None):
    should_close = False
    if conn is None:
        conn = psycopg2.connect(**DB_CONFIG)
        should_close = True

    cur = conn.cursor()

    cur.execute("TRUNCATE products, reviews, product_categories CASCADE;")

    for p in products:
        cur.execute(
            "INSERT INTO products (name, description, stock_count) VALUES (%s, %s, %s) RETURNING product_id",
            (p['name'], p['description'], p['stock'])
        )
        pid = cur.fetchone()[0]

        for review in p['reviews']:
            cur.execute(
                "INSERT INTO reviews (product_id, rating, review_text) VALUES (%s, %s, %s)",
                (pid, review['rating'], review['text'])
            )

        for category in p['categories']:
            cur.execute(
                "INSERT INTO product_categories (product_id, category_name) VALUES (%s, %s)",
                (pid, category)
            )

    conn.commit()
    cur.close()

    if should_close:
        conn.close()


# Scenario 1: Cold Start - Full Insert and Contextual Embedding Generation

def scenario1_unified(size: int, resources: SharedResources, products: List[Dict]) -> float:
    conn = resources.get_db_connection()
    conn.autocommit = False
    cur = conn.cursor()

    start = time.perf_counter()

    try:
        cur.execute("TRUNCATE products, reviews, product_categories CASCADE;")

        product_ids = []
        for p in products:
            cur.execute(
                "INSERT INTO products (name, description, stock_count) VALUES (%s, %s, %s) RETURNING product_id",
                (p['name'], p['description'], p['stock'])
            )
            pid = cur.fetchone()[0]
            product_ids.append(pid)

            for review in p['reviews']:
                cur.execute(
                    "INSERT INTO reviews (product_id, rating, review_text) VALUES (%s, %s, %s)",
                    (pid, review['rating'], review['text'])
                )

            for category in p['categories']:
                cur.execute(
                    "INSERT INTO product_categories (product_id, category_name) VALUES (%s, %s)",
                    (pid, category)
                )

        cur.execute("""
                    WITH review_summary AS (SELECT r.product_id, string_agg(r.review_text, ' | ') AS reviews
                                            FROM (SELECT product_id,
                                                         review_text,
                                                         ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY created_at DESC) as rn
                                                  FROM reviews) r
                                            WHERE r.rn <= 5
                                            GROUP BY r.product_id),
                         cat_summary AS (SELECT product_id, string_agg(category_name, ', ') AS categories
                                         FROM product_categories
                                         GROUP BY product_id),
                         combined AS (SELECT p.product_id,
                                             p.name || '. ' || p.description ||
                                             '. Reviews: ' || COALESCE(rs.reviews, 'No reviews') ||
                                             '. Categories: ' || COALESCE(cs.categories, 'Uncategorized') AS full_text
                                      FROM products p
                                               LEFT JOIN review_summary rs ON p.product_id = rs.product_id
                                               LEFT JOIN cat_summary cs ON p.product_id = cs.product_id
                                      ORDER BY p.product_id),
                         embeddings AS (SELECT sentence_id, embedding
                                        FROM embed_texts_with_ids(
                                                'fastembed',
                                                %s,
                                                (SELECT array_agg(product_id) FROM combined),
                                                (SELECT array_agg(full_text) FROM combined)
                                             ))
                    UPDATE products p
                    SET embedding = embeddings.embedding
                    FROM embeddings
                    WHERE p.product_id = embeddings.sentence_id;
                    """, (PG_GEMBED_MODEL,))

        conn.commit()
        elapsed = time.perf_counter() - start

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.autocommit = True

    return elapsed


def scenario1_distributed(resources: SharedResources, products: List[Dict]) -> float:
    conn = resources.get_db_connection()
    conn.autocommit = False
    cur = conn.cursor()
    model = resources.get_embedding_model()
    client = resources.get_chroma_client()

    try:
        client.delete_collection("products_s1")
    except:
        pass

    collection = client.create_collection("products_s1", metadata={"hnsw:space": "cosine"})

    start = time.perf_counter()

    cur.execute("TRUNCATE products, reviews, product_categories CASCADE;")

    product_ids = []
    for p in products:
        cur.execute(
            "INSERT INTO products (name, description, stock_count) VALUES (%s, %s, %s) RETURNING product_id",
            (p['name'], p['description'], p['stock'])
        )
        pid = cur.fetchone()[0]
        product_ids.append(pid)

        for review in p['reviews']:
            cur.execute(
                "INSERT INTO reviews (product_id, rating, review_text) VALUES (%s, %s, %s)",
                (pid, review['rating'], review['text'])
            )

        for category in p['categories']:
            cur.execute(
                "INSERT INTO product_categories (product_id, category_name) VALUES (%s, %s)",
                (pid, category)
            )

        conn.commit()

    # Build documents directly from the products data
    documents = []
    for i, p in enumerate(products):
        reviews_text = ' | '.join([r['text'] for r in p['reviews']]) if p['reviews'] else 'No reviews'
        categories_text = ', '.join(p['categories']) if p['categories'] else 'Uncategorized'
        doc = f"{p['name']}. {p['description']}. Reviews: {reviews_text}. Categories: {categories_text}"
        documents.append(doc)

    embeddings = list(model.embed(documents))

    collection.add(
        ids=[str(pid) for pid in product_ids],
        embeddings=[e.tolist() for e in embeddings],
        documents=documents
    )

    elapsed = time.perf_counter() - start
    cur.close()

    return elapsed


# Scenario 2: Contextual Embedding Generation (Reviews + Categories)

def scenario2_unified(size: int, resources: SharedResources) -> float:
    conn = resources.get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()

    start = time.perf_counter()

    cur.execute("""
                WITH review_summary AS (SELECT r.product_id, string_agg(r.review_text, ' | ') AS reviews
                                        FROM (SELECT product_id,
                                                     review_text,
                                                     ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY created_at DESC) as rn
                                              FROM reviews) r
                                        WHERE r.rn <= 5
                                        GROUP BY r.product_id),
                     cat_summary AS (SELECT product_id, string_agg(category_name, ', ') AS categories
                                     FROM product_categories
                                     GROUP BY product_id),
                     combined AS (SELECT p.product_id,
                                         p.name || '. ' || p.description ||
                                         '. Reviews: ' || COALESCE(rs.reviews, 'No reviews') ||
                                         '. Categories: ' || COALESCE(cs.categories, 'Uncategorized') AS full_text
                                  FROM products p
                                           LEFT JOIN review_summary rs ON p.product_id = rs.product_id
                                           LEFT JOIN cat_summary cs ON p.product_id = cs.product_id
                                  ORDER BY p.product_id),
                     embeddings AS (SELECT sentence_id, embedding
                                    FROM embed_texts_with_ids(
                                            'fastembed',
                                            %s,
                                            (SELECT array_agg(product_id) FROM combined),
                                            (SELECT array_agg(full_text) FROM combined)
                                         ))
                UPDATE products p
                SET embedding = embeddings.embedding
                FROM embeddings
                WHERE p.product_id = embeddings.sentence_id;
                """, (PG_GEMBED_MODEL,))

    elapsed = time.perf_counter() - start
    cur.close()

    return elapsed


def scenario2_distributed(size: int, resources: SharedResources) -> float:
    conn = resources.get_db_connection()
    cur = conn.cursor()
    model = resources.get_embedding_model()
    client = resources.get_chroma_client()

    try:
        client.delete_collection("products_s2")
    except:
        pass

    collection = client.create_collection("products_s2", metadata={"hnsw:space": "cosine"})

    start = time.perf_counter()

    cur.execute("""
                SELECT p.product_id,
                       p.name,
                       p.description,
                       COALESCE(string_agg(DISTINCT r.review_text, ' | ') FILTER (WHERE r.review_id IS NOT NULL),
                                'No reviews')    AS reviews,
                       COALESCE(string_agg(DISTINCT c.category_name, ', ') FILTER (WHERE c.category_name IS NOT NULL),
                                'Uncategorized') AS categories
                FROM products p
                         LEFT JOIN reviews r ON r.product_id = p.product_id
                         LEFT JOIN product_categories c ON c.product_id = p.product_id
                GROUP BY p.product_id, p.name, p.description
                ORDER BY p.product_id;
                """)

    rows = cur.fetchall()

    documents = [f"{name}. {desc}. Reviews: {reviews}. Categories: {cats}"
                 for _, name, desc, reviews, cats in rows]
    embeddings = list(model.embed(documents))

    collection.add(
        ids=[str(r[0]) for r in rows],
        embeddings=[e.tolist() for e in embeddings],
        documents=documents
    )

    elapsed = time.perf_counter() - start
    cur.close()

    return elapsed


# Scenario 3: Embedding Generation without Context

def scenario3_unified(size: int, resources: SharedResources) -> float:
    conn = resources.get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()

    start = time.perf_counter()

    cur.execute("""
                WITH ordered_products AS (SELECT product_id, name, description
                                          FROM products
                                          ORDER BY product_id),
                     embeddings AS (SELECT sentence_id, embedding
                                    FROM embed_texts_with_ids(
                                            'fastembed',
                                            %s,
                                            (SELECT array_agg(product_id) FROM ordered_products),
                                            (SELECT array_agg(name || '. ' || description) FROM ordered_products)
                                         ))
                UPDATE products p
                SET embedding = embeddings.embedding
                FROM embeddings
                WHERE p.product_id = embeddings.sentence_id;
                """, (PG_GEMBED_MODEL,))

    elapsed = time.perf_counter() - start
    cur.close()

    return elapsed


def scenario3_distributed(size: int, resources: SharedResources) -> float:
    conn = resources.get_db_connection()
    cur = conn.cursor()
    model = resources.get_embedding_model()
    client = resources.get_chroma_client()

    try:
        client.delete_collection("products_s3")
    except:
        pass

    collection = client.create_collection("products_s3")

    start = time.perf_counter()

    cur.execute("SELECT product_id, name, description FROM products ORDER BY product_id;")
    rows = cur.fetchall()

    # Batch embed and insert
    texts = [f"{r[1]}. {r[2]}" for r in rows]
    embeddings = list(model.embed(texts))

    collection.add(
        ids=[str(r[0]) for r in rows],
        embeddings=[e.tolist() for e in embeddings],
        documents=texts
    )

    elapsed = time.perf_counter() - start
    cur.close()

    return elapsed


# Scenario 4: Incremental Updates

def scenario4_unified(size: int, resources: SharedResources) -> float:
    conn = resources.get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()

    iterations = UPDATE_ITERATIONS

    start = time.perf_counter()

    for i in range(iterations):
        product_id = (i % size) + 1
        new_stock = random.randint(0, 200)

        cur.execute("""
                    WITH updated AS (
                        UPDATE products
                            SET stock_count = %s,
                                last_updated = NOW()
                            WHERE product_id = %s RETURNING product_id, name, description, stock_count),
                         embeddings AS (SELECT sentence_id, embedding
                                        FROM embed_texts_with_ids(
                                                'fastembed', %s, (SELECT array_agg(product_id) FROM updated),
                                                (SELECT array_agg(
                                                                name || '. ' || description || '. Stock: ' ||
                                                                CASE
                                                                    WHEN stock_count > 0 THEN 'Available'
                                                                    ELSE 'Out of Stock' END
                                                        )
                                                 FROM updated)
                                             ))
                    UPDATE products p
                    SET embedding = embeddings.embedding
                    FROM embeddings
                    WHERE p.product_id = embeddings.sentence_id;
                    """, (new_stock, product_id, PG_GEMBED_MODEL))

    elapsed = time.perf_counter() - start
    cur.close()

    return elapsed


def scenario4_distributed(size: int, resources: SharedResources) -> float:
    conn = resources.get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()
    model = resources.get_embedding_model()
    client = resources.get_chroma_client()

    iterations = UPDATE_ITERATIONS

    try:
        client.delete_collection("products_s4")
    except:
        pass

    collection = client.create_collection("products_s4")

    cur.execute("SELECT product_id, name, description, stock_count FROM products;")
    rows = cur.fetchall()

    docs = [f"{n}. {d}. Stock: {'Available' if s > 0 else 'Out of Stock'}" for _, n, d, s in rows]
    embeddings = list(model.embed(docs))

    collection.add(
        ids=[str(r[0]) for r in rows],
        embeddings=[e.tolist() for e in embeddings],
        documents=docs
    )

    start = time.perf_counter()

    batch_ids = []
    batch_docs = []

    for i in range(iterations):
        product_id = (i % size) + 1
        new_stock = random.randint(0, 200)

        # Update DB
        cur.execute(
            """
            UPDATE products
            SET stock_count  = %s,
                last_updated = NOW()
            WHERE product_id = %s
            RETURNING name, description;
            """, (new_stock, product_id))

        row = cur.fetchone()
        if not row:
            continue

        text = f"{row[0]}. {row[1]}. Stock: {'Available' if new_stock > 0 else 'Out of Stock'}"

        batch_ids.append(str(product_id))
        batch_docs.append(text)

    if batch_ids:
        batch_embeddings = list(model.embed(batch_docs))

        collection.update(
            ids=batch_ids,
            embeddings=[e.tolist() for e in batch_embeddings],
            documents=batch_docs
        )

    elapsed = time.perf_counter() - start
    cur.close()

    return elapsed


def run_scenario_with_variance(
        scenario_name: str,
        size: int,
        unified_fn,
        distributed_fn,
        resources: SharedResources,
        data: List[Dict]
) -> Tuple[BenchmarkResult, BenchmarkResult]:
    unified_times = []
    distributed_times = []

    for run in range(RUNS_PER_SCENARIO):
        # Unified run
        insert_products(data, resources.get_db_connection())
        time.sleep(0.1)  # Small settle time
        unified_time = unified_fn(size, resources)
        unified_times.append(unified_time)

        # Distributed run
        insert_products(data, resources.get_db_connection())
        time.sleep(0.1)
        distributed_time = distributed_fn(size, resources)
        distributed_times.append(distributed_time)

    unified_result = BenchmarkResult(
        scenario=scenario_name,
        system="Unified (pg_gembed)",
        size=size,
        times_sec=unified_times,
        mean_time=mean(unified_times),
        std_dev=stdev(unified_times) if len(unified_times) > 1 else 0,
        mean_throughput=size / mean(unified_times)
    )

    distributed_result = BenchmarkResult(
        scenario=scenario_name,
        system="Distributed (Chroma)",
        size=size,
        times_sec=distributed_times,
        mean_time=mean(distributed_times),
        std_dev=stdev(distributed_times) if len(distributed_times) > 1 else 0,
        mean_throughput=size / mean(distributed_times)
    )

    return unified_result, distributed_result


def print_header(title: str):
    print("\n" + "=" * 100)
    print(f"{title:^100}")
    print("=" * 100)


def print_result_with_variance(result: BenchmarkResult):
    time_str = f"{result.mean_time:>8.3f}s ±{result.std_dev:>5.3f}s"
    print(f"  {result.system:<30} | {time_str:<20} | {result.mean_throughput:>12.1f}")


def main():
    setup_database()
    resources = SharedResources()

    try:
        # Warmup: initialize all resources
        print("Warming up systems...")
        warmup_data = generate_products(10)
        insert_products(warmup_data, resources.get_db_connection())
        _ = resources.get_embedding_model()
        _ = resources.get_chroma_client()
        print("✓ Systems warmed\n")

        print_header("SCENARIO 1: Cold Start - Full Insert with Contextual Embeddings")
        print("Insert ALL data (products, reviews, categories) + generate embeddings from scratch")
        print("Tests complete system initialization performance\n")
        print(f"{'System':<32} | {'Mean Time':<20} | {'Items/sec':<12}")
        print("-" * 100)

        for size in COLD_START_SIZES:
            data = generate_products(size)

            unified_times = []
            distributed_times = []

            for run in range(RUNS_PER_SCENARIO):
                unified_time = scenario1_unified(size, resources, data)
                unified_times.append(unified_time)
                time.sleep(0.1)

                distributed_time = scenario1_distributed(resources, data)
                distributed_times.append(distributed_time)
                time.sleep(0.1)

            unified_result = BenchmarkResult(
                scenario="Cold Start",
                system="Unified (pg_gembed)",
                size=size,
                times_sec=unified_times,
                mean_time=mean(unified_times),
                std_dev=stdev(unified_times),
                mean_throughput=size / mean(unified_times)
            )

            distributed_result = BenchmarkResult(
                scenario="Cold Start",
                system="Distributed (Chroma)",
                size=size,
                times_sec=distributed_times,
                mean_time=mean(distributed_times),
                std_dev=stdev(distributed_times),
                mean_throughput=size / mean(distributed_times)
            )

            results.append(unified_result)
            results.append(distributed_result)

            print(f"\nSize: {size}")
            print_result_with_variance(unified_result)
            print_result_with_variance(distributed_result)

            speedup = unified_result.mean_throughput / distributed_result.mean_throughput
            print(f"  {'→ Speedup':<30} | {'':<20} | {speedup:>11.2f}x")

        print_header("SCENARIO 2: Contextual Embedding Generation")
        print("Embedding requires: description + aggregated reviews + categories\n")
        print(f"{'System':<32} | {'Mean Time':<20} | {'Items/sec':<12}")
        print("-" * 100)

        for size in AGGREGATION_SIZES:
            data = generate_products(size)

            unified_result, distributed_result = run_scenario_with_variance(
                "Contextual Embedding Generation",
                size,
                scenario2_unified,
                scenario2_distributed,
                resources,
                data
            )

            results.append(unified_result)
            results.append(distributed_result)

            print(f"\nSize: {size}")
            print_result_with_variance(unified_result)
            print_result_with_variance(distributed_result)

            speedup = unified_result.mean_throughput / distributed_result.mean_throughput
            print(f"  {'→ Speedup':<30} | {'':<20} | {speedup:>11.2f}x")

        print_header("SCENARIO 3: Embedding Generation without Context")
        print("Re-embed all products using only Name + Description\n")
        print(f"{'System':<32} | {'Mean Time':<20} | {'Items/sec':<12}")
        print("-" * 100)

        for size in BASIC_EMBEDDING_SIZES:
            data = generate_products(size)

            unified_result, distributed_result = run_scenario_with_variance(
                "Basic Embedding Generation",
                size,
                scenario3_unified,
                scenario3_distributed,
                resources,
                data
            )

            results.append(unified_result)
            results.append(distributed_result)

            print(f"\nSize: {size}")
            print_result_with_variance(unified_result)
            print_result_with_variance(distributed_result)

            speedup = unified_result.mean_throughput / distributed_result.mean_throughput
            print(f"  {'→ Speedup':<30} | {'':<20} | {speedup:>11.2f}x")

        print_header("SCENARIO 4: Incremental Updates")
        print(f"Perform {UPDATE_ITERATIONS} random stock updates per size\n")
        print(f"{'System':<32} | {'Mean Time':<20} | {'Items/sec':<12}")
        print("-" * 100)

        for size in UPDATE_SIZES:
            data = generate_products(size)

            unified_result, distributed_result = run_scenario_with_variance(
                "Incremental Updates",
                size,
                scenario4_unified,
                scenario4_distributed,
                resources,
                data
            )

            unified_result.mean_throughput = UPDATE_ITERATIONS / unified_result.mean_time
            distributed_result.mean_throughput = UPDATE_ITERATIONS / distributed_result.mean_time

            results.append(unified_result)
            results.append(distributed_result)

            print(f"\nSize: {size} (Updates: {UPDATE_ITERATIONS})")
            print_result_with_variance(unified_result)
            print_result_with_variance(distributed_result)

            speedup = unified_result.mean_throughput / distributed_result.mean_throughput
            print(f"  {'→ Speedup':<30} | {'':<20} | {speedup:>11.2f}x")

    finally:
        resources.cleanup()


if __name__ == "__main__":
    main()
