import argparse
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import chromadb
import embed_anything
import numpy as np
from embed_anything import EmbeddingModel
from psycopg2.extras import execute_values
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
)
from utils.benchmark_utils import (
    QDRANT_URL, QDRANT_CONTAINER_NAME,
    BenchmarkResult, ResourceMonitor,
    connect_and_get_pid, get_pg_pid, warmup_pg_connection,
    cleanup_chroma, temp_image_dir, clear_model_cache,
)
from utils.plot_utils import save_single_run_csv

OUTPUT_DIR = Path(__file__).parent / "output"
DATA_DIR = Path(__file__).parent.parent / "data" / "CUSTOMER_IMAGES"

MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM = 512  # CLIP ViT-B/32
TOP_K = 5

INGEST_PER_PERSON = 64  # images per customer used to build the customer profile
QUERY_PER_PERSON = 16  # images per customer reserved for serving queries

CATEGORIES = [
    "Electronics", "Home & Garden", "Sports", "Books", "Clothing",
    "Toys", "Health", "Automotive", "Food", "Office",
]


# =============================================================================
# Data Generation
# =============================================================================

# =============================================================================
# Image Embedding Client (for Qdrant / Chroma competitors)
# =============================================================================

class EmbedAnythingImageClient:
    """Thin wrapper that embeds image files through the embed_anything library."""

    def embed_files(self, paths: List[str]) -> List[List[float]]:
        model = EmbeddingModel.from_pretrained_hf(MODEL_NAME)
        with temp_image_dir(paths, "s3_img_") as tmp:
            res = embed_anything.embed_image_directory(str(tmp), embedder=model)
        if not isinstance(res, list):
            return []
        return [item.embedding for item in res if hasattr(item, 'embedding')]


# ---------------------------------------------------------------------------
# Image-path helpers
# ---------------------------------------------------------------------------

_CUSTOMER_DIRS: List[Path] = []


def load_customer_dirs() -> List[Path]:
    """Return all person directories that have enough images for ingestion + query."""
    global _CUSTOMER_DIRS
    if not _CUSTOMER_DIRS:
        required = INGEST_PER_PERSON + QUERY_PER_PERSON
        _CUSTOMER_DIRS = sorted([
            d for d in DATA_DIR.iterdir()
            if d.is_dir() and
               sum(1 for p in d.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
               >= required
        ])
    return _CUSTOMER_DIRS


def _person_images(d: Path) -> List[str]:
    imgs = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        imgs.extend(sorted(d.glob(ext)))
    return [str(p.absolute()) for p in imgs]


def get_ingest_paths(n_customers: int) -> List[List[str]]:
    """Return INGEST_PER_PERSON image paths for each of the first n_customers."""
    dirs = load_customer_dirs()
    result = []
    for i in range(n_customers):
        imgs = _person_images(dirs[i % len(dirs)])
        result.append(imgs[:INGEST_PER_PERSON])
    return result


def get_query_paths(n_customers: int) -> List[str]:
    """Return QUERY_PER_PERSON image paths (reserved set) for the first n_customers."""
    dirs = load_customer_dirs()
    result = []
    for i in range(n_customers):
        imgs = _person_images(dirs[i % len(dirs)])
        result.extend(imgs[INGEST_PER_PERSON:INGEST_PER_PERSON + QUERY_PER_PERSON])
    return result


def get_person_name(path: str) -> str:
    return Path(path).parent.name


def generate_data(n_customers: int, customer_dirs: List[Path], seed: int = 42) -> Tuple[list, list, list, list]:
    """Generate products (synthetic), customers (real names), purchases, reviews.

    Each customer directory becomes one registered customer with a real name.
    Products are assigned randomly across all customers.
    """
    rng = random.Random(seed)
    n_products = max(50, n_customers * 5)

    adjectives = ["Premium", "Professional", "Essential", "Advanced", "Classic",
                  "Compact", "Deluxe", "Ultra", "Budget", "Standard"]

    # ── Products (no embeddings – catalog only) ───────────────────────────────
    products = []
    for i in range(n_products):
        cat = CATEGORIES[i % len(CATEGORIES)]
        adj = adjectives[i % len(adjectives)]
        products.append({
            'product_id': i + 1,
            'name': f"{adj} {cat} Item {i}",
            'category': cat,
            'price': round(10.0 + rng.random() * 190.0, 2),
            'stock_count': rng.randint(0, 200),
        })

    # ── Customers (real directory names, synthetic tiers) ─────────────────────
    tier_probs = [('bronze', 0.50), ('silver', 0.30), ('gold', 0.15), ('platinum', 0.05)]
    customers = []
    for i, d in enumerate(customer_dirs[:n_customers]):
        r = rng.random()
        cumulative, tier = 0.0, 'bronze'
        for t, p in tier_probs:
            cumulative += p
            if r < cumulative:
                tier = t
                break
        customers.append({'customer_id': i + 1, 'name': d.name, 'tier': tier})

    # ── Purchases ────────────────────────────────────────────────────────────
    purchases, purchase_id = [], 1
    for c in customers:
        n_buys = rng.randint(8, 15) if c['tier'] in ('gold', 'platinum') else rng.randint(2, 6)
        for p in rng.sample(products, min(n_buys, len(products))):
            purchases.append({
                'purchase_id': purchase_id,
                'customer_id': c['customer_id'],
                'product_id': p['product_id'],
            })
            purchase_id += 1

    # ── Reviews ──────────────────────────────────────────────────────────────
    reviews, review_id = [], 1
    cust_by_id = {c['customer_id']: c for c in customers}
    for pu in purchases:
        if rng.random() < 0.70:
            c = cust_by_id[pu['customer_id']]
            rating = rng.randint(3, 5) if c['tier'] in ('gold', 'platinum') else rng.randint(1, 5)
            reviews.append({
                'review_id': review_id,
                'customer_id': pu['customer_id'],
                'product_id': pu['product_id'],
                'rating': rating,
                'text': f"Product {pu['product_id']} review: rating {rating}.",
            })
            review_id += 1

    return products, customers, purchases, reviews


# =============================================================================
# PostgreSQL Setup & Ingestion
# =============================================================================

def setup_pg_schema(conn):
    cur = conn.cursor()
    cur.execute("""
                CREATE
                EXTENSION IF NOT EXISTS vector;
        CREATE
                EXTENSION IF NOT EXISTS pg_gembed;

                DROP TABLE IF EXISTS s3_reviews CASCADE;
                DROP TABLE IF EXISTS s3_purchases CASCADE;
                DROP TABLE IF EXISTS s3_products CASCADE;
                DROP TABLE IF EXISTS s3_customers CASCADE;
                CREATE TABLE s3_customers
                (
                    customer_id SERIAL PRIMARY KEY,
                    name        TEXT NOT NULL,
                    tier        TEXT NOT NULL,
                    embedding   vector(%s)
                );
                CREATE TABLE s3_products
                (
                    product_id  SERIAL PRIMARY KEY,
                    name        TEXT           NOT NULL,
                    category    TEXT           NOT NULL,
                    price       DECIMAL(10, 2) NOT NULL,
                    stock_count INTEGER DEFAULT 0
                );
                CREATE TABLE s3_purchases
                (
                    purchase_id SERIAL PRIMARY KEY,
                    customer_id INTEGER REFERENCES s3_customers (customer_id),
                    product_id  INTEGER REFERENCES s3_products (product_id)
                );
                CREATE TABLE s3_reviews
                (
                    review_id   SERIAL PRIMARY KEY,
                    customer_id INTEGER REFERENCES s3_customers (customer_id),
                    product_id  INTEGER REFERENCES s3_products (product_id),
                    rating      INTEGER CHECK (rating BETWEEN 1 AND 5),
                    review_text TEXT
                );
                """, (EMBEDDING_DIM,))
    conn.commit()
    cur.close()


def populate_pg(conn,
                products: list, customers: list,
                purchases: list, reviews: list,
                customer_embeddings: List[List[float]]):
    """Bulk-insert all four tables and build the HNSW index on customer embeddings."""
    cur = conn.cursor()

    # customers with face/appearance embeddings
    execute_values(cur,
                   "INSERT INTO s3_customers (customer_id, name, tier, embedding) VALUES %s",
                   [(c['customer_id'], c['name'], c['tier'], emb)
                    for c, emb in zip(customers, customer_embeddings)])

    # products (catalog only, no embeddings)
    execute_values(cur,
                   "INSERT INTO s3_products (product_id, name, category, price, stock_count) VALUES %s",
                   [(p['product_id'], p['name'], p['category'], p['price'], p['stock_count'])
                    for p in products])

    # purchases
    execute_values(cur,
                   "INSERT INTO s3_purchases (purchase_id, customer_id, product_id) VALUES %s",
                   [(pu['purchase_id'], pu['customer_id'], pu['product_id']) for pu in purchases])

    # reviews
    execute_values(cur,
                   "INSERT INTO s3_reviews (review_id, customer_id, product_id, rating, review_text) VALUES %s",
                   [(r['review_id'], r['customer_id'], r['product_id'], r['rating'], r['text'])
                    for r in reviews])

    # indexes for the join path
    cur.execute("CREATE INDEX ON s3_purchases(customer_id);")
    cur.execute("CREATE INDEX ON s3_purchases(product_id);")
    cur.execute("CREATE INDEX ON s3_reviews(product_id);")
    cur.execute("CREATE INDEX ON s3_customers(tier);")

    # HNSW vector index on customer embeddings (the ANN target)
    cur.execute(
        "CREATE INDEX ON s3_customers USING hnsw (embedding vector_cosine_ops) "
        "WITH (m=16, ef_construction=100);"
    )
    conn.commit()
    cur.close()


# =============================================================================
# Qdrant / Chroma Setup
# =============================================================================

def setup_qdrant(client: QdrantClient,
                 customers: list, embeddings: List[List[float]]):
    if client.collection_exists("s3_customers"):
        client.delete_collection("s3_customers")
    hnsw_cfg = models.HnswConfigDiff(m=16, ef_construct=100)
    client.create_collection(
        "s3_customers",
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE,
                                    hnsw_config=hnsw_cfg),
    )
    points = [
        PointStruct(
            id=c['customer_id'],
            vector=emb,
            payload={'name': c['name'], 'tier': c['tier']},
        )
        for c, emb in zip(customers, embeddings)
    ]
    client.upsert("s3_customers", points, wait=True)


def setup_chroma(base_path: str, customers: list, embeddings: List[List[float]]):
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    client = chromadb.PersistentClient(path=base_path)
    cfg = {"hnsw": {"space": "cosine", "max_neighbors": 16, "ef_construction": 100}}
    col = client.create_collection("s3_customers", configuration=cfg)
    ids = [str(c['customer_id']) for c in customers]
    metas = [{'customer_id': c['customer_id'], 'name': c['name'], 'tier': c['tier']}
             for c in customers]
    col.add(ids=ids, embeddings=embeddings, metadatas=metas)
    return client, col


# =============================================================================
# Serving Functions
# =============================================================================

def serve_pg_gembed_unified(conn, query_image_paths: List[str], top_k: int):
    """Single SQL: embed_image_directory() inside PG + HNSW on customers + JOIN purchases/reviews.

    PG reads query images from a local temp dir, embeds them in one kernel call,
    laterally joins against the customer HNSW index to find the top-K most
    visually similar customers, then traverses purchases → products filtered by
    review quality – all in a single round-trip.  No blob transfer.
    """
    cur = conn.cursor()
    with temp_image_dir(query_image_paths, "s3_pg_") as tmp:
        cur.execute("""
                    WITH query_embeddings
                             AS (SELECT unnest(embed_image_directory('embed_anything', %s, %s)) AS embedding)
                    SELECT DISTINCT pr.product_id, pr.name
                    FROM query_embeddings q,
                         LATERAL (
                             SELECT c.customer_id
                             FROM s3_customers c
                             ORDER BY c.embedding < - > q.embedding
                                 LIMIT %s ) matched_c
            JOIN s3_purchases pu
                    ON pu.customer_id = matched_c.customer_id
                        JOIN s3_products pr ON pr.product_id = pu.product_id
                    WHERE EXISTS (
                        SELECT 1 FROM s3_reviews r
                        WHERE r.product_id = pr.product_id
                      AND r.rating >= 4
                        );
                    """, (MODEL_NAME, str(tmp), top_k))
        _ = cur.fetchall()
    conn.commit()
    cur.close()


def serve_pg_direct(conn, embed_client: EmbedAnythingImageClient,
                    query_image_paths: List[str], top_k: int):
    """Embed in Python, then single SQL JOIN in PG."""
    embeddings = embed_client.embed_files(query_image_paths)
    cur = conn.cursor()
    sql = """
          SELECT DISTINCT pr.product_id, pr.name
          FROM unnest(%s::vector[]) AS i(embedding)
                   JOIN LATERAL (
              SELECT customer_id
              FROM s3_customers
              ORDER BY embedding < - > i.embedding
                  LIMIT %s ) cu
          ON true
              JOIN s3_purchases pu ON pu.customer_id = cu.customer_id
              JOIN s3_products pr ON pr.product_id = pu.product_id
          WHERE EXISTS (
              SELECT 1 FROM s3_reviews r
              WHERE r.product_id = pr.product_id
            AND r.rating >= 4
              ); \
          """
    for emb in embeddings:
        cur.execute(sql, ([emb], top_k))
        _ = cur.fetchall()
    conn.commit()
    cur.close()


def serve_two_step_qdrant(conn, qd_client: QdrantClient,
                          embed_client: EmbedAnythingImageClient,
                          query_image_paths: List[str], top_k: int):
    """Three-step poly-store: embed in Python → Qdrant ANN on customers → PG JOIN purchases.

    Step 1 (Python): embed_files() on query images – CPU cost within timed window.
    Step 2 (Qdrant): HNSW search on customer collection → top-K customer IDs.
    Step 3 (PG):     JOIN s3_purchases → s3_products → s3_reviews filter – round-trip 2.
    """
    embeddings = embed_client.embed_files(query_image_paths)  # step 1
    requests = [
        models.QueryRequest(query=emb, limit=top_k, with_payload=False)
        for emb in embeddings
    ]
    matched_ids: List[int] = []
    if requests:
        results = qd_client.query_batch_points(
            collection_name="s3_customers", requests=requests)
        for batch in results:
            matched_ids.extend(int(p.id) for p in batch.points)
    matched_ids = list(set(matched_ids))
    if matched_ids:
        cur = conn.cursor()
        cur.execute("""
                    SELECT DISTINCT pr.product_id, pr.name
                    FROM s3_purchases pu
                             JOIN s3_products pr ON pr.product_id = pu.product_id
                    WHERE pu.customer_id = ANY (%s)
                      AND EXISTS (SELECT 1
                                  FROM s3_reviews r
                                  WHERE r.product_id = pr.product_id
                                    AND r.rating >= 4);
                    """, (matched_ids,))
        _ = cur.fetchall()
        cur.close()


def serve_two_step_chroma(conn, chroma_col,
                          embed_client: EmbedAnythingImageClient,
                          query_image_paths: List[str], top_k: int):
    """Three-step poly-store: embed in Python → Chroma ANN on customers → PG JOIN purchases."""
    embeddings = embed_client.embed_files(query_image_paths)  # step 1
    results = chroma_col.query(
        query_embeddings=embeddings,
        n_results=max(1, top_k),
        include=["metadatas"],
    )
    matched_ids: List[int] = []
    for meta_batch in (results.get("metadatas") or []):
        for meta in meta_batch:
            if meta and 'customer_id' in meta:
                matched_ids.append(int(meta['customer_id']))
    matched_ids = list(set(matched_ids))
    if matched_ids:
        cur = conn.cursor()
        cur.execute("""
                    SELECT DISTINCT pr.product_id, pr.name
                    FROM s3_purchases pu
                             JOIN s3_products pr ON pr.product_id = pu.product_id
                    WHERE pu.customer_id = ANY (%s)
                      AND EXISTS (SELECT 1
                                  FROM s3_reviews r
                                  WHERE r.product_id = pr.product_id
                                    AND r.rating >= 4);
                    """, (matched_ids,))
        _ = cur.fetchall()
        cur.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark 5 UC9 – Scenario 3: Customer ID + Purchase History JOIN serving")
    parser.add_argument('--sizes', type=int, nargs='+', required=True,
                        help='Query batch sizes to test')
    parser.add_argument('--db-size', type=int, required=True,
                        help='Customer-DB size to benchmark (number of customers)')
    parser.add_argument('--topk', type=int, default=TOP_K,
                        help='Number of nearest-neighbour customers to match (default: 5)')
    parser.add_argument('--run-id', type=str, default=None)
    args = parser.parse_args()

    test_sizes = args.sizes
    db_size = args.db_size
    top_k = args.topk
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    methods = ['pg_gembed_unified', 'pg_direct', 'two_step_qdrant', 'two_step_chroma']

    print(f"\nBenchmark 5 UC9 – Scenario 3: Customer ID + Purchase History JOIN (Serving)")
    print(f"Run ID  : {run_id}")
    print(f"Batch sizes : {test_sizes}")
    print(f"DB Size : {db_size} customers  TopK: {top_k}")
    print(f"Ingest  : {INGEST_PER_PERSON} imgs/customer → mean embedding")
    print(f"Query   : {QUERY_PER_PERSON} imgs/customer reserved for serving")

    customer_dirs = load_customer_dirs()
    embed_client = EmbedAnythingImageClient()
    py_pid = os.getpid()

    if db_size > len(customer_dirs):
        print(f"  [WARN] Only {len(customer_dirs)} customer dirs available; "
              f"cycling to fill {db_size}.")

    # ── Embed customer profiles (mean per customer) ───────────────────────
    ingest_paths_per_customer = get_ingest_paths(db_size)
    all_ingest_paths = [p for paths in ingest_paths_per_customer for p in paths]
    all_query_paths = get_query_paths(db_size)

    print(f"  Embedding {db_size} × {INGEST_PER_PERSON} = "
          f"{len(all_ingest_paths)} customer images …", flush=True)
    all_ingest_vecs = embed_client.embed_files(all_ingest_paths)

    customer_embeddings: List[List[float]] = []
    for i in range(db_size):
        start = i * INGEST_PER_PERSON
        vecs = all_ingest_vecs[start:start + INGEST_PER_PERSON]
        mean_vec = np.mean(vecs, axis=0).tolist() if vecs else [0.0] * EMBEDDING_DIM
        customer_embeddings.append(mean_vec)

    # ── Generate relational data ──────────────────────────────────────────
    products, customers, purchases, reviews = generate_data(
        db_size, customer_dirs[:db_size])

    eligible = {r['product_id'] for r in reviews if r['rating'] >= 4} & \
               {pu['product_id'] for pu in purchases}
    n_products = len(products)
    print(f"  Eligible products (purchased + rating ≥ 4): "
          f"{len(eligible)} / {n_products} = {len(eligible) / n_products:.1%}", flush=True)

    # ── PG setup ─────────────────────────────────────────────────────────
    conn_pg, pg_pid = connect_and_get_pid()
    warmup_pg_connection(conn_pg)
    setup_pg_schema(conn_pg)
    populate_pg(conn_pg, products, customers, purchases, reviews, customer_embeddings)

    # ── Qdrant setup ─────────────────────────────────────────────────────
    qd_client = QdrantClient(url=QDRANT_URL)
    setup_qdrant(qd_client, customers, customer_embeddings)

    # ── Chroma setup ─────────────────────────────────────────────────────
    ch_path = f"./chroma_s3_{time.time_ns()}"
    ch_client, ch_col = setup_chroma(ch_path, customers, customer_embeddings)

    # ── Warm-up ──────────────────────────────────────────────────────────
    warmup_q = all_query_paths[:2]
    clear_model_cache()
    serve_pg_gembed_unified(conn_pg, warmup_q, top_k)
    serve_pg_direct(conn_pg, embed_client, warmup_q, top_k)
    clear_model_cache()
    serve_two_step_qdrant(conn_pg, qd_client, embed_client, warmup_q, top_k)
    clear_model_cache()
    serve_two_step_chroma(conn_pg, ch_col, embed_client, warmup_q, top_k)
    clear_model_cache()

    results_by_size = {s: {m: None for m in methods} for s in test_sizes}

    try:
        # ── Benchmark loop over batch sizes ──────────────────────────────────
        for size in test_sizes:
            print(f"\n{'─' * 60}", flush=True)
            print(f"Query Batch size: {size} queries", flush=True)

            query_image_paths = all_query_paths[:size]
            n_eff_queries = len(query_image_paths)
            if n_eff_queries == 0:
                print("  [WARN] Not enough query images available.")
                continue

            elapsed, _, stats = ResourceMonitor.measure(
                py_pid, get_pg_pid(conn_pg),
                lambda: serve_pg_gembed_unified(conn_pg, query_image_paths, top_k))
            results_by_size[size]['pg_gembed_unified'] = BenchmarkResult(elapsed, stats)
            print(f"  pg_gembed_unified : {elapsed:.3f}s  "
                  f"({n_eff_queries / elapsed:.1f} q/s)", flush=True)

            elapsed, _, stats = ResourceMonitor.measure(
                py_pid, get_pg_pid(conn_pg),
                lambda: serve_pg_direct(conn_pg, embed_client, query_image_paths, top_k))
            results_by_size[size]['pg_direct'] = BenchmarkResult(elapsed, stats)
            print(f"  pg_direct         : {elapsed:.3f}s  "
                  f"({n_eff_queries / elapsed:.1f} q/s)", flush=True)
            clear_model_cache()

            elapsed, _, stats = ResourceMonitor.measure(
                py_pid, get_pg_pid(conn_pg),
                lambda: serve_two_step_qdrant(conn_pg, qd_client, embed_client,
                                              query_image_paths, top_k),
                container_name=QDRANT_CONTAINER_NAME)
            results_by_size[size]['two_step_qdrant'] = BenchmarkResult(elapsed, stats)
            print(f"  two_step_qdrant   : {elapsed:.3f}s  "
                  f"({n_eff_queries / elapsed:.1f} q/s)", flush=True)
            clear_model_cache()

            elapsed, _, stats = ResourceMonitor.measure(
                py_pid, get_pg_pid(conn_pg),
                lambda: serve_two_step_chroma(conn_pg, ch_col, embed_client,
                                              query_image_paths, top_k))
            results_by_size[size]['two_step_chroma'] = BenchmarkResult(elapsed, stats)
            print(f"  two_step_chroma   : {elapsed:.3f}s  "
                  f"({n_eff_queries / elapsed:.1f} q/s)", flush=True)
            clear_model_cache()

    finally:
        conn_pg.close()
        if qd_client.collection_exists("s3_customers"):
            qd_client.delete_collection("s3_customers")
        qd_client.close()
        cleanup_chroma(ch_client, ch_path)

    # ── Collect & save results ────────────────────────────────────────────────
    all_results = []
    for size in test_sizes:
        entry = {'size': size}
        for method in methods:
            r = results_by_size[size][method]
            if r is None:
                continue
            n_q = size
            entry[method] = {
                'time_s': r.time_s,
                'throughput': n_q / r.time_s if r.time_s > 0 else 0,
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

    output_dir = OUTPUT_DIR / "scenario3_serving"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_single_run_csv(all_results, output_dir, run_id, methods)
    print("\nDone.")


if __name__ == "__main__":
    main()
