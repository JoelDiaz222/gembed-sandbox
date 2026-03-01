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
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
)
from utils.benchmark_utils import (
    QDRANT_URL, QDRANT_CONTAINER_NAME,
    BenchmarkResult, ResourceMonitor,
    connect_and_get_pid, get_pg_pid, warmup_pg_connection,
    cleanup_chroma, temp_image_dir,
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

# The category used for the hybrid filter query (most even distribution).
TARGET_CATEGORY = "Electronics"

# =============================================================================
# Image Embedding Client (for Qdrant / Chroma competitors)
# =============================================================================

_model_cache: dict = {}


class EmbedAnythingImageClient:
    """Thin wrapper that embeds image files through the embed_anything library."""

    def embed_files(self, paths: List[str]) -> List[List[float]]:
        model = self._get_model()
        with temp_image_dir(paths, "s4_img_") as tmp:
            res = embed_anything.embed_image_directory(str(tmp), embedder=model)
        if not isinstance(res, list):
            return []
        return [item.embedding for item in res if hasattr(item, 'embedding')]

    @staticmethod
    def _get_model():
        if MODEL_NAME not in _model_cache:
            _model_cache[MODEL_NAME] = EmbeddingModel.from_pretrained_hf(MODEL_NAME)
        return _model_cache[MODEL_NAME]


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
    """Return QUERY_PER_PERSON image paths (reserved set) for each of the first n_customers."""
    dirs = load_customer_dirs()
    result = []
    for i in range(n_customers):
        imgs = _person_images(dirs[i % len(dirs)])
        result.extend(imgs[INGEST_PER_PERSON:INGEST_PER_PERSON + QUERY_PER_PERSON])
    return result


# =============================================================================
# Data Generation
# =============================================================================

def generate_data(n_customers: int, customer_dirs: List[Path],
                  seed: int = 42) -> Tuple[list, list, list]:
    """Generate products (synthetic catalog), customers (real names + tiers), purchases."""
    rng = random.Random(seed)
    n_products = max(50, n_customers * 5)
    adjectives = ["Premium", "Professional", "Essential", "Advanced", "Classic",
                  "Compact", "Deluxe", "Ultra", "Budget", "Standard"]

    # ── Products (no embeddings) ──────────────────────────────────────────────────
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

    return products, customers, purchases


def compute_price_range(products: list, category: str, frac: float) -> Tuple[float, float]:
    """Return (price_lo, price_hi) such that ≈frac of *category* products pass.

    Products are sorted by price within the category; the range covers the
    cheapest `frac` fraction of them, giving exact quantile-based control.
    """
    cat_prices = sorted(p['price'] for p in products if p['category'] == category)
    if not cat_prices:
        return 0.0, 0.0
    n_eligible = max(1, round(frac * len(cat_prices)))
    price_lo = cat_prices[0]
    price_hi = cat_prices[min(n_eligible - 1, len(cat_prices) - 1)]
    return price_lo, price_hi


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

                DROP TABLE IF EXISTS s4_purchases CASCADE;
                DROP TABLE IF EXISTS s4_products CASCADE;
                DROP TABLE IF EXISTS s4_customers CASCADE;
                CREATE TABLE s4_customers
                (
                    customer_id SERIAL PRIMARY KEY,
                    name        TEXT NOT NULL,
                    tier        TEXT NOT NULL,
                    embedding   vector(%s)
                );
                CREATE TABLE s4_products
                (
                    product_id  SERIAL PRIMARY KEY,
                    name        TEXT           NOT NULL,
                    category    TEXT           NOT NULL,
                    price       DECIMAL(10, 2) NOT NULL,
                    stock_count INTEGER DEFAULT 0
                );
                CREATE TABLE s4_purchases
                (
                    purchase_id SERIAL PRIMARY KEY,
                    customer_id INTEGER REFERENCES s4_customers (customer_id),
                    product_id  INTEGER REFERENCES s4_products (product_id)
                );
                """, (EMBEDDING_DIM,))
    cur.close()


def populate_pg(conn, products: list, customers: list, purchases: list,
                customer_embeddings: List[List[float]]):
    cur = conn.cursor()
    execute_values(cur,
                   "INSERT INTO s4_customers (customer_id, name, tier, embedding) VALUES %s",
                   [(c['customer_id'], c['name'], c['tier'], np.array(emb))
                    for c, emb in zip(customers, customer_embeddings)])
    execute_values(cur,
                   "INSERT INTO s4_products (product_id, name, category, price, stock_count) VALUES %s",
                   [(p['product_id'], p['name'], p['category'], p['price'], p['stock_count'])
                    for p in products])
    execute_values(cur,
                   "INSERT INTO s4_purchases (purchase_id, customer_id, product_id) VALUES %s",
                   [(pu['purchase_id'], pu['customer_id'], pu['product_id']) for pu in purchases])

    # B-tree index for structured filter on purchased products
    cur.execute("CREATE INDEX ON s4_products (category, price);")
    cur.execute("CREATE INDEX ON s4_purchases (customer_id);")
    cur.execute("CREATE INDEX ON s4_purchases (product_id);")
    # HNSW index on customer embeddings (the ANN target)
    cur.execute(
        "CREATE INDEX ON s4_customers USING hnsw (embedding vector_cosine_ops) "
        "WITH (m=16, ef_construction=100);"
    )
    cur.close()


# =============================================================================
# Qdrant / Chroma Setup
# =============================================================================

def setup_qdrant(client: QdrantClient, customers: list, embeddings: List[List[float]]):
    if client.collection_exists("s4_customers"):
        client.delete_collection("s4_customers")
    hnsw_cfg = models.HnswConfigDiff(m=16, ef_construct=100)
    client.create_collection(
        "s4_customers",
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
    client.upsert("s4_customers", points, wait=True)


def setup_chroma(base_path: str, customers: list, embeddings: List[List[float]]):
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    client = chromadb.PersistentClient(path=base_path)
    cfg = {"hnsw": {"space": "cosine", "max_neighbors": 16, "ef_construction": 100}}
    col = client.create_collection("s4_customers", configuration=cfg)
    ids = [str(c['customer_id']) for c in customers]
    metas = [{'customer_id': c['customer_id'], 'name': c['name'], 'tier': c['tier']}
             for c in customers]
    col.add(ids=ids, embeddings=embeddings, metadatas=metas)
    return client, col


# =============================================================================
# Serving Functions
# =============================================================================

def serve_pg_gembed_unified(conn, query_image_paths: List[str],
                            category: str, price_lo: float, price_hi: float,
                            top_k: int):
    """Single SQL: embed_image_directory() inside PG + HNSW on customers + JOIN purchases filter.

    PG embeds the query images in one kernel call, laterally joins against the
    customer HNSW index (top-K visual match), then traverses purchases → products
    with a WHERE category + price predicate – all in a single round-trip.
    """
    cur = conn.cursor()
    with temp_image_dir(query_image_paths, "s4_pg_") as tmp:
        cur.execute("""
                    WITH query_embeddings
                             AS (SELECT unnest(embed_image_directory('embed_anything', %s, %s)) AS embedding)
                    SELECT DISTINCT pr.product_id, pr.name
                    FROM query_embeddings q,
                         LATERAL (
                             SELECT c.customer_id
                             FROM s4_customers c
                             ORDER BY c.embedding <-> q.embedding
                                 LIMIT %s ) matched_c
            JOIN s4_purchases pu
                    ON pu.customer_id = matched_c.customer_id
                        JOIN s4_products pr ON pr.product_id = pu.product_id
                    WHERE pr.category = %s
                      AND pr.price BETWEEN %s
                      AND %s;
                    """, (MODEL_NAME, str(tmp), top_k, category, price_lo, price_hi))
        _ = cur.fetchall()
    cur.close()


def serve_qdrant_native(conn, qd_client: QdrantClient,
                        embed_client: EmbedAnythingImageClient,
                        query_image_paths: List[str],
                        category: str, price_lo: float, price_hi: float, top_k: int):
    """Three-step: embed in Python (timed) → Qdrant ANN on customers → PG JOIN purchases."""
    embeddings = embed_client.embed_files(query_image_paths)  # ← in timed window
    requests = [
        models.QueryRequest(query=emb, limit=top_k, with_payload=False)
        for emb in embeddings
    ]
    matched_ids: List[int] = []
    if requests:
        results = qd_client.query_batch_points(
            collection_name="s4_customers", requests=requests)
        for batch in results:
            matched_ids.extend(int(p.id) for p in batch.points)
    matched_ids = list(set(matched_ids))
    if matched_ids:
        cur = conn.cursor()
        cur.execute("""
                    SELECT DISTINCT pr.product_id, pr.name
                    FROM s4_purchases pu
                             JOIN s4_products pr ON pr.product_id = pu.product_id
                    WHERE pu.customer_id = ANY (%s)
                      AND pr.category = %s
                      AND pr.price BETWEEN %s AND %s;
                    """, (matched_ids, category, price_lo, price_hi))
        _ = cur.fetchall()
        cur.close()


def serve_chroma_native(conn, chroma_col,
                        embed_client: EmbedAnythingImageClient,
                        query_image_paths: List[str],
                        category: str, price_lo: float, price_hi: float,
                        top_k: int):
    """Three-step: embed in Python (timed) → Chroma ANN on customers → PG JOIN purchases."""
    embeddings = embed_client.embed_files(query_image_paths)  # ← in timed window
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
                    FROM s4_purchases pu
                             JOIN s4_products pr ON pr.product_id = pu.product_id
                    WHERE pu.customer_id = ANY (%s)
                      AND pr.category = %s
                      AND pr.price BETWEEN %s AND %s;
                    """, (matched_ids, category, price_lo, price_hi))
        _ = cur.fetchall()
        cur.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark 5 UC9 – Scenario 4: Customer ID + Hybrid Product Filter")
    parser.add_argument('--sizes', type=int, nargs='+', required=True,
                        help='Query batch sizes to test')
    parser.add_argument('--db-size', type=int, required=True,
                        help='Number of images in the DB (n_customers = db_size // INGEST_PER_PERSON)')
    parser.add_argument('--fracs', type=float, nargs='+',
                        default=[0.05, 0.10, 0.20, 0.40, 0.60, 0.80],
                        help='Filter fractions α to test')
    parser.add_argument('--topk', type=int, default=TOP_K,
                        help='Number of nearest-neighbour customers to match (default: 5)')
    parser.add_argument('--run-id', type=str, default=None)
    args = parser.parse_args()

    test_sizes = args.sizes
    db_size = args.db_size
    n_customers = max(1, db_size // INGEST_PER_PERSON)
    fracs = sorted(args.fracs)
    top_k = args.topk
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    base_methods = ['pg_gembed_unified', 'qdrant_native', 'chroma_native']
    methods = [f"{m}_f{f}" for m in base_methods for f in fracs]

    print(f"\nBenchmark 5 UC9 – Scenario 4: Customer ID + Hybrid Product Filter")
    print(f"Run ID         : {run_id}")
    print(f"Batch sizes    : {test_sizes}")
    print(f"DB size        : {db_size} images → {n_customers} customers")
    print(f"Filter fracs α : {fracs}  TopK: {top_k}")
    print(f"Category filter: {TARGET_CATEGORY}")
    print(f"Ingest  : {INGEST_PER_PERSON} imgs/customer → mean embedding")
    print(f"Query   : {QUERY_PER_PERSON} imgs/customer reserved for serving")

    customer_dirs = load_customer_dirs()
    embed_client = EmbedAnythingImageClient()
    py_pid = os.getpid()

    all_results = []  # collected across all (db_size × frac) combinations

    if n_customers > len(customer_dirs):
        print(f"  [WARN] Only {len(customer_dirs)} customer dirs available; "
              f"cycling to fill {n_customers}.")

    # ── Image paths ───────────────────────────────────────────────────────
    ingest_paths_per_customer = get_ingest_paths(n_customers)
    all_ingest_paths = [p for paths in ingest_paths_per_customer for p in paths]
    all_query_paths = get_query_paths(n_customers)

    # ── Embed customer profiles (mean per customer) ───────────────────────
    print(f"  Embedding {n_customers} × {INGEST_PER_PERSON} = "
          f"{len(all_ingest_paths)} customer images …", flush=True)
    all_ingest_vecs = embed_client.embed_files(all_ingest_paths)

    customer_embeddings: List[List[float]] = []
    for i in range(n_customers):
        start = i * INGEST_PER_PERSON
        vecs = all_ingest_vecs[start:start + INGEST_PER_PERSON]
        mean_vec = np.mean(vecs, axis=0).tolist() if vecs else [0.0] * EMBEDDING_DIM
        customer_embeddings.append(mean_vec)

    # ── Generate relational data ──────────────────────────────────────────
    products, customers, purchases = generate_data(
        n_customers, customer_dirs[:n_customers])

    cat_products = [p for p in products if p['category'] == TARGET_CATEGORY]
    n_cat = len(cat_products)
    print(f"  Products in '{TARGET_CATEGORY}': {n_cat}")

    # Pre-compute price ranges for every frac level
    frac_ranges = {}
    print(f"  α     → (price_lo, price_hi)  #eligible")
    for frac in fracs:
        lo, hi = compute_price_range(products, TARGET_CATEGORY, frac)
        n_elig = sum(1 for p in cat_products if lo <= p['price'] <= hi)
        frac_ranges[frac] = (lo, hi, n_elig)
        print(f"  {frac:.2f}  → ({lo:.2f}, {hi:.2f})  {n_elig}")

    # ── PG setup ─────────────────────────────────────────────────────────
    conn_pg, pg_pid = connect_and_get_pid()
    register_vector(conn_pg)
    warmup_pg_connection(conn_pg)
    setup_pg_schema(conn_pg)
    populate_pg(conn_pg, products, customers, purchases, customer_embeddings)

    # ── Qdrant setup ─────────────────────────────────────────────────────
    qd_client = QdrantClient(url=QDRANT_URL)
    setup_qdrant(qd_client, customers, customer_embeddings)

    # ── Chroma setup ─────────────────────────────────────────────────────
    ch_path = f"./chroma_s4_{time.time_ns()}"
    ch_client, ch_col = setup_chroma(ch_path, customers, customer_embeddings)

    # ── Warm-up (smallest frac) ──────────────────────────────────────────
    wlo, whi, _ = frac_ranges[fracs[0]]
    warmup_q = all_query_paths[:2]
    serve_pg_gembed_unified(conn_pg, warmup_q, TARGET_CATEGORY, wlo, whi, top_k)
    serve_qdrant_native(conn_pg, qd_client, embed_client, warmup_q,
                        TARGET_CATEGORY, wlo, whi, top_k)
    serve_chroma_native(conn_pg, ch_col, embed_client, warmup_q,
                        TARGET_CATEGORY, wlo, whi, top_k)

    try:
        # ── Benchmark loop over Batch Sizes ──────────────────────────────────
        for size in test_sizes:
            print(f"\n  {'═' * 56}", flush=True)
            print(f"  Query Batch size: {size} queries", flush=True)

            query_image_paths = all_query_paths[:size]
            n_eff_queries = len(query_image_paths)
            if n_eff_queries == 0:
                print("  [WARN] Not enough query images available.")
                continue

            entry = {'size': size}

            for frac in fracs:
                price_lo, price_hi, n_eligible = frac_ranges[frac]
                print(f"\n    {'─' * 40}", flush=True)
                print(f"    α = {frac} ({n_eligible} eligible products)", flush=True)

                elapsed, _, stats = ResourceMonitor.measure(
                    py_pid, get_pg_pid(conn_pg),
                    lambda lo=price_lo, hi=price_hi: serve_pg_gembed_unified(
                        conn_pg, query_image_paths, TARGET_CATEGORY, lo, hi, top_k))
                conn_pg.commit()
                entry[f'pg_gembed_unified_f{frac}'] = {
                    'time_s': elapsed,
                    'throughput': n_eff_queries / elapsed if elapsed > 0 else 0,
                    'py_cpu': stats.py_cpu, 'py_mem_delta': stats.py_delta_mb,
                    'py_mem_peak': stats.py_peak_mb,
                    'pg_cpu': stats.pg_cpu, 'pg_mem_delta': stats.pg_delta_mb,
                    'pg_mem_peak': stats.pg_peak_mb,
                    'qd_cpu': stats.qd_cpu, 'qd_mem_delta': stats.qd_delta_mb,
                    'qd_mem_peak': stats.qd_peak_mb,
                    'sys_cpu': stats.sys_cpu, 'sys_mem': stats.sys_mem_mb,
                }
                print(f"      pg_gembed_unified : {elapsed:.3f}s  ({n_eff_queries / elapsed:.1f} q/s)")

                elapsed, _, stats = ResourceMonitor.measure(
                    py_pid, None,
                    lambda lo=price_lo, hi=price_hi: serve_qdrant_native(
                        conn_pg, qd_client, embed_client, query_image_paths,
                        TARGET_CATEGORY, lo, hi, top_k),
                    container_name=QDRANT_CONTAINER_NAME)
                conn_pg.commit()
                entry[f'qdrant_native_f{frac}'] = {
                    'time_s': elapsed,
                    'throughput': n_eff_queries / elapsed if elapsed > 0 else 0,
                    'py_cpu': stats.py_cpu, 'py_mem_delta': stats.py_delta_mb,
                    'py_mem_peak': stats.py_peak_mb,
                    'pg_cpu': stats.pg_cpu, 'pg_mem_delta': stats.pg_delta_mb,
                    'pg_mem_peak': stats.pg_peak_mb,
                    'qd_cpu': stats.qd_cpu, 'qd_mem_delta': stats.qd_delta_mb,
                    'qd_mem_peak': stats.qd_peak_mb,
                    'sys_cpu': stats.sys_cpu, 'sys_mem': stats.sys_mem_mb,
                }
                print(f"      qdrant_native     : {elapsed:.3f}s  ({n_eff_queries / elapsed:.1f} q/s)")

                elapsed, _, stats = ResourceMonitor.measure(
                    py_pid, None,
                    lambda lo=price_lo, hi=price_hi: serve_chroma_native(
                        conn_pg, ch_col, embed_client, query_image_paths,
                        TARGET_CATEGORY, lo, hi, top_k))
                conn_pg.commit()
                entry[f'chroma_native_f{frac}'] = {
                    'time_s': elapsed,
                    'throughput': n_eff_queries / elapsed if elapsed > 0 else 0,
                    'py_cpu': stats.py_cpu, 'py_mem_delta': stats.py_delta_mb,
                    'py_mem_peak': stats.py_peak_mb,
                    'pg_cpu': stats.pg_cpu, 'pg_mem_delta': stats.pg_delta_mb,
                    'pg_mem_peak': stats.pg_peak_mb,
                    'qd_cpu': stats.qd_cpu, 'qd_mem_delta': stats.qd_delta_mb,
                    'qd_mem_peak': stats.qd_peak_mb,
                    'sys_cpu': stats.sys_cpu, 'sys_mem': stats.sys_mem_mb,
                }
                print(f"      chroma_native     : {elapsed:.3f}s  ({n_eff_queries / elapsed:.1f} q/s)")

            all_results.append(entry)

    finally:
        conn_pg.close()
        if qd_client.collection_exists("s4_customers"):
            qd_client.delete_collection("s4_customers")
        qd_client.close()
        cleanup_chroma(ch_client, ch_path)

    # ── Save results ──────────────────────────────────────────────────────────
    output_dir = OUTPUT_DIR / "scenario4_serving"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_single_run_csv(all_results, output_dir, run_id, methods)
    print("\nDone.")


if __name__ == "__main__":
    main()
