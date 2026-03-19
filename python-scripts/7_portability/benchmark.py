#!/usr/bin/env python3
import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import mysql.connector
import psutil
import redis
from utils.benchmark_utils import (
    BenchmarkResult,
    ResourceMonitor,
    connect_and_get_pid,
    get_pg_pid,
    clear_model_cache,
)
from utils.plot_utils import save_single_run_csv
from data.loader import get_review_texts

OUTPUT_DIR = Path(__file__).parent / "output"

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-large-en-v1.5",
]
BACKEND = "embed_anything"

MYSQL_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "database": "benchmark",
}

REDIS_HOST = "127.0.0.1"
REDIS_PORT = 6379


# =============================================================================
# PostgreSQL (pg_gembed)
# =============================================================================

def warmup_pg(conn, model: str) -> None:
    cur = conn.cursor()
    cur.execute(
        "SELECT embed_texts(%s, %s, ARRAY['warmup text']::text[])",
        (BACKEND, model),
    )
    cur.fetchall()
    conn.commit()
    cur.close()


def run_pg(conn, model: str, texts: List[str]) -> None:
    cur = conn.cursor()
    cur.execute(
        "SELECT embed_texts(%s, %s, %s::text[])",
        (BACKEND, model, texts),
    )
    _ = cur.fetchall()
    cur.close()


# =============================================================================
# MySQL (mysql_gembed)
# =============================================================================

def connect_mysql() -> mysql.connector.MySQLConnection:
    """Connect to MySQL; create the benchmark database if it doesn't exist."""
    cfg = dict(MYSQL_CONFIG)
    cfg.pop("database", None)
    conn = mysql.connector.connect(**cfg)
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_CONFIG['database']}")
    cur.close()
    conn.database = MYSQL_CONFIG["database"]

    # Install the component if not already installed
    try:
        cur = conn.cursor()
        cur.execute("INSTALL COMPONENT 'file://component_mysql_gembed'")
        cur.close()
    except Exception:
        pass  # Already installed

    return conn


def get_mysql_pid(conn) -> Optional[int]:
    try:
        cur = conn.cursor()
        cur.execute("SELECT CONNECTION_ID()")
        cid = cur.fetchone()[0]
        cur.close()
        # Map MySQL connection ID to OS PID via information_schema
        cur = conn.cursor()
        cur.execute(
            "SELECT PROCESSLIST_OS_ID FROM performance_schema.threads "
            "WHERE PROCESSLIST_ID = %s", (cid,)
        )
        row = cur.fetchone()
        cur.close()
        return int(row[0]) if row and row[0] else None
    except Exception:
        return None


def warmup_mysql(conn, model: str) -> None:
    cur = conn.cursor()
    cur.execute(
        "SELECT EMBED_TEXTS(%s, %s, %s)",
        (BACKEND, model, '["warmup text"]'),
    )
    cur.fetchall()
    cur.close()


def run_mysql(conn, model: str, texts: List[str]) -> None:
    import json
    json_texts = json.dumps(texts)
    cur = conn.cursor()
    cur.execute(
        "SELECT EMBED_TEXTS(%s, %s, %s)",
        (BACKEND, model, json_texts),
    )
    _ = cur.fetchall()
    cur.close()


# =============================================================================
# Redis (redis_gembed)
# =============================================================================

def connect_redis() -> redis.Redis:
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)


def get_redis_server_pid() -> Optional[int]:
    """Find the redis-server process PID via psutil."""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = proc.info["name"] or ""
            if "redis-server" in name or "redis-server" in " ".join(proc.info.get("cmdline") or []):
                return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None


def warmup_redis(r: redis.Redis, model: str) -> None:
    r.execute_command("G.EMBEDS", BACKEND, model, "warmup text")


def run_redis(r: redis.Redis, model: str, texts: List[str]) -> None:
    _ = r.execute_command("G.EMBEDS", BACKEND, model, *texts)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark 7: Portability (pg vs mysql vs redis)")
    parser.add_argument("--sizes", type=int, nargs="+", required=True,
                        help="Number of texts to embed per run")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run identifier for file naming")
    parser.add_argument("--adapters", type=str, nargs="+",
                        default=["pg", "mysql", "redis"],
                        choices=["pg", "mysql", "redis"],
                        help="Adapters to benchmark (default: all three)")
    args = parser.parse_args()

    test_sizes = args.sizes
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    adapters = args.adapters

    # Method name: <adapter>_<model>  e.g. "pg_sentence-transformers/all-MiniLM-L6-v2"
    methods = [
        f"{adapter}_{model}"
        for model in MODELS
        for adapter in adapters
    ]

    print(f"\nBenchmark 7 — Portability")
    print(f"Run ID   : {run_id}")
    print(f"Sizes    : {test_sizes}")
    print(f"Adapters : {adapters}")
    print(f"Models   : {MODELS}")
    print(f"Embedder : {BACKEND}")

    py_pid = os.getpid()

    # Pre-load texts for all sizes
    texts_by_size = {size: get_review_texts(size, shuffle=False) for size in test_sizes}

    # ------------------------------------------------------------------
    # Warm-up phase (all adapters × all models, excluded from timing)
    # ------------------------------------------------------------------
    print("\nWarming up...")

    if "pg" in adapters:
        for model in MODELS:
            conn, _ = connect_and_get_pid()
            try:
                warmup_pg(conn, model)
                conn.commit()
            finally:
                conn.close()
            clear_model_cache()

    if "mysql" in adapters:
        for model in MODELS:
            try:
                conn_my = connect_mysql()
                warmup_mysql(conn_my, model)
                conn_my.close()
            except Exception as e:
                print(f"  [WARN] MySQL warmup failed for {model}: {e}")
            clear_model_cache()

    if "redis" in adapters:
        for model in MODELS:
            try:
                r = connect_redis()
                warmup_redis(r, model)
                r.close()
            except Exception as e:
                print(f"  [WARN] Redis warmup failed for {model}: {e}")
            clear_model_cache()

    # ------------------------------------------------------------------
    # Benchmark loop
    # ------------------------------------------------------------------
    results_by_size = {size: {m: None for m in methods} for size in test_sizes}

    for size in test_sizes:
        print(f"\nSize: {size}", flush=True)
        texts = texts_by_size[size]

        for model in MODELS:
            # ── PostgreSQL ──────────────────────────────────────────────
            if "pg" in adapters:
                method_name = f"pg_{model}"
                conn, pg_pid = connect_and_get_pid()
                try:
                    warmup_pg(conn, model)
                    conn.commit()
                    elapsed, _, stats = ResourceMonitor.measure(
                        py_pid, pg_pid,
                        lambda: run_pg(conn, model, texts),
                    )
                    conn.commit()
                    results_by_size[size][method_name] = BenchmarkResult(elapsed, stats)
                    print(f"  pg    {model}: {elapsed:.3f}s "
                          f"({size / elapsed:.1f} texts/s)", flush=True)
                except Exception as e:
                    print(f"  pg    {model}: FAILED — {e}", flush=True)
                finally:
                    conn.close()
                clear_model_cache()

            # ── MySQL ───────────────────────────────────────────────────
            if "mysql" in adapters:
                method_name = f"mysql_{model}"
                try:
                    conn_my = connect_mysql()
                    my_pid = get_mysql_pid(conn_my)
                    warmup_mysql(conn_my, model)
                    elapsed, _, stats = ResourceMonitor.measure(
                        py_pid, my_pid,
                        lambda: run_mysql(conn_my, model, texts),
                    )
                    results_by_size[size][method_name] = BenchmarkResult(elapsed, stats)
                    print(f"  mysql {model}: {elapsed:.3f}s "
                          f"({size / elapsed:.1f} texts/s)", flush=True)
                except Exception as e:
                    print(f"  mysql {model}: FAILED — {e}", flush=True)
                finally:
                    try:
                        conn_my.close()
                    except Exception:
                        pass
                clear_model_cache()

            # ── Redis ───────────────────────────────────────────────────
            if "redis" in adapters:
                method_name = f"redis_{model}"
                try:
                    r = connect_redis()
                    redis_pid = get_redis_server_pid()
                    warmup_redis(r, model)
                    elapsed, _, stats = ResourceMonitor.measure(
                        py_pid, redis_pid,
                        lambda: run_redis(r, model, texts),
                    )
                    results_by_size[size][method_name] = BenchmarkResult(elapsed, stats)
                    print(f"  redis {model}: {elapsed:.3f}s "
                          f"({size / elapsed:.1f} texts/s)", flush=True)
                except Exception as e:
                    print(f"  redis {model}: FAILED — {e}", flush=True)
                finally:
                    try:
                        r.close()
                    except Exception:
                        pass
                clear_model_cache()

    # ------------------------------------------------------------------
    # Collect and save results
    # ------------------------------------------------------------------
    all_results = []
    for size in test_sizes:
        entry = {"size": size}
        for method_name in methods:
            result = results_by_size[size][method_name]
            if result is None:
                continue
            entry[method_name] = {
                "time_s": result.time_s,
                "throughput": size / result.time_s if result.time_s > 0 else 0,
                "py_cpu": result.stats.py_cpu,
                "py_mem_delta": result.stats.py_delta_mb,
                "py_mem_peak": result.stats.py_peak_mb,
                # pg_* fields: populated for pg/mysql adapters (the DB-side process)
                "pg_cpu": result.stats.pg_cpu,
                "pg_mem_delta": result.stats.pg_delta_mb,
                "pg_mem_peak": result.stats.pg_peak_mb,
                "sys_cpu": result.stats.sys_cpu,
                "sys_mem": result.stats.sys_mem_mb,
            }
        all_results.append(entry)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_single_run_csv(all_results, OUTPUT_DIR, run_id, methods)
    print("\nRun completed!")


if __name__ == "__main__":
    main()
