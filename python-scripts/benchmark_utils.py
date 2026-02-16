"""
Shared benchmark utilities for consistent experiment execution.

Provides common infrastructure including resource monitoring, statistics,
data classes, client wrappers, and database connection helpers.

All benchmarks should import from this module to ensure:
- Consistent timing methodology (external, via ResourceMonitor.measure())
- Consistent memory unit reporting (MiB = 1024*1024 bytes)
- Consistent PostgreSQL connection warm-up
- Consistent statistics computation
"""

import os
import shutil
import time
from dataclasses import dataclass
from statistics import mean, stdev, median, quantiles
from typing import Callable, List, Optional, Tuple

import embed_anything
import psutil
import psycopg2
from embed_anything import EmbeddingModel, WhichModel

# =============================================================================
# Configuration
# =============================================================================

DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'joeldiaz',
    'user': 'joeldiaz',
}

QDRANT_URL = "http://localhost:6333"
QDRANT_CONTAINER_NAME = "qdrant"

EMBED_ANYTHING_MODEL = "Qdrant/all-MiniLM-L6-v2-onnx"

MiB = 1024 * 1024  # Bytes → Mebibytes conversion factor

# Global model cache (shared across all benchmarks)
model_cache = {}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ResourceStats:
    """Resource statistics for Python, PostgreSQL, and (optional) container processes."""
    py_delta_mb: float
    py_peak_mb: float
    py_cpu: float
    pg_delta_mb: float
    pg_peak_mb: float
    pg_cpu: float
    qd_delta_mb: float = 0.0
    qd_peak_mb: float = 0.0
    qd_cpu: float = 0.0
    sys_mem_mb: float = 0.0
    sys_cpu: float = 0.0


@dataclass
class BenchmarkResult:
    time_s: float
    stats: ResourceStats


# =============================================================================
# Statistics
# =============================================================================

def safe_stdev(values: List[float]) -> float:
    """Standard deviation, returning 0 for lists with fewer than 2 elements."""
    return stdev(values) if len(values) > 1 else 0.0


def calc_iqr(values: List[float]) -> float:
    """Interquartile range (Q3 - Q1), returning 0 for lists with fewer than 4 elements."""
    if len(values) < 4:
        return 0.0
    q = quantiles(values, n=4, method='inclusive')
    return q[2] - q[0]


def calc_q1(values: List[float]) -> float:
    """First quartile (Q1), returning median for lists with fewer than 4 elements."""
    if len(values) < 4:
        return median(values) if values else 0.0
    q = quantiles(values, n=4, method='inclusive')
    return q[0]


def calc_q3(values: List[float]) -> float:
    """Third quartile (Q3), returning median for lists with fewer than 4 elements."""
    if len(values) < 4:
        return median(values) if values else 0.0
    q = quantiles(values, n=4, method='inclusive')
    return q[2]


# =============================================================================
# Resource Monitoring
# =============================================================================

class ResourceMonitor:
    """Unified resource monitor for Python, PostgreSQL, and Docker container processes.

    Key design decisions for consistency:
    - Timing is always done externally (around the measured function call)
    - Memory is reported in MiB (1024*1024 bytes)
    - Docker container monitoring is optional (only when container_name is provided)
    """

    def __init__(self, py_pid: int, pg_pid: int = None, container_name: str = None):
        self.py_process = psutil.Process(py_pid)
        self.pg_process = psutil.Process(pg_pid) if pg_pid else None

        # Docker container (only initialized when explicitly requested)
        self.container = None
        self.docker_client = None
        if container_name:
            try:
                import docker
                self.docker_client = docker.from_env()
                self.container = self.docker_client.containers.get(container_name)
            except Exception:
                pass

        # Capture baselines
        self.py_baseline = self._get_process_mem(self.py_process)
        self.pg_baseline = self._get_process_mem(self.pg_process)
        self.container_baseline_stats = self._get_container_stats() if self.container else None

        # Warm up CPU counters (first call always returns 0)
        self.py_process.cpu_percent()
        if self.pg_process:
            self.pg_process.cpu_percent()
        time.sleep(0.1)

    @staticmethod
    def _get_process_mem(process) -> int:
        """Get process memory in bytes (USS preferred, RSS fallback)."""
        if process is None:
            return 0
        try:
            mem = process.memory_full_info()
            return mem.uss if hasattr(mem, 'uss') else mem.rss
        except (psutil.AccessDenied, AttributeError):
            try:
                return process.memory_info().rss
            except Exception:
                return 0
        except Exception:
            return 0

    def _get_container_stats(self):
        """Get Docker container stats snapshot."""
        if self.container is None:
            return None
        try:
            return self.container.stats(stream=False)
        except Exception:
            return None

    def _calc_container_cpu(self, start_stats, end_stats) -> float:
        """Calculate container CPU % between two stats snapshots."""
        try:
            cpu_delta = (end_stats['cpu_stats']['cpu_usage']['total_usage'] -
                         start_stats['cpu_stats']['cpu_usage']['total_usage'])
            system_delta = (end_stats['cpu_stats']['system_cpu_usage'] -
                            start_stats['cpu_stats']['system_cpu_usage'])
            if system_delta > 0.0 and cpu_delta > 0.0:
                cpus = end_stats['cpu_stats'].get('online_cpus', psutil.cpu_count())
                return (cpu_delta / system_delta) * cpus * 100.0
        except KeyError:
            pass
        return 0.0

    @staticmethod
    def measure(py_pid: int, pg_pid: int, func: Callable,
                container_name: str = None):
        """
        Measure time and resource usage around a function call.

        Timing is done externally (around the function call) for consistency
        across all benchmarks.

        Args:
            py_pid: Python process PID
            pg_pid: PostgreSQL backend PID (None if not applicable)
            func: Function to measure (called with no arguments)
            container_name: Docker container name for monitoring (None to skip)

        Returns:
            Tuple of (elapsed_seconds, func_return_value, ResourceStats)
        """
        monitor = ResourceMonitor(py_pid, pg_pid, container_name)

        start_time = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - start_time

        # Python process
        py_peak = ResourceMonitor._get_process_mem(monitor.py_process)
        py_cpu = monitor.py_process.cpu_percent()

        # PostgreSQL process
        pg_peak = ResourceMonitor._get_process_mem(monitor.pg_process)
        pg_cpu = monitor.pg_process.cpu_percent() if monitor.pg_process else 0.0

        # Container (Qdrant, etc.)
        qd_delta_mb = 0.0
        qd_peak_mb = 0.0
        qd_cpu = 0.0
        if monitor.container and monitor.container_baseline_stats:
            end_stats = monitor._get_container_stats()
            if end_stats:
                mem_usage = end_stats['memory_stats'].get('usage', 0)
                qd_peak_mb = mem_usage / MiB
                baseline_usage = monitor.container_baseline_stats['memory_stats'].get('usage', 0)
                qd_delta_mb = qd_peak_mb - (baseline_usage / MiB)
                qd_cpu = monitor._calc_container_cpu(
                    monitor.container_baseline_stats, end_stats)

        # System-wide
        sys_v = psutil.virtual_memory()

        stats = ResourceStats(
            py_delta_mb=(py_peak - monitor.py_baseline) / MiB,
            py_peak_mb=py_peak / MiB,
            py_cpu=py_cpu,
            pg_delta_mb=(pg_peak - monitor.pg_baseline) / MiB if monitor.pg_process else 0.0,
            pg_peak_mb=pg_peak / MiB if monitor.pg_process else 0.0,
            pg_cpu=pg_cpu,
            qd_delta_mb=qd_delta_mb,
            qd_peak_mb=qd_peak_mb,
            qd_cpu=qd_cpu,
            sys_mem_mb=sys_v.used / MiB,
            sys_cpu=psutil.cpu_percent()
        )

        return elapsed, result, stats


# =============================================================================
# Embedding Clients
# =============================================================================

class EmbedAnythingDirectClient:
    """Direct Python client for EmbedAnything text embeddings."""

    def embed(self, texts: List[str],
              model_name: str = EMBED_ANYTHING_MODEL) -> List[List[float]]:
        """Generate embeddings."""
        model = self._get_model(model_name)
        data = embed_anything.embed_query(texts, embedder=model)
        return [item.embedding for item in data]

    @staticmethod
    def _get_model(model_name: str):
        """Get or load model from cache."""
        if model_name not in model_cache:
            model_cache[model_name] = EmbeddingModel.from_pretrained_onnx(
                WhichModel.Bert, hf_model_id=model_name
            )
        return model_cache[model_name]


class EmbeddingWrapper:
    """Wrapper matching ChromaDB's EmbeddingFunction interface."""

    def __init__(self, fn: Callable):
        self._fn = fn

    def __call__(self, input):
        return self._fn(list(input))


# =============================================================================
# PostgreSQL Connection Helpers
# =============================================================================

def connect_pg() -> psycopg2.extensions.connection:
    """Connect to PostgreSQL with autocommit disabled."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    return conn


def get_pg_pid(conn) -> Optional[int]:
    """Get the PostgreSQL backend PID for a connection."""
    try:
        cur = conn.cursor()
        cur.execute("SELECT pg_backend_pid();")
        pid = cur.fetchone()[0]
        cur.close()
        return pid
    except Exception:
        return None


def connect_and_get_pid() -> Tuple[psycopg2.extensions.connection, int]:
    """Connect to PostgreSQL and return (connection, backend_pid)."""
    conn = connect_pg()
    pid = get_pg_pid(conn)
    return conn, pid


def warmup_pg_connection(conn, provider: str = 'embed_anything',
                         model_name: str = EMBED_ANYTHING_MODEL):
    """
    Warm up a PostgreSQL connection by running a small embedding query.

    This ensures:
    - The connection is fully established and ready
    - pg_gembed extension is loaded and model is cached in shared memory
    - Any JIT compilation in PostgreSQL is completed
    - The backend process has allocated initial working memory

    Should be called on every fresh connection before benchmark iterations.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT embed_texts(%s, %s, ARRAY['warmup text for benchmark']::text[])",
        (provider, model_name)
    )
    cur.fetchall()
    conn.commit()
    cur.close()


# =============================================================================
# ChromaDB Helpers
# =============================================================================

def cleanup_chroma(client, db_path: str, delay: float = 0.5):
    """Clean up ChromaDB client and remove data files."""
    del client
    time.sleep(delay)
    if os.path.exists(db_path):
        shutil.rmtree(db_path)


# =============================================================================
# Compute Metrics
# =============================================================================

def compute_metrics(size: int, results: List[BenchmarkResult]) -> dict:
    """
    Compute comprehensive mean/std and median/IQR statistics from benchmark results.

    Returns a dict with all metric fields, ensuring consistent CSV and plot output
    across all benchmarks.
    """
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
        'throughput_q1': size / calc_q3(times) if calc_q3(times) > 0 else 0,  # Q1 throughput uses Q3 time
        'throughput_q3': size / calc_q1(times) if calc_q1(times) > 0 else 0,  # Q3 throughput uses Q1 time
        # Time
        'time_s': mean(times),
        'time_s_std': safe_stdev(times),
        'time_s_median': median(times),
        'time_s_iqr': calc_iqr(times),
        'time_s_q1': calc_q1(times),
        'time_s_q3': calc_q3(times),
        # Python
        'py_cpu': mean(py_cpu),
        'py_cpu_std': safe_stdev(py_cpu),
        'py_cpu_median': median(py_cpu),
        'py_cpu_iqr': calc_iqr(py_cpu),
        'py_cpu_q1': calc_q1(py_cpu),
        'py_cpu_q3': calc_q3(py_cpu),
        'py_mem_delta': mean(py_delta),
        'py_mem_delta_std': safe_stdev(py_delta),
        'py_mem_delta_median': median(py_delta),
        'py_mem_delta_iqr': calc_iqr(py_delta),
        'py_mem_delta_q1': calc_q1(py_delta),
        'py_mem_delta_q3': calc_q3(py_delta),
        'py_mem_peak': mean(py_peak),
        'py_mem_peak_std': safe_stdev(py_peak),
        'py_mem_peak_median': median(py_peak),
        'py_mem_peak_iqr': calc_iqr(py_peak),
        'py_mem_peak_q1': calc_q1(py_peak),
        'py_mem_peak_q3': calc_q3(py_peak),
        # PostgreSQL
        'pg_cpu': mean(pg_cpu),
        'pg_cpu_std': safe_stdev(pg_cpu),
        'pg_cpu_median': median(pg_cpu),
        'pg_cpu_iqr': calc_iqr(pg_cpu),
        'pg_cpu_q1': calc_q1(pg_cpu),
        'pg_cpu_q3': calc_q3(pg_cpu),
        'pg_mem_delta': mean(pg_delta),
        'pg_mem_delta_std': safe_stdev(pg_delta),
        'pg_mem_delta_median': median(pg_delta),
        'pg_mem_delta_iqr': calc_iqr(pg_delta),
        'pg_mem_delta_q1': calc_q1(pg_delta),
        'pg_mem_delta_q3': calc_q3(pg_delta),
        'pg_mem_peak': mean(pg_peak),
        'pg_mem_peak_std': safe_stdev(pg_peak),
        'pg_mem_peak_median': median(pg_peak),
        'pg_mem_peak_iqr': calc_iqr(pg_peak),
        'pg_mem_peak_q1': calc_q1(pg_peak),
        'pg_mem_peak_q3': calc_q3(pg_peak),
        # Container (Qdrant/etc.)
        'qd_cpu': mean(qd_cpu),
        'qd_cpu_std': safe_stdev(qd_cpu),
        'qd_cpu_median': median(qd_cpu),
        'qd_cpu_iqr': calc_iqr(qd_cpu),
        'qd_cpu_q1': calc_q1(qd_cpu),
        'qd_cpu_q3': calc_q3(qd_cpu),
        'qd_mem_delta': mean(qd_delta),
        'qd_mem_delta_std': safe_stdev(qd_delta),
        'qd_mem_delta_median': median(qd_delta),
        'qd_mem_delta_iqr': calc_iqr(qd_delta),
        'qd_mem_delta_q1': calc_q1(qd_delta),
        'qd_mem_delta_q3': calc_q3(qd_delta),
        'qd_mem_peak': mean(qd_peak),
        'qd_mem_peak_std': safe_stdev(qd_peak),
        'qd_mem_peak_median': median(qd_peak),
        'qd_mem_peak_iqr': calc_iqr(qd_peak),
        'qd_mem_peak_q1': calc_q1(qd_peak),
        'qd_mem_peak_q3': calc_q3(qd_peak),
        # System
        'sys_cpu': mean(sys_cpu),
        'sys_cpu_std': safe_stdev(sys_cpu),
        'sys_cpu_median': median(sys_cpu),
        'sys_cpu_iqr': calc_iqr(sys_cpu),
        'sys_cpu_q1': calc_q1(sys_cpu),
        'sys_cpu_q3': calc_q3(sys_cpu),
        'sys_mem': mean(sys_mem),
        'sys_mem_std': safe_stdev(sys_mem),
        'sys_mem_median': median(sys_mem),
        'sys_mem_iqr': calc_iqr(sys_mem),
        'sys_mem_q1': calc_q1(sys_mem),
        'sys_mem_q3': calc_q3(sys_mem),
    }
