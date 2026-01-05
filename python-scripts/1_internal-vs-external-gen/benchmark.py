import csv
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Callable, List

import embed_anything
import grpc
import matplotlib.pyplot as plt
import psutil
import psycopg2
import requests
import tei_pb2 as pb2
import tei_pb2_grpc as pb2_grpc
from embed_anything import EmbeddingModel, WhichModel
from psycopg2.extras import execute_values

# Configuration
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'joeldiaz',
    'user': 'joeldiaz',
}

TEST_SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048]
BATCH_SIZE = 32
EMBED_ANYTHING_MODEL = "Qdrant/all-MiniLM-L6-v2-onnx"
RUNS_PER_SIZE = 3  # Number of runs per test size

# Output directory
OUTPUT_DIR = Path(__file__).parent / "output"

# Global model cache for direct Python calls
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

        # Get baseline memory - try USS first, fall back to RSS on macOS
        try:
            mem_info = self.process.memory_full_info()
            self.baseline = mem_info.uss if hasattr(mem_info, 'uss') else mem_info.rss
        except (psutil.AccessDenied, AttributeError):
            # Fall back to RSS on macOS or when USS not available
            mem_info = self.process.memory_info()
            self.baseline = mem_info.rss

        # Initial CPU measurement (need two samples for CPU usage)
        self.process.cpu_percent()
        time.sleep(0.1)

    @staticmethod
    def measure(pid: int, func: Callable):
        """Measure resource usage during function execution."""
        monitor = ResourceMonitor(pid)

        # Execute the function
        result = func()

        # Get final measurements - try USS first, fall back to RSS
        try:
            mem_info = monitor.process.memory_full_info()
            peak = mem_info.uss if hasattr(mem_info, 'uss') else mem_info.rss
        except (psutil.AccessDenied, AttributeError):
            mem_info = monitor.process.memory_info()
            peak = mem_info.rss

        cpu_usage = monitor.process.cpu_percent()

        # Convert to MB
        baseline_mb = monitor.baseline / (1024 * 1024)
        peak_mb = peak / (1024 * 1024)
        delta_mb = peak_mb - baseline_mb

        # System-wide stats
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


class EmbedAnythingGrpcClient:
    """gRPC client for EmbedAnything server."""

    def __init__(self, address: str = "localhost:50051"):
        self.channel = grpc.insecure_channel(address)
        self.stub = pb2_grpc.EmbedStub(self.channel)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via gRPC."""
        request = pb2.EmbedBatchRequest(
            inputs=texts,
            truncate=True,
            normalize=True,
            truncation_direction=0,
            model=EMBED_ANYTHING_MODEL
        )

        response = self.stub.EmbedBatch(request)
        return [list(emb.values) for emb in response.embeddings]

    def close(self):
        self.channel.close()


class EmbedAnythingHttpClient:
    """HTTP client for EmbedAnything server."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via HTTP."""
        response = self.session.post(
            f"{self.base_url}/v1/embed",
            json={
                "model": EMBED_ANYTHING_MODEL,
                "input": texts
            }
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]


class EmbedAnythingDirectClient:
    """Direct Python client for EmbedAnything."""

    def __init__(self):
        pass

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


def connect_and_get_pid():
    """Connect to PostgreSQL and get backend PID."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("SELECT pg_backend_pid();")
    pid = cur.fetchone()[0]
    cur.close()
    return conn, pid


def setup_database(conn):
    """Initialize database schema."""
    cur = conn.cursor()
    cur.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE EXTENSION IF NOT EXISTS pg_gembed;
                DROP TABLE IF EXISTS embeddings_test;
                CREATE TABLE embeddings_test
                (
                    id        SERIAL PRIMARY KEY,
                    text      TEXT,
                    embedding vector(384)
                );
                """)
    conn.commit()
    cur.close()


def truncate_table(conn):
    """Clear test table."""
    cur = conn.cursor()
    cur.execute("TRUNCATE embeddings_test;")
    conn.commit()
    cur.close()


def make_inputs(n: int) -> List[str]:
    """Generate test sentences."""
    return [
        f"This is test sentence number {i} for embedding generation."
        for i in range(n)
    ]


def benchmark_internal_db_gen(conn, texts: List[str], provider: str, model: str) -> float:
    """Benchmark pg_gembed internal generation."""
    cur = conn.cursor()
    start = time.perf_counter()

    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i:i + BATCH_SIZE]

        sql = """
              INSERT INTO embeddings_test (text, embedding)
              SELECT t, e
              FROM unnest(%s::text[]) t,
                   unnest(embed_texts(%s, %s, %s::text[])) e \
              """
        cur.execute(sql, (chunk, provider, model, chunk))

    conn.commit()
    elapsed = time.perf_counter() - start
    cur.close()

    return elapsed


def benchmark_external_client_gen(conn, texts: List[str], embed_fn: Callable) -> float:
    """Benchmark external embedding client."""
    cur = conn.cursor()
    start = time.perf_counter()

    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i:i + BATCH_SIZE]
        embeddings = embed_fn(chunk)

        values = [(text, embedding) for text, embedding in zip(chunk, embeddings)]
        execute_values(
            cur,
            "INSERT INTO embeddings_test (text, embedding) VALUES %s",
            values,
            template="(%s, %s::vector)"
        )

    conn.commit()
    elapsed = time.perf_counter() - start
    cur.close()

    return elapsed


def run_benchmark_iteration(conn, pid: int, benchmark_fn: Callable) -> BenchmarkResult:
    """Run a single benchmark iteration with resource monitoring."""
    truncate_table(conn)

    def execute_benchmark():
        return benchmark_fn(conn)

    elapsed, stats = ResourceMonitor.measure(pid, execute_benchmark)

    return BenchmarkResult(time_s=elapsed, stats=stats)


def run_multiple_iterations(conn, pid: int, benchmark_fn: Callable,
                            runs: int) -> List[BenchmarkResult]:
    """Run multiple benchmark iterations."""
    results = []
    for _ in range(runs):
        result = run_benchmark_iteration(conn, pid, benchmark_fn)
        results.append(result)
    return results


def run_method_with_fresh_connection(texts: List[str], benchmark_fn: Callable,
                                     runs: int) -> List[BenchmarkResult]:
    """Create a fresh connection, warm it up, then run all benchmark iterations."""
    # Create fresh connection for this method
    conn, pid = connect_and_get_pid()

    try:
        # Warm up this specific method on the fresh connection
        warmup_texts = make_inputs(8)
        truncate_table(conn)
        benchmark_fn(conn, warmup_texts)

        # Now run the actual benchmark iterations
        results = run_multiple_iterations(
            conn, pid,
            lambda c: benchmark_fn(c, texts),
            runs
        )
        return results
    finally:
        conn.close()


def print_header():
    """Print benchmark results header."""
    print("Benchmark Results:")
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

    avg_time = mean(times)
    avg_delta_mem = mean(delta_mems)
    avg_peak_mem = mean(peak_mems)
    avg_cpu = mean(cpus)
    avg_sys_peak_mem = mean(sys_peak_mems)
    avg_sys_cpu = mean(sys_cpus)

    print(f"  {label:12} | {avg_time:>9.3f} | {avg_delta_mem:>10.1f} | {avg_peak_mem:>10.1f} | "
          f"{avg_cpu:>9.1f} | {avg_sys_peak_mem:>12.0f} | {avg_sys_cpu:>11.1f}")


def main():
    # Initialize setup connection (just for schema setup)
    conn, _ = connect_and_get_pid()
    setup_database(conn)
    conn.close()

    # Initialize external clients (these are reused across connections)
    http_client = EmbedAnythingHttpClient()
    grpc_client = EmbedAnythingGrpcClient()
    direct_client = EmbedAnythingDirectClient()

    try:
        print_header()

        all_results = []

        for size in TEST_SIZES:
            texts = make_inputs(size)

            print(f"Size: {size}")

            # Benchmark PG EmbedAnything (fresh connection per method)
            pg_local_results = run_method_with_fresh_connection(
                texts,
                lambda c, t: benchmark_internal_db_gen(
                    c, t, "embed_anything", EMBED_ANYTHING_MODEL
                ),
                RUNS_PER_SIZE
            )

            # Benchmark PG gRPC (fresh connection per method)
            pg_grpc_results = run_method_with_fresh_connection(
                texts,
                lambda c, t: benchmark_internal_db_gen(
                    c, t, "grpc", EMBED_ANYTHING_MODEL
                ),
                RUNS_PER_SIZE
            )

            # Benchmark Direct Python (fresh connection per method)
            ext_direct_results = run_method_with_fresh_connection(
                texts,
                lambda c, t: benchmark_external_client_gen(
                    c, t, direct_client.embed
                ),
                RUNS_PER_SIZE
            )

            # Benchmark External gRPC (fresh connection per method)
            ext_grpc_results = run_method_with_fresh_connection(
                texts,
                lambda c, t: benchmark_external_client_gen(
                    c, t, grpc_client.embed
                ),
                RUNS_PER_SIZE
            )

            # Benchmark External HTTP (fresh connection per method)
            ext_http_results = run_method_with_fresh_connection(
                texts,
                lambda c, t: benchmark_external_client_gen(
                    c, t, http_client.embed
                ),
                RUNS_PER_SIZE
            )

            print_result("PG local", pg_local_results)
            print_result("PG gRPC", pg_grpc_results)
            print_result("Ext Direct", ext_direct_results)
            print_result("Ext gRPC", ext_grpc_results)
            print_result("Ext HTTP", ext_http_results)
            print()

            # Store metrics for summary and plots
            all_results.append({
                'size': size,
                'pg_local': {
                    'throughput': size / mean([r.time_s for r in pg_local_results]),
                    'time_s': mean([r.time_s for r in pg_local_results]),
                    'cpu': mean([r.stats.cpu_usage for r in pg_local_results]),
                    'mem_delta': mean([r.stats.delta_mb for r in pg_local_results]),
                    'mem_peak': mean([r.stats.peak_mb for r in pg_local_results]),
                    'sys_cpu': mean([r.stats.sys_cpu_usage for r in pg_local_results]),
                    'sys_mem': mean([r.stats.sys_peak_mb for r in pg_local_results]),
                },
                'pg_grpc': {
                    'throughput': size / mean([r.time_s for r in pg_grpc_results]),
                    'time_s': mean([r.time_s for r in pg_grpc_results]),
                    'cpu': mean([r.stats.cpu_usage for r in pg_grpc_results]),
                    'mem_delta': mean([r.stats.delta_mb for r in pg_grpc_results]),
                    'mem_peak': mean([r.stats.peak_mb for r in pg_grpc_results]),
                    'sys_cpu': mean([r.stats.sys_cpu_usage for r in pg_grpc_results]),
                    'sys_mem': mean([r.stats.sys_peak_mb for r in pg_grpc_results]),
                },
                'ext_direct': {
                    'throughput': size / mean([r.time_s for r in ext_direct_results]),
                    'time_s': mean([r.time_s for r in ext_direct_results]),
                    'cpu': mean([r.stats.cpu_usage for r in ext_direct_results]),
                    'mem_delta': mean([r.stats.delta_mb for r in ext_direct_results]),
                    'mem_peak': mean([r.stats.peak_mb for r in ext_direct_results]),
                    'sys_cpu': mean([r.stats.sys_cpu_usage for r in ext_direct_results]),
                    'sys_mem': mean([r.stats.sys_peak_mb for r in ext_direct_results]),
                },
                'ext_grpc': {
                    'throughput': size / mean([r.time_s for r in ext_grpc_results]),
                    'time_s': mean([r.time_s for r in ext_grpc_results]),
                    'cpu': mean([r.stats.cpu_usage for r in ext_grpc_results]),
                    'mem_delta': mean([r.stats.delta_mb for r in ext_grpc_results]),
                    'mem_peak': mean([r.stats.peak_mb for r in ext_grpc_results]),
                    'sys_cpu': mean([r.stats.sys_cpu_usage for r in ext_grpc_results]),
                    'sys_mem': mean([r.stats.sys_peak_mb for r in ext_grpc_results]),
                },
                'ext_http': {
                    'throughput': size / mean([r.time_s for r in ext_http_results]),
                    'time_s': mean([r.time_s for r in ext_http_results]),
                    'cpu': mean([r.stats.cpu_usage for r in ext_http_results]),
                    'mem_delta': mean([r.stats.delta_mb for r in ext_http_results]),
                    'mem_peak': mean([r.stats.peak_mb for r in ext_http_results]),
                    'sys_cpu': mean([r.stats.sys_cpu_usage for r in ext_http_results]),
                    'sys_mem': mean([r.stats.sys_peak_mb for r in ext_http_results]),
                },
            })

        # Print summary
        print("=" * 95)

        avg_pg_local = mean([r['pg_local']['throughput'] for r in all_results])
        avg_pg_grpc = mean([r['pg_grpc']['throughput'] for r in all_results])
        avg_ext_direct = mean([r['ext_direct']['throughput'] for r in all_results])
        avg_ext_grpc = mean([r['ext_grpc']['throughput'] for r in all_results])
        avg_ext_http = mean([r['ext_http']['throughput'] for r in all_results])

        print("\nAverage Throughput Across All Sizes (texts/sec):")
        print(f"  PG local:          {avg_pg_local:.2f}")
        print(f"  PG gRPC:           {avg_pg_grpc:.2f}")
        print(f"  External Direct:   {avg_ext_direct:.2f}")
        print(f"  External gRPC:     {avg_ext_grpc:.2f}")
        print(f"  External HTTP:     {avg_ext_http:.2f}")

        # Save results to CSV and generate plots
        save_results_csv(all_results)
        generate_plots(all_results)

    finally:
        grpc_client.close()


def save_results_csv(all_results: List[dict]):
    """Save benchmark results to CSV file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"benchmark_{timestamp}.csv"

    methods = ['pg_local', 'pg_grpc', 'ext_direct', 'ext_grpc', 'ext_http']
    metrics = ['throughput', 'time_s', 'cpu', 'mem_delta', 'mem_peak', 'sys_cpu', 'sys_mem']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: size, then method_metric for each combination
        header = ['size']
        for method in methods:
            for metric in metrics:
                header.append(f"{method}_{metric}")
        writer.writerow(header)

        for r in all_results:
            row = [r['size']]
            for method in methods:
                for metric in metrics:
                    row.append(r[method][metric])
            writer.writerow(row)

    print(f"\nResults saved to: {csv_path}")


def generate_plots(all_results: List[dict]):
    """Generate comparison plots for throughput, CPU, and memory."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    sizes = [r['size'] for r in all_results]
    methods = ['pg_local', 'pg_grpc', 'ext_direct', 'ext_grpc', 'ext_http']
    labels = ['PG Local', 'PG gRPC', 'In-Process', 'Ext gRPC', 'Ext HTTP']
    colors = ['#2ecc71', '#27ae60', '#3498db', '#e74c3c', '#c0392b']
    markers = ['o', 's', '^', 'd', 'x']

    # Plot 1: Throughput comparison
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        plt.plot(sizes, [r[method]['throughput'] for r in all_results],
                 f'{marker}-', label=label, linewidth=2, color=color)
    plt.xlabel('Number of Texts')
    plt.ylabel('Throughput (texts/sec)')
    plt.title(f'Embedding Generation: Throughput (batch size={BATCH_SIZE})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"throughput_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Process CPU Usage
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        plt.plot(sizes, [r[method]['cpu'] for r in all_results],
                 f'{marker}-', label=label, linewidth=2, color=color)
    plt.xlabel('Number of Texts')
    plt.ylabel('Process CPU Usage (%)')
    plt.title(
        f'Process CPU Usage (batch size={BATCH_SIZE})\n(PG methods: PostgreSQL process, External: Python process)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"cpu_process_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: System CPU Usage
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        plt.plot(sizes, [r[method]['sys_cpu'] for r in all_results],
                 f'{marker}-', label=label, linewidth=2, color=color)
    plt.xlabel('Number of Texts')
    plt.ylabel('System CPU Usage (%)')
    plt.title(f'Embedding Generation: System CPU (batch size={BATCH_SIZE})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"cpu_system_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 4: Process Memory Delta
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        plt.plot(sizes, [r[method]['mem_delta'] for r in all_results],
                 f'{marker}-', label=label, linewidth=2, color=color)
    plt.xlabel('Number of Texts')
    plt.ylabel('Process Memory Delta (MB)')
    plt.title(
        f'Process Memory Change (batch size={BATCH_SIZE})\n(PG methods: PostgreSQL process, External: Python process)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"memory_delta_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 5: Process Peak Memory
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        plt.plot(sizes, [r[method]['mem_peak'] for r in all_results],
                 f'{marker}-', label=label, linewidth=2, color=color)
    plt.xlabel('Number of Texts')
    plt.ylabel('Process Peak Memory (MB)')
    plt.title(
        f'Process Peak Memory (batch size={BATCH_SIZE})\n(PG methods: PostgreSQL process, External: Python process)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"memory_peak_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 6: System Memory
    plt.figure(figsize=(10, 6))
    for method, label, color, marker in zip(methods, labels, colors, markers):
        plt.plot(sizes, [r[method]['sys_mem'] for r in all_results],
                 f'{marker}-', label=label, linewidth=2, color=color)
    plt.xlabel('Number of Texts')
    plt.ylabel('System Memory (MB)')
    plt.title(f'Embedding Generation: System Memory (batch size={BATCH_SIZE})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log', base=2)
    plt.savefig(OUTPUT_DIR / f"memory_system_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 7: Summary bar chart (2x3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Average Metrics (batch size={BATCH_SIZE})', fontsize=14)

    # Throughput
    avgs = [mean([r[m]['throughput'] for r in all_results]) for m in methods]
    bars = axes[0, 0].bar(labels, avgs, color=colors)
    axes[0, 0].set_ylabel('Throughput (texts/sec)')
    axes[0, 0].set_title('Throughput')
    axes[0, 0].bar_label(bars, fmt='%.1f')
    axes[0, 0].tick_params(axis='x', rotation=15)

    # Process CPU (PG=PostgreSQL, Ext=Python)
    avgs = [mean([r[m]['cpu'] for r in all_results]) for m in methods]
    bars = axes[0, 1].bar(labels, avgs, color=colors)
    axes[0, 1].set_ylabel('CPU Usage (%)')
    axes[0, 1].set_title('Process CPU*')
    axes[0, 1].bar_label(bars, fmt='%.1f')
    axes[0, 1].tick_params(axis='x', rotation=15)

    # System CPU
    avgs = [mean([r[m]['sys_cpu'] for r in all_results]) for m in methods]
    bars = axes[0, 2].bar(labels, avgs, color=colors)
    axes[0, 2].set_ylabel('CPU Usage (%)')
    axes[0, 2].set_title('System CPU')
    axes[0, 2].bar_label(bars, fmt='%.1f')
    axes[0, 2].tick_params(axis='x', rotation=15)

    # Memory Delta (PG=PostgreSQL, Ext=Python)
    avgs = [mean([r[m]['mem_delta'] for r in all_results]) for m in methods]
    bars = axes[1, 0].bar(labels, avgs, color=colors)
    axes[1, 0].set_ylabel('Memory (MB)')
    axes[1, 0].set_title('Process Mem Δ*')
    axes[1, 0].bar_label(bars, fmt='%.1f')
    axes[1, 0].tick_params(axis='x', rotation=15)

    # Peak Memory (PG=PostgreSQL, Ext=Python)
    avgs = [mean([r[m]['mem_peak'] for r in all_results]) for m in methods]
    bars = axes[1, 1].bar(labels, avgs, color=colors)
    axes[1, 1].set_ylabel('Memory (MB)')
    axes[1, 1].set_title('Process Peak Mem*')
    axes[1, 1].bar_label(bars, fmt='%.1f')
    axes[1, 1].tick_params(axis='x', rotation=15)

    # System Memory
    avgs = [mean([r[m]['sys_mem'] for r in all_results]) for m in methods]
    bars = axes[1, 2].bar(labels, avgs, color=colors)
    axes[1, 2].set_ylabel('Memory (MB)')
    axes[1, 2].set_title('System Memory')
    axes[1, 2].bar_label(bars, fmt='%.0f')
    axes[1, 2].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"summary_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
