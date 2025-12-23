mod tei_grpc_embedder;
mod tei_http_embedder;

use anyhow::Result;
use postgres::{Client, NoTls};
use std::time::Instant;
use sysinfo::{Pid, ProcessesToUpdate, System};
use tei_grpc_embedder::TeiGrpcEmbedder;
use tei_http_embedder::TeiHttpEmbedder;

pub mod tei {
    pub mod v1 {
        tonic::include_proto!("tei.v1");
    }
}

const TEST_SIZES: &[usize] = &[16, 32, 64, 128, 256, 512, 1024, 2048];
const BATCH_SIZE: usize = 32;
const DB_CONNECTION: &str = "host=localhost port=5432 dbname=joeldiaz";

#[derive(Debug, Clone, Copy, Default)]
struct ResourceStats {
    delta_mb: f64,
    peak_mb: f64,
    cpu_usage: f32,
    sys_peak_mb: f64,
    sys_cpu_usage: f32,
}

struct ResourceMonitor {
    sys: System,
    pid: Pid,
    baseline: u64,
}

impl ResourceMonitor {
    fn new(pid: i32) -> Self {
        let mut sys = System::new();
        let pid = Pid::from_u32(pid as u32);

        // Refresh specific process and global system stats
        sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        sys.refresh_memory();
        sys.refresh_cpu_all(); // First refresh for CPU diff

        let baseline = sys.process(pid).map_or(0, |p| p.memory());

        Self { sys, pid, baseline }
    }

    fn measure<F, T>(pid: i32, f: F) -> Result<(T, ResourceStats)>
    where
        F: FnOnce() -> Result<T>,
    {
        let mut monitor = Self::new(pid);
        let result = f()?;

        monitor
            .sys
            .refresh_processes(ProcessesToUpdate::Some(&[monitor.pid]), true);
        monitor.sys.refresh_memory();
        monitor.sys.refresh_cpu_all();

        let process = monitor.sys.process(monitor.pid);
        let peak = process.map_or(monitor.baseline, |p| p.memory());
        let cpu_usage = process.map_or(0.0, |p| p.cpu_usage());

        let baseline_mb = monitor.baseline as f64 / 1024.0 / 1024.0;
        let peak_mb = peak as f64 / 1024.0 / 1024.0;
        let delta_mb = peak_mb - baseline_mb;

        let sys_peak_mb = monitor.sys.used_memory() as f64 / 1024.0 / 1024.0;
        let sys_cpu_usage = monitor.sys.global_cpu_usage();

        Ok((
            result,
            ResourceStats {
                delta_mb,
                peak_mb,
                cpu_usage,
                sys_peak_mb,
                sys_cpu_usage,
            },
        ))
    }
}

fn connect_and_get_pid() -> Result<(Client, i32)> {
    let mut client = Client::connect(DB_CONNECTION, NoTls)?;
    let pid: i32 = client.query_one("SELECT pg_backend_pid()", &[])?.get(0);
    Ok((client, pid))
}

fn setup_database(client: &mut Client) -> Result<()> {
    client.batch_execute(
        "
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE EXTENSION IF NOT EXISTS pg_gembed;
        DROP TABLE IF EXISTS embeddings_test;
        CREATE TABLE embeddings_test (
            id SERIAL PRIMARY KEY,
            text TEXT,
            embedding vector(384)
        );
        ",
    )?;
    Ok(())
}

fn truncate_table(client: &mut Client) -> Result<()> {
    client.execute("TRUNCATE embeddings_test", &[])?;
    Ok(())
}

fn make_inputs(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            format!(
                "This is test sentence number {} for embedding generation.",
                i
            )
        })
        .collect()
}

fn benchmark_internal_db_gen(
    client: &mut Client,
    texts: &[String],
    provider: &str,
    model: &str,
) -> Result<f64> {
    let start = Instant::now();

    for chunk in texts.chunks(BATCH_SIZE) {
        let text_literals: Vec<String> = chunk
            .iter()
            .map(|s| format!("'{}'", s.replace("'", "''")))
            .collect();
        let array_str = text_literals.join(", ");

        let sql = format!(
            "INSERT INTO embeddings_test (text, embedding)
             SELECT t, e FROM unnest(ARRAY[{}]) t,
             unnest(embed_texts('{}', '{}', ARRAY[{}])) e",
            array_str, provider, model, array_str
        );

        client.execute(&sql, &[])?;
    }

    Ok(start.elapsed().as_secs_f64())
}

fn benchmark_external_client_gen<F>(
    client: &mut Client,
    texts: &[String],
    mut embed_fn: F,
) -> Result<f64>
where
    F: FnMut(Vec<String>) -> Result<Vec<Vec<f32>>>,
{
    let start = Instant::now();

    for chunk in texts.chunks(BATCH_SIZE) {
        let embeddings = embed_fn(chunk.to_vec())?;

        for (text, embedding) in chunk.iter().zip(embeddings.iter()) {
            let vector_str = format!(
                "[{}]",
                embedding
                    .iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
            let text_escaped = text.replace("'", "''");

            let sql = format!(
                "INSERT INTO embeddings_test (text, embedding) VALUES ('{}', '{}'::vector)",
                text_escaped, vector_str
            );

            client.execute(&sql, &[])?;
        }
    }

    Ok(start.elapsed().as_secs_f64())
}

fn run_benchmark_iteration<F>(benchmark_fn: F) -> Result<(f64, ResourceStats)>
where
    F: FnOnce(&mut Client) -> Result<f64>,
{
    let (mut client, pid) = connect_and_get_pid()?;
    truncate_table(&mut client)?;
    let result = ResourceMonitor::measure(pid, || benchmark_fn(&mut client))?;
    Ok(result)
}

fn main() -> Result<()> {
    // Initial setup
    {
        let mut client = Client::connect(DB_CONNECTION, NoTls)?;
        setup_database(&mut client)?;
    }

    let mut grpc_service = TeiGrpcEmbedder::new()?;
    let http_service = TeiHttpEmbedder::new()?;

    // Warm-up
    {
        let warmup_texts = make_inputs(8);
        let mut client = Client::connect(DB_CONNECTION, NoTls)?;

        truncate_table(&mut client)?;
        benchmark_internal_db_gen(
            &mut client,
            &warmup_texts,
            "fastembed",
            "Qdrant/all-MiniLM-L6-v2-onnx",
        )?;

        truncate_table(&mut client)?;
        benchmark_internal_db_gen(
            &mut client,
            &warmup_texts,
            "grpc",
            "sentence-transformers/all-MiniLM-L6-v2",
        )?;

        truncate_table(&mut client)?;
        benchmark_external_client_gen(&mut client, &warmup_texts, |t| grpc_service.embed(t))?;

        truncate_table(&mut client)?;
        benchmark_external_client_gen(&mut client, &warmup_texts, |t| http_service.embed(t))?;

        println!("Warm-up completed.\n");
    }

    println!("Benchmark Results:");
    println!(
        "{:<14} | {:>9} | {:>10} | {:>10} | {:>9} | {:>12} | {:>9}",
        "", "Time (s)", "Δ Mem (MB)", "Peak (MB)", "CPU (%)", "Sys Mem (MB)", "Sys CPU (%)"
    );
    println!("{}", "=".repeat(95));

    let mut results = Vec::new();

    for &size in TEST_SIZES {
        let texts = make_inputs(size);

        // Benchmark PG FastEmbed
        let (pg_fast_time, pg_fast_mem) = run_benchmark_iteration(|c| {
            benchmark_internal_db_gen(c, &texts, "fastembed", "Qdrant/all-MiniLM-L6-v2-onnx")
        })?;

        // Benchmark PG gRPC
        let (pg_grpc_time, pg_grpc_mem) = run_benchmark_iteration(|c| {
            benchmark_internal_db_gen(c, &texts, "grpc", "sentence-transformers/all-MiniLM-L6-v2")
        })?;

        // Benchmark External gRPC
        let (ext_grpc_time, ext_grpc_mem) = run_benchmark_iteration(|c| {
            benchmark_external_client_gen(c, &texts, |t| grpc_service.embed(t))
        })?;

        // Benchmark External HTTP
        let (ext_http_time, ext_http_mem) = run_benchmark_iteration(|c| {
            benchmark_external_client_gen(c, &texts, |t| http_service.embed(t))
        })?;

        println!("Size: {}", size);
        println!(
            "  {:<12} | {:>9.3} | {:>10.1} | {:>10.1} | {:>9.1} | {:>12.0} | {:>11.1}",
            "PG FastEmbed",
            pg_fast_time,
            pg_fast_mem.delta_mb,
            pg_fast_mem.peak_mb,
            pg_fast_mem.cpu_usage,
            pg_fast_mem.sys_peak_mb,
            pg_fast_mem.sys_cpu_usage
        );
        println!(
            "  {:<12} | {:>9.3} | {:>10.1} | {:>10.1} | {:>9.1} | {:>12.0} | {:>11.1}",
            "PG gRPC",
            pg_grpc_time,
            pg_grpc_mem.delta_mb,
            pg_grpc_mem.peak_mb,
            pg_grpc_mem.cpu_usage,
            pg_grpc_mem.sys_peak_mb,
            pg_grpc_mem.sys_cpu_usage
        );
        println!(
            "  {:<12} | {:>9.3} | {:>10.1} | {:>10.1} | {:>9.1} | {:>12.0} | {:>11.1}",
            "Ext gRPC",
            ext_grpc_time,
            ext_grpc_mem.delta_mb,
            ext_grpc_mem.peak_mb,
            ext_grpc_mem.cpu_usage,
            ext_grpc_mem.sys_peak_mb,
            ext_grpc_mem.sys_cpu_usage
        );
        println!(
            "  {:<12} | {:>9.3} | {:>10.1} | {:>10.1} | {:>9.1} | {:>12.0} | {:>11.1}",
            "Ext HTTP",
            ext_http_time,
            ext_http_mem.delta_mb,
            ext_http_mem.peak_mb,
            ext_http_mem.cpu_usage,
            ext_http_mem.sys_peak_mb,
            ext_http_mem.sys_cpu_usage
        );
        println!();

        let tps = |time| size as f64 / time;
        results.push((
            tps(pg_fast_time),
            tps(pg_grpc_time),
            tps(ext_grpc_time),
            tps(ext_http_time),
        ));
    }

    println!("{}", "=".repeat(95));
    let n = results.len() as f64;
    let sums = results.iter().fold((0.0, 0.0, 0.0, 0.0), |acc, r| {
        (acc.0 + r.0, acc.1 + r.1, acc.2 + r.2, acc.3 + r.3)
    });

    println!("\nAverage Throughput (texts/sec):");
    println!("  PG FastEmbed:      {:.2}", sums.0 / n);
    println!("  PG gRPC:           {:.2}", sums.1 / n);
    println!("  External gRPC:     {:.2}", sums.2 / n);
    println!("  External HTTP:     {:.2}", sums.3 / n);

    Ok(())
}
