mod tei_grpc_batch;
mod tei_grpc_individual;
mod tei_http;

use anyhow::Result;
use std::time::Instant;
use tei_grpc_batch::TeiGrpcBatch;
use tei_grpc_individual::TeiGrpcIndividual;
use tei_http::TeiHttp;

pub mod tei {
    pub mod v1 {
        tonic::include_proto!("tei.v1");
    }
}

const TOTAL_SIZES: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128];
const CHUNK_SIZE: usize = 32;

/// Generate dummy text inputs for testing.
fn make_inputs(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| format!("Example sentence number {} for embedding test.", i))
        .collect()
}

fn main() -> Result<()> {
    // Initialize services
    let mut grpc_batch_service = TeiGrpcBatch::new()?;
    let mut grpc_individual_service = TeiGrpcIndividual::new()?;
    let http_service = TeiHttp::new()?;

    benchmark(
        &mut grpc_batch_service,
        &mut grpc_individual_service,
        http_service,
    )?;

    Ok(())
}

fn benchmark(
    grpc_batch_service: &mut TeiGrpcBatch,
    grpc_individual_service: &mut TeiGrpcIndividual,
    http_service: TeiHttp,
) -> Result<()> {
    println!(
        "{:<8} {:>18} {:>18} {:>18} {:>18} {:>18} {:>18} {:>18} {:>18}",
        "Total",
        "gRPC Batch (ms)",
        "gRPC Indiv (ms)",
        "HTTP (ms)",
        "Batch (sent/s)",
        "Indiv (sent/s)",
        "HTTP (sent/s)",
        "Batch/Indiv×",
        "Batch/HTTP×"
    );

    for &n in TOTAL_SIZES {
        let inputs = make_inputs(n);

        // gRPC Batch
        let t0 = Instant::now();
        for chunk in inputs.chunks(CHUNK_SIZE) {
            let _ = grpc_batch_service.embed(chunk.to_vec())?;
        }
        let batch_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let batch_sps = n as f64 / (batch_ms / 1000.0);

        // gRPC Individual
        let t1 = Instant::now();
        for chunk in inputs.chunks(CHUNK_SIZE) {
            let _ = grpc_individual_service.embed(chunk.to_vec())?;
        }
        let individual_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let individual_sps = n as f64 / (individual_ms / 1000.0);

        // HTTP
        let t2 = Instant::now();
        for chunk in inputs.chunks(CHUNK_SIZE) {
            let _ = http_service.embed(chunk.to_vec())?;
        }
        let http_ms = t2.elapsed().as_secs_f64() * 1000.0;
        let http_sps = n as f64 / (http_ms / 1000.0);

        // Speedup ratios
        let batch_speedup = batch_sps / individual_sps;
        let batch_http_speedup = batch_sps / http_sps;

        println!(
            "{:<8} {:>18.2} {:>18.2} {:>18.2} {:>18.1} {:>18.1} {:>18.1} {:>18.2} {:>18.2}",
            n,
            batch_ms,
            individual_ms,
            http_ms,
            batch_sps,
            individual_sps,
            http_sps,
            batch_speedup,
            batch_http_speedup
        );
    }

    Ok(())
}
