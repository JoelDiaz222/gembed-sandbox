mod candle;
mod grpc_embed_client;

use crate::candle::CandleEmbedService;
use crate::grpc_embed_client::GrpcEmbedClient;
use anyhow::{bail, Result};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use std::path::PathBuf;
use std::time::Instant;

pub mod tei {
    pub mod v1 {
        tonic::include_proto!("tei.v1");
    }
}

const TOTAL_SIZES: &[usize] = &[1, 2, 4, 8, 16, 32, 64, 128];
const CHUNK_SIZE: usize = 32;
const EPSILON: f32 = 1e-5;

/// Calculate cosine similarity between two vectors.
fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let dot = v1.iter().zip(v2).map(|(a, b)| a * b).sum::<f32>();
    let norm1 = v1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2 = v2.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm1 * norm2)
}

/// Generate dummy text inputs for testing.
fn make_inputs(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| format!("Example sentence number {} for embedding test.", i))
        .collect()
}

fn main() -> Result<()> {
    // Initialize models
    let mut fastembed_model = initialize_fastembed_model()?;
    let mut grpc_embed_service = GrpcEmbedClient::new()?;
    let candle_embed_service = CandleEmbedService::new()?;

    // Generate embeddings for the same input with all models
    let test_input = vec!["This is a test sentence.".to_string()];
    let fastembed_vec = fastembed_model.embed(test_input.clone(), None)?[0].clone();
    let grpc_vec = grpc_embed_service.embed(test_input.clone())?[0].clone();
    let candle_vec = candle_embed_service.embed(test_input.clone())?[0].clone();

    verify_similarity(&fastembed_vec, &grpc_vec, &candle_vec)?;
    benchmark(
        &mut fastembed_model,
        &mut grpc_embed_service,
        candle_embed_service,
    )?;
    Ok(())
}

fn initialize_fastembed_model() -> Result<TextEmbedding> {
    let fastembed = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::AllMiniLML6V2)
            .with_cache_dir(PathBuf::from("./fastembed_model")),
    )?;
    Ok(fastembed)
}

fn verify_similarity(
    fastembed_vec: &Vec<f32>,
    grpc_vec: &Vec<f32>,
    candle_vec: &Vec<f32>,
) -> Result<()> {
    println!("--- Verification Check ---");
    let mut similarity = cosine_similarity(&fastembed_vec, &grpc_vec);
    println!("Similarity (FastEmbed, gRPC): {:.6}", similarity);
    if (1.0 - similarity).abs() > EPSILON {
        bail!(
            " ❌ Verification error: the FastEmbed-rs and the Sentence Transformers embeddings are semantically different."
        );
    }

    similarity = cosine_similarity(&fastembed_vec, &candle_vec);
    println!("Similarity (FastEmbed, Candle): {:.6}", similarity);
    if (1.0 - similarity).abs() > EPSILON {
        bail!(
            " ❌ Verification error: the FastEmbed-rs and the Candle embeddings are semantically different."
        );
    }

    println!(" ✅ Verification complete.\n");
    Ok(())
}

fn benchmark(
    fastembed_model: &mut TextEmbedding,
    grpc_embed_service: &mut GrpcEmbedClient,
    candle_embed_service: CandleEmbedService,
) -> Result<()> {
    println!("--- Benchmark Results ---");
    println!(
        "{:<8} {:>15} {:>15} {:>15} {:>15} {:>15} {:>15} {:>15} {:>15}",
        "Total",
        "FastEmbed (ms)",
        "gRPC (ms)",
        "Candle (ms)",
        "Fast (sent/s)",
        "gRPC (sent/s)",
        "Candle (sent/s)",
        "gRPC/Fast×",
        "Fast/Candle×"
    );

    for &n in TOTAL_SIZES {
        let inputs = make_inputs(n);

        // FastEmbed
        let t0 = Instant::now();
        for chunk in inputs.chunks(CHUNK_SIZE) {
            let _ = fastembed_model.embed(chunk.to_vec(), None)?;
        }
        let fast_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let fast_sps = n as f64 / (fast_ms / 1000.0);

        // gRPC
        let t1 = Instant::now();
        for chunk in inputs.chunks(CHUNK_SIZE) {
            let _ = grpc_embed_service.embed(chunk.to_vec())?;
        }
        let grpc_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let grpc_sps = n as f64 / (grpc_ms / 1000.0);

        // Candle
        let t2 = Instant::now();
        for chunk in inputs.chunks(CHUNK_SIZE) {
            let _ = candle_embed_service.embed(chunk.to_vec())?;
        }
        let candle_ms = t2.elapsed().as_secs_f64() * 1000.0;
        let candle_sps = n as f64 / (candle_ms / 1000.0);

        // Speedup ratios
        let grpc_speedup = grpc_sps / fast_sps;
        let fast_speedup = fast_sps / candle_sps;

        println!(
            "{:<8} {:>15.2} {:>15.2} {:>15.2} {:>15.1} {:>15.1} {:>15.1} {:>15.2} {:>15.2}",
            n,
            fast_ms,
            grpc_ms,
            candle_ms,
            fast_sps,
            grpc_sps,
            candle_sps,
            grpc_speedup,
            fast_speedup
        );
    }

    Ok(())
}
