mod tei_grpc_embedder;
mod tei_http_embedder;

use anyhow::Result;
use postgres::{Client, NoTls};
use std::time::Instant;
use tei_grpc_embedder::TeiGrpcEmbedder;
use tei_http_embedder::TeiHttpEmbedder;

pub mod tei {
    pub mod v1 {
        tonic::include_proto!("tei.v1");
    }
}

const TEST_SIZES: &[usize] = &[16, 32, 64, 128, 256, 512];
const BATCH_SIZE: usize = 32;
const DB_CONNECTION: &str = "host=localhost port=5432 dbname=joeldiaz";

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

fn benchmark_pg_fastembed(client: &mut Client, texts: &[String]) -> Result<f64> {
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
             unnest(embed_texts('fastembed', 'Qdrant/all-MiniLM-L6-v2-onnx', ARRAY[{}])) e",
            array_str, array_str
        );

        client.execute(&sql, &[])?;
    }

    Ok(start.elapsed().as_secs_f64())
}

fn benchmark_pg_grpc(client: &mut Client, texts: &[String]) -> Result<f64> {
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
             unnest(embed_texts('grpc', 'sentence-transformers/all-MiniLM-L6-v2', ARRAY[{}])) e",
            array_str, array_str
        );

        client.execute(&sql, &[])?;
    }

    Ok(start.elapsed().as_secs_f64())
}

fn benchmark_external_grpc(
    grpc_service: &mut TeiGrpcEmbedder,
    client: &mut Client,
    texts: &[String],
) -> Result<f64> {
    let start = Instant::now();

    for chunk in texts.chunks(BATCH_SIZE) {
        let embeddings = grpc_service.embed(chunk.to_vec())?;

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

fn benchmark_external_http(
    http_service: &TeiHttpEmbedder,
    client: &mut Client,
    texts: &[String],
) -> Result<f64> {
    let start = Instant::now();

    for chunk in texts.chunks(BATCH_SIZE) {
        let embeddings = http_service.embed(chunk.to_vec())?;

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

fn warm_up_services(
    grpc_service: &mut TeiGrpcEmbedder,
    http_service: &TeiHttpEmbedder,
    client: &mut Client,
) -> Result<()> {
    let warmup_texts = make_inputs(8);

    truncate_table(client)?;
    let _ = benchmark_pg_fastembed(client, &warmup_texts)?;
    truncate_table(client)?;
    let _ = benchmark_pg_grpc(client, &warmup_texts)?;
    truncate_table(client)?;
    let _ = benchmark_external_grpc(grpc_service, client, &warmup_texts)?;
    truncate_table(client)?;
    let _ = benchmark_external_http(http_service, client, &warmup_texts)?;

    println!("Warm-up completed.\n");
    Ok(())
}

fn main() -> Result<()> {
    let mut client = Client::connect(DB_CONNECTION, NoTls)?;
    setup_database(&mut client)?;

    let mut grpc_service = TeiGrpcEmbedder::new()?;
    let http_service = TeiHttpEmbedder::new()?;

    warm_up_services(&mut grpc_service, &http_service, &mut client)?;

    println!(
        "\n{:<8} {:>15} {:>15} {:>15} {:>15} {:>15} {:>15} {:>15} {:>15}",
        "Size",
        "PG Fast (s)",
        "PG gRPC (s)",
        "Ext gRPC (s)",
        "Ext HTTP (s)",
        "PG Fast/s",
        "PG gRPC/s",
        "Ext gRPC/s",
        "Ext HTTP/s"
    );
    println!("{}", "-".repeat(140));

    let mut results = Vec::new();

    for &size in TEST_SIZES {
        let texts = make_inputs(size);

        truncate_table(&mut client)?;
        let pg_fast_time = benchmark_pg_fastembed(&mut client, &texts)?;

        truncate_table(&mut client)?;
        let pg_grpc_time = benchmark_pg_grpc(&mut client, &texts)?;

        truncate_table(&mut client)?;
        let ext_grpc_time = benchmark_external_grpc(&mut grpc_service, &mut client, &texts)?;

        truncate_table(&mut client)?;
        let ext_http_time = benchmark_external_http(&http_service, &mut client, &texts)?;

        let pg_fast_tps = size as f64 / pg_fast_time;
        let pg_grpc_tps = size as f64 / pg_grpc_time;
        let ext_grpc_tps = size as f64 / ext_grpc_time;
        let ext_http_tps = size as f64 / ext_http_time;

        println!(
            "{:<8} {:>15.3} {:>15.3} {:>15.3} {:>15.3} {:>15.1} {:>15.1} {:>15.1} {:>15.1}",
            size,
            pg_fast_time,
            pg_grpc_time,
            ext_grpc_time,
            ext_http_time,
            pg_fast_tps,
            pg_grpc_tps,
            ext_grpc_tps,
            ext_http_tps
        );

        results.push((pg_fast_tps, pg_grpc_tps, ext_grpc_tps, ext_http_tps));
    }

    println!("\n{}", "=".repeat(140));
    let avg_pg_fast = results.iter().map(|r| r.0).sum::<f64>() / results.len() as f64;
    let avg_pg_grpc = results.iter().map(|r| r.1).sum::<f64>() / results.len() as f64;
    let avg_ext_grpc = results.iter().map(|r| r.2).sum::<f64>() / results.len() as f64;
    let avg_ext_http = results.iter().map(|r| r.3).sum::<f64>() / results.len() as f64;

    println!("Average Throughput:");
    println!("  PG FastEmbed: {:.1} texts/sec", avg_pg_fast);
    println!("  PG gRPC:      {:.1} texts/sec", avg_pg_grpc);
    println!("  Ext gRPC:     {:.1} texts/sec", avg_ext_grpc);
    println!("  Ext HTTP:     {:.1} texts/sec", avg_ext_http);

    Ok(())
}
