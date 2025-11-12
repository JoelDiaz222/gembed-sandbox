use std::fs::File;
use std::io::{BufRead, BufReader};
use anyhow::Result;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Serialize)]
struct EmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

fn main() -> Result<()> {
    let client = Client::new();

    let file = File::open("/Users/joeldiaz/PycharmProjects/CBDE-L1/data_used/bookcorpus_sentences.txt")?;
    let reader = BufReader::new(file);
    let inputs: Vec<String> = reader
        .lines()
        .filter_map(Result::ok)
        .collect();

    let batch_size = 32;
    let mut embeddings: Vec<Vec<f32>> = Vec::new();
    let start = Instant::now();

    for chunk in inputs.chunks(batch_size) {
        let req_body = EmbeddingRequest {
            model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            input: chunk.to_vec(),
        };

        let resp: EmbeddingResponse = client
            .post("http://localhost:8080/v1/embeddings")
            .json(&req_body)
            .send()?
            .json()?;

        embeddings.extend(resp.data.into_iter().map(|d| d.embedding));
    }

    println!("Generated {} embeddings", embeddings.len());
    println!("Elapsed time: {:.2?}", start.elapsed());

    Ok(())
}
