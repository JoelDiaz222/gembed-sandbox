use anyhow::Result;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use tei::v1::{embed_client::EmbedClient, EmbedBatchRequest};

pub mod tei {
    pub mod v1 {
        tonic::include_proto!("tei.v1");
    }
}

const CHUNK_SIZE: usize = 32;

fn main() -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    let mut client = rt.block_on(async {
        EmbedClient::connect("http://localhost:8080").await
    })?;

    let file = File::open("/Users/joeldiaz/PycharmProjects/CBDE-L1/data_used/bookcorpus_sentences.txt")?;
    let reader = BufReader::new(file);
    let inputs: Vec<String> = reader
        .lines()
        .filter_map(Result::ok)
        .collect();

    let mut total_embeddings = 0;
    let start = Instant::now();

    for chunk in inputs.chunks(CHUNK_SIZE) {
        let request = tonic::Request::new(EmbedBatchRequest {
            inputs: chunk.iter().cloned().collect(),
            truncate: true,
            normalize: true,
            truncation_direction: 0,
            prompt_name: None,
            dimensions: None,
        });

        let response = rt.block_on(async { client.embed_batch(request).await })?;
        let batch_response = response.into_inner();
        total_embeddings += batch_response.embeddings.len();

        for (i, embedding) in batch_response.embeddings.iter().enumerate() {
            println!(
                "Sentence {}: first 5 values: {:?}",
                i,
                &embedding.values[..5.min(embedding.values.len())]
            );
        }
    }

    let elapsed = start.elapsed();
    println!("\nProcessed {} sentences in {:.2?}", inputs.len(), elapsed);
    println!("Total embeddings: {}", total_embeddings);

    Ok(())
}
