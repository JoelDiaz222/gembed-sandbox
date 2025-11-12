use std::fs::File;
use std::io::{BufRead, BufReader};
use anyhow::Result;
use std::time::Instant;

pub mod tei {
    pub mod v1 {
        tonic::include_proto!("tei.v1");
    }
}

use tei::v1::{embed_client::EmbedClient, EmbedRequest};

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

    let mut embeddings: Vec<Vec<f32>> = Vec::new();
    let start = Instant::now();

    for input in inputs {
        let request = tonic::Request::new(EmbedRequest {
            inputs: input,
            truncate: true,
            normalize: true,
            truncation_direction: 0,
            prompt_name: None,
            dimensions: None,
        });

        let response = rt.block_on(async {
            client.embed(request).await
        })?;

        embeddings.push(response.into_inner().embeddings);
    }

    println!("Generated {} embeddings", embeddings.len());
    println!("Elapsed time: {:.2?}", start.elapsed());

    Ok(())
}
