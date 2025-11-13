use anyhow::Result;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

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

pub struct TeiHttp {
    client: Client,
}

impl TeiHttp {
    pub fn new() -> Result<Self> {
        Ok(TeiHttp {
            client: Client::new(),
        })
    }

    pub fn embed(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let req_body = EmbeddingRequest {
            model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            input: inputs,
        };

        let resp: EmbeddingResponse = self
            .client
            .post("http://localhost:8080/v1/embeddings")
            .json(&req_body)
            .send()?
            .json()?;

        let embeddings = resp
            .data
            .into_iter()
            .map(|d| d.embedding)
            .collect::<Vec<_>>();

        Ok(embeddings)
    }
}
