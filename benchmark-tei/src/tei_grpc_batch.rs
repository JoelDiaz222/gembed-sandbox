use crate::tei::v1::embed_client::EmbedClient;
use crate::tei::v1::EmbedBatchRequest;
use anyhow::Result;
use tokio::runtime::Runtime;
use tonic::transport::Channel;

pub struct TeiGrpcBatch {
    runtime: Runtime,
    client: EmbedClient<Channel>,
}

impl TeiGrpcBatch {
    pub fn new() -> Result<Self> {
        let runtime = Runtime::new()?;
        let client =
            runtime.block_on(async { EmbedClient::connect("http://localhost:50052").await })?;

        Ok(TeiGrpcBatch { runtime, client })
    }

    pub fn embed(&mut self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let req = tonic::Request::new(EmbedBatchRequest {
            inputs: inputs,
            truncate: true,
            normalize: true,
            truncation_direction: 0,
            prompt_name: None,
            dimensions: None,
            model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        });

        let response = self.runtime.block_on(self.client.embed_batch(req))?;

        let embeddings = response
            .into_inner()
            .embeddings
            .into_iter()
            .map(|e| e.values)
            .collect::<Vec<_>>();

        Ok(embeddings)
    }
}
