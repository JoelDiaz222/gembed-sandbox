use crate::tei::v1::embed_client::EmbedClient;
use crate::tei::v1::EmbedBatchRequest;
use crate::{Embedder, ModelType};
use anyhow::Result;
use tokio::runtime::Runtime;
use tonic::transport::Channel;

pub struct GrpcEmbedder {
    runtime: Runtime,
    client: EmbedClient<Channel>,
    model_id: String,
}

impl GrpcEmbedder {
    pub fn new(model_type: ModelType) -> Result<Self> {
        let runtime = Runtime::new()?;
        let client =
            runtime.block_on(async { EmbedClient::connect("http://localhost:50051").await })?;

        Ok(GrpcEmbedder {
            runtime,
            client,
            model_id: model_type.model_id().to_string(),
        })
    }
}

impl Embedder for GrpcEmbedder {
    fn embed(&mut self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let req = tonic::Request::new(EmbedBatchRequest {
            inputs,
            truncate: true,
            normalize: true,
            truncation_direction: 0,
            prompt_name: None,
            dimensions: None,
            model: self.model_id.clone(),
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
