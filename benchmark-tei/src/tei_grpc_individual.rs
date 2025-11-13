use crate::tei::v1::embed_client::EmbedClient;
use crate::tei::v1::EmbedRequest;
use anyhow::Result;
use tokio::runtime::Runtime;
use tonic::transport::Channel;

pub struct TeiGrpcIndividual {
    runtime: Runtime,
    client: EmbedClient<Channel>,
}

impl TeiGrpcIndividual {
    pub fn new() -> Result<Self> {
        let runtime = Runtime::new()?;
        let client =
            runtime.block_on(async { EmbedClient::connect("http://localhost:50052").await })?;

        Ok(TeiGrpcIndividual { runtime, client })
    }

    pub fn embed(&mut self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let mut embeddings = Vec::new();

        for input in inputs {
            let req = tonic::Request::new(EmbedRequest {
                inputs: input,
                truncate: true,
                normalize: true,
                truncation_direction: 0,
                prompt_name: None,
                dimensions: None,
                model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            });

            let response = self.runtime.block_on(self.client.embed(req))?;
            embeddings.push(response.into_inner().embeddings);
        }

        Ok(embeddings)
    }
}
