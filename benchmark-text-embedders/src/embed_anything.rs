use crate::{Embedder, ModelType};
use anyhow::Result;
use embed_anything::embeddings::embed::{Embedder as EAEmbedder, EmbedderBuilder};
use embed_anything::embeddings::local::text_embedding::ONNXModel;
use tokio::runtime::Runtime;

pub struct EmbedAnythingEmbedder {
    runtime: Runtime,
    embedder: EAEmbedder,
}

impl EmbedAnythingEmbedder {
    pub fn new(model_type: ModelType) -> Result<Self> {
        let runtime = Runtime::new()?;

        let onnx_model = match model_type {
            ModelType::AllMiniLML6V2 => ONNXModel::AllMiniLML6V2,
            ModelType::BGELargeENV15 => ONNXModel::BGELargeENV15,
        };

        let embedder = EmbedderBuilder::new()
            .model_architecture("bert")
            .onnx_model_id(Some(onnx_model))
            .from_pretrained_onnx()?;

        Ok(Self { runtime, embedder })
    }
}

impl Embedder for EmbedAnythingEmbedder {
    fn embed(&mut self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let refs: Vec<&str> = inputs.iter().map(|s| s.as_str()).collect();

        let embeddings = self
            .runtime
            .block_on(self.embedder.embed(&refs, None, None))?;

        let vectors: Vec<Vec<f32>> = embeddings
            .into_iter()
            .map(|e| e.to_dense())
            .collect::<Result<_, _>>()?;

        Ok(vectors)
    }
}
