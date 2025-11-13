use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::BertModel;
use std::path::PathBuf;
use tokenizers::{PaddingParams, Tokenizer};

pub struct CandleEmbedService {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl CandleEmbedService {
    pub fn new() -> Result<Self> {
        let device = Device::Cpu;
        let model_dir = PathBuf::from("./all-MiniLM-L6-v2");
        let config = serde_json::from_reader(std::fs::File::open(model_dir.join("config.json"))?)?;
        let mut tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_dir.join("model.safetensors")],
                DType::F32,
                &device,
            )?
        };
        let model = BertModel::load(vb, &config)?;

        let _ = tokenizer
            .with_padding(Some(PaddingParams::default()))
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: config.max_position_embeddings,
                ..Default::default()
            }));

        Ok(Self {
            device,
            tokenizer,
            model,
        })
    }

    /// Run a forward pass through the Candle model and return normalized embeddings
    pub fn embed(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let tokens = self.tokenizer.encode_batch(inputs, true).unwrap();

        let token_ids = Tensor::stack(
            &tokens
                .iter()
                .map(|t| Tensor::new(t.get_ids(), &self.device))
                .collect::<Result<Vec<_>, _>>()?,
            0,
        )?;
        let token_type_ids = token_ids.zeros_like()?;
        let attention_mask = Tensor::stack(
            &tokens
                .iter()
                .map(|t| Tensor::new(t.get_attention_mask(), &self.device))
                .collect::<Result<Vec<_>, _>>()?,
            0,
        )?;

        let embeddings = self
            .model
            .forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        // Mean Pooling (masked average)
        let (b, n, h) = embeddings.dims3()?;
        let mask = attention_mask
            .to_dtype(DType::F32)?
            .unsqueeze(2)? // [b, n, 1]
            .broadcast_as((b, n, h))?; // [b, n, h]

        let masked = (&embeddings * &mask)?;
        let sum_embeddings = masked.sum(1)?; // [b, h]

        let sum_mask = mask.sum(1)?; // [b, h] after sum
        let pooled = (&sum_embeddings / &sum_mask)?;

        // L2 Normalization
        // Calculate norm across the embedding dimension (dim=1)
        let norm = pooled.sqr()?.sum_keepdim(1)?; // [b, 1]
        let norm = norm.sqrt()?; // [b, 1]

        // Broadcast norm to match pooled shape [b, h]
        let norm_broadcast = norm.broadcast_as(pooled.shape())?;
        let normalized = pooled.broadcast_div(&norm_broadcast)?;

        Ok(normalized.to_vec2::<f32>()?)
    }
}
