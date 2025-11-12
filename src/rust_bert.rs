use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use anyhow::Result;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsBuilder;
use ndarray::Array1;
use rust_bert::pipelines::common::{ModelResource, ModelType};
use rust_bert::pipelines::summarization::{SummarizationConfig, SummarizationModel};
use rust_bert::pipelines::zero_shot_classification::ZeroShotClassificationModel;
use rust_bert::resources::LocalResource;

fn compute_similarity() -> Result<()> {
    let start = Instant::now();

    let model = SentenceEmbeddingsBuilder::local("/Users/joeldiaz/RustroverProjects/pgnlp/all-MiniLM-L6-v2")
        .create_model()?;

    let inputs = ["Linux", "Ubuntu"];

    let embeddings = model.encode(&inputs)?;

    let embedding1 = Array1::from(embeddings[0].clone());
    let embedding2 = Array1::from(embeddings[1].clone());

    let similarity = cosine_similarity(&embedding1, &embedding2);
    println!("Cosine similarity: {:.4}", similarity);

    println!("Elapsed time: {:.2?}", start.elapsed());
    Ok(())
}

/// Calculates cosine similarity between two vectors
fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot_product = a.dot(b);
    let norm_a = a.dot(a).sqrt();
    let norm_b = b.dot(b).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

fn summarize() -> Result<()> {
    let config = SummarizationConfig::new(
        ModelType::Bart,
        ModelResource::Torch(Box::new(LocalResource {
            local_path: "/Users/joeldiaz/Downloads/bart-large-cnn/rust_model.ot".into(),
        })),
        LocalResource {
            local_path: "/Users/joeldiaz/Downloads/bart-large-cnn/config.json".into(),
        },
        LocalResource {
            local_path: "/Users/joeldiaz/Downloads/bart-large-cnn/vocab.json".into(),
        },
        Some(LocalResource {
            local_path: "/Users/joeldiaz/Downloads/bart-large-cnn/merges.txt".into(),
        }),
    );

    let summarization_model = SummarizationModel::new(config)?;

    let input = ["In findings published Tuesday in Cornell University's arXiv by a team of scientists \
from the University of Montreal and a separate report published Wednesday in Nature Astronomy by a team \
from University College London (UCL), the presence of water vapour was confirmed in the atmosphere of K2-18b, \
a planet circling a star in the constellation Leo. This is the first such discovery in a planet in its star's \
habitable zone — not too hot and not too cold for liquid water to exist. The Montreal team, led by Björn Benneke, \
used data from the NASA's Hubble telescope to assess changes in the light coming from K2-18b's star as the planet \
passed between it and Earth. They found that certain wavelengths of light, which are usually absorbed by water, \
weakened when the planet was in the way, indicating not only does K2-18b have an atmosphere, but the atmosphere \
contains water in vapour form. The team from UCL then analyzed the Montreal team's data using their own software \
and confirmed their conclusion. This was not the first time scientists have found signs of water on an exoplanet, \
but previous discoveries were made on planets with high temperatures or other pronounced differences from Earth. \
\"This is the first potentially habitable planet where the temperature is right and where we now know there is water,\" \
said UCL astronomer Angelos Tsiaras. \"It's the best candidate for habitability right now.\" \"It's a good sign\", \
said Ryan Cloutier of the Harvard–Smithsonian Center for Astrophysics, who was not one of either study's authors. \
\"Overall,\" he continued, \"the presence of water in its atmosphere certainly improves the prospect of K2-18b being \
a potentially habitable planet, but further observations will be required to say for sure. \"
K2-18b was first identified in 2015 by the Kepler space telescope. It is about 110 light-years from Earth and larger \
but less dense. Its star, a red dwarf, is cooler than the Sun, but the planet's orbit is much closer, such that a year \
on K2-18b lasts 33 Earth days. According to The Guardian, astronomers were optimistic that NASA's James Webb space \
telescope — scheduled for launch in 2021 — and the European Space Agency's 2028 ARIEL program, could reveal more \
about exoplanets like K2-18b."];

    let output: Vec<String> = summarization_model
        .summarize(&input)?;

    println!("{}", output.get(0).unwrap());

    Ok(())
}

fn qa() -> Result<()> {
    use rust_bert::pipelines::question_answering::{QaInput, QuestionAnsweringModel};
    let qa_model = QuestionAnsweringModel::new(Default::default())?;

    let question = String::from("What do you suggest to optimize this query? Index or analyze?");
    let context = String::from("Query: SELECT val, COUNT(*) FROM test_data GROUP BY val;
NOTICE:  Plan Node: Agg | Estimated rows: 200 | Actual rows: 101
NOTICE:    Plan Node: SeqScan | Estimated rows: 10170");

    let answers = qa_model.predict(&[QaInput { question, context }], 1, 32);

    println!("{:?}", answers);

    Ok(())
}

fn main() -> Result<()> {
    let classification_model = ZeroShotClassificationModel::new(Default::default())?;
    let context = "What do you suggest to optimize this? Query: SELECT val, COUNT(*) FROM test_data GROUP BY val;
NOTICE:  Plan Node: Agg | Estimated rows: 200 | Actual rows: 101
NOTICE:    Plan Node: SeqScan | Estimated rows: 10170";

    let candidate_labels = &["index", "analyze", "nothing"];
    let output = classification_model.predict_multilabel(&[context], candidate_labels, None, 128);

    println!("{:?}", output);

    Ok(())
}
