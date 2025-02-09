use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::llama::{Cache, Llama, LlamaConfig, LlamaEosToks},
};
use hf_hub::{api::tokio::Api, Repo, RepoType};
use std::{error::Error, path::PathBuf};
use tokenizers::Tokenizer;
use tracing::{debug, Level};
use utils::files::hub_load_local_safetensors;
mod utils;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing::subscriber::set_global_default(
        tracing_subscriber::fmt()
            .with_max_level(Level::DEBUG)
            .finish(),
    )?;

    let repo_id = String::from("meta-llama/Llama-3.2-3B");
    let repo_type = String::from("meta-llama/Llama-3.2-3B");
    let device = Device::new_metal(0)?;
    debug!("🚧 {:#?}", device);

    let api = Api::new()?;
    debug!(" 🚧 {:#?}", api);
    let api = api.repo(Repo::with_revision(
        repo_id,
        RepoType::Model,
        String::from("main"),
    ));

    // let tokenizer_filename = api.get("tokenizer.json").await?;
    // let config_filename = api.get("config.json").await?;
    let tokenizer_filename = PathBuf::from("llama-small/tokenizer.json");
    let config_filename = PathBuf::from("llama-small/config.json");
    let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    let config = config.into_config(false);

    let weights = vec!["llama-small/model.safetensors"];
    let mut cache = Cache::new(false, DType::F16, &config, &device)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights, DType::F16, &device)? };
    let model = Llama::load(vb, &config)?;

    let mut tokenizer = Tokenizer::from_file(tokenizer_filename).unwrap();

    let end_id = tokenizer.token_to_id("</s>").map(LlamaEosToks::Single);

    let mut tokens = tokenizer
        .encode("What is a dog?", true)
        .unwrap()
        .get_ids()
        .to_vec();

    let mut logits_processor =
        LogitsProcessor::from_sampling(235264422, Sampling::All { temperature: 0.3 });

    for index in 0..100 {
        let tensor = Tensor::new(&tokens[0..], &device)?.unsqueeze(0)?;
        let logits = model
            .forward(&tensor, tokens.len() - 1, &mut cache)?
            .squeeze(0)?;
        let next_token = logits_processor.sample(&logits)?;

        tokens.push(next_token);
        let output = tokenizer.decode(&tokens, true);
        debug!("{:#?}", output);
    }

    Ok(())
}
