use anyhow::Error as E;

use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::models::marian;

use candle_core::{DType, Device, Tensor};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

pub struct MarianTranslator {
    config: marian::Config,
    tokenizer: Tokenizer,
    tokenizer_dec: TokenOutputStream,
    device: Device,
    model_path: std::path::PathBuf, // 改为存路径
}

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum LanguagePair {
    // value(name = "en-zh")
    EnZh,
}
impl MarianTranslator {
    fn build_model(&self) -> anyhow::Result<marian::MTModel> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&self.model_path], DType::F32, &self.device)?
        };
        Ok(marian::MTModel::new(&self.config, vb)?)
    }

    pub fn translate(language_pair: &LanguagePair) -> anyhow::Result<Self> {
        let config = match language_pair {
            LanguagePair::EnZh => marian::Config::opus_mt_en_zh(),
        };

        let tokenizer_default_repo = match language_pair {
            LanguagePair::EnZh => "KeighBee/candle-marian",
        };

        // encoder tokenizer (source language)
        let tokenizer = {
            let filename = match language_pair {
                LanguagePair::EnZh => "tokenizer-marian-base-en-zh-en.json",
            };
            let path = Api::new()?
                .model(tokenizer_default_repo.to_string())
                .get(filename)?;
            Tokenizer::from_file(path).map_err(E::msg)?
        };

        // decoder tokenizer (target language)
        let tokenizer_dec = {
            let filename = match language_pair {
                LanguagePair::EnZh => "tokenizer-marian-base-en-zh-zh.json",
            };
            let path = Api::new()?
                .model(tokenizer_default_repo.to_string())
                .get(filename)?;
            Tokenizer::from_file(path).map_err(E::msg)?
        };

        let tokenizer_dec = TokenOutputStream::new(tokenizer_dec);

        let device = candle_examples::device(true)?;

        let model_path = {
            let api = Api::new()?;
            let api = match language_pair {
                LanguagePair::EnZh => api.repo(hf_hub::Repo::with_revision(
                    "Helsinki-NLP/opus-mt-en-zh".to_string(),
                    hf_hub::RepoType::Model,
                    "refs/pr/13".to_string(),
                )),
            };
            api.get("model.safetensors")?
        };

        Ok(Self {
            config,
            tokenizer,
            tokenizer_dec,
            device,
            model_path, // 存路径而不是 model
        })
    }

    pub fn translate_text(&mut self, text: &str) -> anyhow::Result<String> {
        self.tokenizer_dec.clear();
        let mut output = String::new();

        // 每次翻译重建 model，彻底消除 cache 污染
        let mut model = self.build_model()?;

        let mut logits_processor =
            candle_transformers::generation::LogitsProcessor::new(1337, None, None);

        let encoder_xs = {
            let mut tokens = self
                .tokenizer
                .encode(text, true)
                .map_err(E::msg)?
                .get_ids()
                .to_vec();
            tokens.push(self.config.eos_token_id);
            let tokens = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
            model.encoder().forward(&tokens, 0)?
        };

        let mut token_ids = vec![self.config.decoder_start_token_id];
        for index in 0..1000 {
            let context_size = if index >= 1 { 1 } else { token_ids.len() };
            let start_pos = token_ids.len().saturating_sub(context_size);
            let input_ids = Tensor::new(&token_ids[start_pos..], &self.device)?.unsqueeze(0)?;
            let logits = model.decode(&input_ids, &encoder_xs, start_pos)?;
            let logits = logits.squeeze(0)?;
            let logits = logits.get(logits.dim(0)? - 1)?;
            let token = logits_processor.sample(&logits)?;
            token_ids.push(token);

            if let Some(t) = self.tokenizer_dec.next_token(token)? {
                output.push_str(&t);
            }

            if token == self.config.eos_token_id || token == self.config.forced_eos_token_id {
                break;
            }
        }

        if let Some(rest) = self.tokenizer_dec.decode_rest().map_err(E::msg)? {
            output.push_str(&rest);
        }

        Ok(output)
    }
}
