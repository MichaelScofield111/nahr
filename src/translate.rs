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
    // to init once
    model: marian::MTModel,
}

const JA_ZH_ALIAS_TARGET: &str = "lm_head.weight";

trait MarianConfigExt {
    fn opus_mt_ja_zh() -> Self;
}

impl MarianConfigExt for marian::Config {
    fn opus_mt_ja_zh() -> Self {
        Self {
            activation_function: candle_nn::Activation::Swish,
            d_model: 512,
            decoder_attention_heads: 8,
            decoder_ffn_dim: 2048,
            decoder_layers: 6,
            decoder_start_token_id: 65000,
            decoder_vocab_size: Some(65001),
            encoder_attention_heads: 8,
            encoder_ffn_dim: 2048,
            encoder_layers: 6,
            eos_token_id: 0,
            forced_eos_token_id: 0,
            is_encoder_decoder: true,
            max_position_embeddings: 512,
            pad_token_id: 65000,
            scale_embedding: true,
            share_encoder_decoder_embeddings: true,
            use_cache: true,
            vocab_size: 65001,
        }
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum LanguagePair {
    // value(name = "en-zh")
    EnZh,
    // value(name = "ja-zh")
    JaZh,
}
impl MarianTranslator {
    pub fn try_new(language_pair: LanguagePair) -> anyhow::Result<Self> {
        let config = match language_pair {
            LanguagePair::EnZh => marian::Config::opus_mt_en_zh(),
            LanguagePair::JaZh => marian::Config::opus_mt_ja_zh(),
        };

        let tokenizer_default_repo = match language_pair {
            LanguagePair::EnZh => "KeighBee/candle-marian",
            // this repo is special: it hosts ja-zh tokenizer configs
            LanguagePair::JaZh => "MichaelScofield111/nahr",
        };

        // encoder tokenizer (source language)
        let tokenizer = {
            let filename = match language_pair {
                LanguagePair::EnZh => "tokenizer-marian-base-en-zh-en.json",
                LanguagePair::JaZh => "tokenizer-marian-base-ja-zh-ja.json",
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
                LanguagePair::JaZh => "tokenizer-marian-base-ja-zh-zh.json",
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
            let repo = match language_pair {
                LanguagePair::EnZh => api.repo(hf_hub::Repo::with_revision(
                    "Helsinki-NLP/opus-mt-en-zh".to_string(),
                    hf_hub::RepoType::Model,
                    "refs/pr/13".to_string(),
                )),
                LanguagePair::JaZh => api.repo(hf_hub::Repo::with_revision(
                    "shun89/opus-mt-ja-zh".to_string(),
                    hf_hub::RepoType::Model,
                    "refs/pr/2".to_string(),
                )),
            };
            repo.get("model.safetensors")?
        };

        // --- 修复：在 rename 之前验证原始文件里的实际 key ---
        let vb_raw =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&model_path], DType::F32, &device)? };
        if language_pair == LanguagePair::JaZh && !vb_raw.contains_tensor(JA_ZH_ALIAS_TARGET) {
            anyhow::bail!(
                "ja-zh model at {} is missing expected tensor '{}'; \
                        the safetensors file may be corrupt or from an incompatible conversion",
                model_path.display(),
                JA_ZH_ALIAS_TARGET,
            );
        }

        // Apply name remapping for ja-zh (see translate.rs module doc for why)
        let vb = match language_pair {
            LanguagePair::EnZh => vb_raw,
            LanguagePair::JaZh => vb_raw.rename_f(|name| remap_ja_zh_tensor_name(name).to_string()),
        };

        let model = marian::MTModel::new(&config, vb)?;

        Ok(Self {
            config,
            tokenizer,
            tokenizer_dec,
            device,
            model,
        })
    }

    pub fn translate_text(&mut self, text: &str) -> anyhow::Result<String> {
        self.tokenizer_dec.clear();
        let mut output = String::new();

        // Marian's KV-cache is local to each forward pass; we only need to reset
        // the decoder token stream above — no need to rebuild the whole model.
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

fn remap_ja_zh_tensor_name(name: &str) -> &str {
    match name {
        "model.shared.weight"
        | "model.encoder.embed_tokens.weight"
        | "model.decoder.embed_tokens.weight" => JA_ZH_ALIAS_TARGET,
        _ => name,
    }
}

#[cfg(test)]
mod tests {
    use super::remap_ja_zh_tensor_name;

    #[test]
    fn ja_zh_aliases_map_to_lm_head() {
        assert_eq!(
            remap_ja_zh_tensor_name("model.shared.weight"),
            "lm_head.weight"
        );
        assert_eq!(
            remap_ja_zh_tensor_name("model.encoder.embed_tokens.weight"),
            "lm_head.weight"
        );
        assert_eq!(
            remap_ja_zh_tensor_name("model.decoder.embed_tokens.weight"),
            "lm_head.weight"
        );
    }

    #[test]
    fn unrelated_tensor_names_remain_unchanged() {
        assert_eq!(
            remap_ja_zh_tensor_name("model.encoder.layers.0.fc1.weight"),
            "model.encoder.layers.0.fc1.weight"
        );
    }
}
