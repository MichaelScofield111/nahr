use anyhow::{Ok, Result, anyhow};
use rust_bert::pipelines::{
    common::ModelType,
    translation::{Language, TranslationModelBuilder},
};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::{fs::File, path::PathBuf};

pub fn gen_cnsrt<P: AsRef<Path>>(en_srt_path: P, language: &str) -> Result<PathBuf> {
    let source_lang = match language {
        "en" => Language::English,
        "ja" => Language::Japanese,
        _ => return Err(anyhow!("unsupported source language: {}", language)),
    };

    let model = TranslationModelBuilder::new()
        .with_model_type(ModelType::Marian)
        .with_source_languages(vec![source_lang])
        .with_target_languages(vec![Language::ChineseMandarin])
        .create_model()?;

    let cn_srt_path = en_srt_path.as_ref().with_extension("cn.srt");
    let reader = BufReader::new(File::open(en_srt_path)?);
    let mut writer = BufWriter::new(File::create(&cn_srt_path)?);

    for line in reader.lines() {
        let line = line?;
        if should_translate_line(&line) {
            let translated = model.translate(&[line.as_str()], None, Language::ChineseMandarin)?;
            writeln!(writer, "{}", translated.first().map_or("", |s| s.as_str()))?;
        } else {
            writeln!(writer, "{line}")?;
        }
    }

    writer.flush()?;
    Ok(cn_srt_path)
}

fn should_translate_line(line: &str) -> bool {
    let trimmed = line.trim();
    !trimmed.is_empty() && !trimmed.contains("-->") && !trimmed.chars().all(|c| c.is_ascii_digit())
}
