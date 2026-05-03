use anyhow::{Result, anyhow};
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
    let mut lines: Vec<String> = reader.lines().collect::<std::io::Result<_>>()?;

    let translatable_indices: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| should_translate_line(line).then_some(index))
        .collect();

    if !translatable_indices.is_empty() {
        let texts: Vec<&str> = translatable_indices
            .iter()
            .map(|&index| lines[index].as_str())
            .collect();
        let translated = model.translate(&texts, None, Language::ChineseMandarin)?;

        for (index, translated_line) in translatable_indices.into_iter().zip(translated) {
            lines[index] = translated_line;
        }
    }

    for line in lines {
        writeln!(writer, "{line}")?;
    }

    writer.flush()?;
    Ok(cn_srt_path)
}

fn should_translate_line(line: &str) -> bool {
    let trimmed = line.trim();
    !trimmed.is_empty() && !trimmed.contains("-->") && !trimmed.chars().all(|c| c.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::should_translate_line;

    #[test]
    fn only_subtitle_content_lines_are_translated() {
        assert!(!should_translate_line(""));
        assert!(!should_translate_line("12"));
        assert!(!should_translate_line("00:00:01,000 --> 00:00:02,000"));
        assert!(should_translate_line("Hello, world!"));
    }
}
