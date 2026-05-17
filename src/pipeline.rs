use crate::{Args, bake::burn_subtitles, srt, wav};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub fn subtitle_pipeline(args: &Args) -> Result<()> {
    let whisper_model_path = resolve_support_file(&args.whisper_model_path);
    let vad_model_path = resolve_support_file(&args.vad_model_path);

    let (wav_path, target_srt_path) = temp_paths_for_input(&args.file);

    wav::extract_wav(&args.file, &wav_path)
        .with_context(|| format!("failed to extract wav from {}", args.file.display()))?;

    srt::str_translate(
        &wav_path,
        &target_srt_path,
        &whisper_model_path,
        &vad_model_path,
        &args.language,
    )?;

    burn_subtitles(&args.file, &target_srt_path)
        .with_context(|| format!("failed to burn subtitles into {}", args.file.display()))?;

    Ok(())
}

fn resolve_support_file(path: &Path) -> PathBuf {
    if path.exists() {
        return path.to_path_buf();
    }

    if let Ok(exe_path) = std::env::current_exe() {
        for ancestor in exe_path.ancestors() {
            let candidate = ancestor.join(path);
            if candidate.exists() {
                return candidate;
            }
        }
    }

    path.to_path_buf()
}

fn temp_paths_for_input(input_file: &Path) -> (PathBuf, PathBuf) {
    let wav_path = input_file.with_extension("wav");
    let target_srt_file = wav_path.with_extension("cn.srt");

    (wav_path, target_srt_file)
}
