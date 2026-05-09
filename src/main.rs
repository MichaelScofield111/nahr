use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

mod bake;
mod cn_srt;
mod pipeline;
mod srt;
mod wav;

#[derive(Debug, Parser)]
#[command(name = "nahr")]
#[command(version = "1.0")]
#[command(
    about = "transcribes videos and translates subtitles ",
    long_about = "A powerful CLI tool that transcribes videos and translates subtitles to Simplified Chinese with CPU acceleration support."
)]
struct Args {
    // input .mp4 file
    #[arg(short, long, value_name = "FILE")]
    input_file: PathBuf,

    #[arg(short = 'l', long, value_name = "LANG", default_value = "en")]
    language: String,

    #[arg(
        long,
        value_name = "FILE",
        default_value = "models/ggml-base.bin",
        hide = true
    )]
    whisper_model_path: PathBuf,

    #[arg(
        long,
        value_name = "FILE",
        default_value = "models/ggml-silero-v5.1.2.bin",
        hide = true
    )]
    vad_model_path: PathBuf,

    #[arg(long, value_name = "BIN", default_value = "python3", hide = true)]
    python_bin: String,

    #[arg(
        long,
        value_name = "FILE",
        default_value = "script/trans.py",
        hide = true
    )]
    translator_script_path: PathBuf,

    #[arg(long, default_value_t = false)]
    keep_temp: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    pipeline::subtitle_pipeline(&args)?;
    Ok(())
}
