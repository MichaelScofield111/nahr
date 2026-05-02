// command
// -- pathbuf
// --whisper-model	Whisper 模型大小: tiny, base, small, medium, large
// -- bake
// --language
// --output-dir
// --help
//

use anyhow::Result;
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

mod audio_transcode;

#[derive(Debug, Clone, ValueEnum)]
enum WhisperModel {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
}

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

    // output .mp4 file
    #[arg(short, long, value_name = "FILE")]
    output_file: Option<PathBuf>,

    // bake is to select the audio track to bake into the output file
    #[arg(long)]
    bake: bool,

    #[arg(long, value_name = "Language")]
    language: Option<String>,

    #[arg(long, default_value = "base", value_name = "MODEL")]
    whisper_model: WhisperModel,
}

fn main() -> Result<()> {
    // parser args
    let args = Args::parse();

    // extract wav from audio stream
    let wav_path = args.input_file.with_extension("wav");
    audio_transcode::extract_wav(&args.input_file, &wav_path)?;

    // match args.whisper_model {
    //     WhisperModel::Tiny => {
    //         todo!()
    //     }
    //     WhisperModel::Base => {
    //         todo!()
    //     }
    //     WhisperModel::Small => {
    //         todo!()
    //     }
    //     WhisperModel::Medium => {
    //         todo!()
    //     }
    //     WhisperModel::Large => {
    //         todo!()
    //     }
    // }
    Ok(())
}
