<p align="center">
  <img src="assets/nahr-logo-black.png" alt="nahr logo" width="260">
</p>

<h1 align="center">nahr</h1>

<p align="center">
  A Rust CLI that extracts speech from an MP4, generates Chinese subtitles, and burns them back into a new video.
</p>

## Overview

`nahr` is a local-first subtitle pipeline for videos with speech.

Given one input MP4, the current version does this:

1. Extract the audio stream to `16kHz` mono WAV
2. Run VAD + Whisper to segment and transcribe speech
3. Translate subtitles into Simplified Chinese
4. Burn the Chinese subtitles into a new MP4 with `ffmpeg`

This version is centered on a single command and a minimal setup. The translation path is now implemented in Rust, so the old Python runtime steps are no longer required for normal usage.

## Current Capabilities

- Input: one local `.mp4` file
- Source languages: `en`, `ja`
- Output subtitles: Simplified Chinese
- Output video suffix: `_cn_bake.mp4`
- Runtime style: local CLI, model-assisted, `ffmpeg`-based burn-in

## Requirements

- Rust toolchain
- `ffmpeg`
- `ffmpeg` built with subtitle filter support (`libass`)
- Internet access on first run for Hugging Face model/tokenizer downloads used by translation

Check your FFmpeg build:

```bash
ffmpeg -version
ffmpeg -filters | grep subtitles
```

## Models

`nahr` currently expects these local Whisper/VAD model files:

- `models/ggml-base.bin`
- `models/ggml-silero-v5.1.2.bin`

Download them:

```bash
mkdir -p models

curl -L "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin" \
  -o models/ggml-base.bin

curl -L "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin" \
  -o models/ggml-silero-v5.1.2.bin
```

The translation model weights and tokenizer files are fetched automatically on first use and cached by `hf-hub`.

## Build

```bash
cargo build --release
```

## Usage

Basic usage:

```bash
cargo run --release -- --file assets/example.mp4
```

Specify the source language explicitly:

```bash
cargo run --release -- \
  --file assets/example-ja.mp4 \
  --language ja
```

Command help:

```bash
cargo run -- --help
```

Current CLI:

- `-f, --file <FILE>`: input MP4 file
- `-l, --language <LANG>`: source language, default `en`
- `--keep-temp`: currently accepted by the CLI, but the pipeline already leaves intermediate files in place, so this flag does not materially change behavior in this version

## Output Files

For an input file like `demo.mp4`, the current pipeline produces:

- `demo.wav`
- `demo.cn.srt`
- `demo_cn_bake.mp4`

At the moment, intermediate files are not automatically cleaned up.

## Example Assets

The repository includes sample assets you can use for quick validation:

- `assets/example.mp4`
- `assets/example-ja.mp4`
- `assets/example-ja.wav`
- `assets/example-ja.cn.srt`
- `assets/example-ja_cn_bake.mp4`

## Troubleshooting

### `failed to extract wav ... Stream not found`

Your input video likely has no audio stream.

Check it with:

```bash
ffprobe -v error -show_entries stream=index,codec_type,codec_name -of compact input.mp4
```

### `failed to load whisper model` or `failed to load vad model`

Check that:

- the files exist under `models/`
- the filenames match the defaults above, or you pass your own model paths in code/custom builds
- the files are readable

### `audio extraction succeeded ..., but VAD detected no speech segments`

This usually means the file has little or no detectable speech, or the VAD thresholds are too strict for that audio.

### `ffmpeg failed to burn subtitles`

Common causes:

- `ffmpeg` is not installed
- your FFmpeg build does not support subtitle burn-in
- the subtitle file path cannot be resolved correctly by FFmpeg

### First run is slow

That is expected. The translation side downloads model/tokenizer artifacts on first use and caches them locally for later runs.

## Development Notes

- The `script/` directory is now a support utility area, mainly for tokenizer conversion, not a required runtime path for the main CLI.
- `cargo test` currently passes for the repository's unit tests.
- The current crate version in `Cargo.toml` is `0.1.0`, while the CLI help text reports version `1.0`; if you plan to publish releases, that metadata is worth aligning separately.

## License

[LICENSE](LICENSE)
