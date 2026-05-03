use anyhow::{Result, bail};
use hound::WavReader;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub fn wav_to_srt<W: AsRef<Path>, M: AsRef<Path>>(
    wav_path: W,
    model_path: M,
    language: &str,
) -> Result<PathBuf> {
    // 1) 构建 whisper 上下文参数（模型加载参数）
    let context_param = WhisperContextParameters::default();

    let ctx = WhisperContext::new_with_params(model_path.as_ref(), context_param)
        .map_err(|e| anyhow::anyhow!("failed to load whisper model: {e}"))?;
    let mut state = ctx
        .create_state()
        .map_err(|e| anyhow::anyhow!("failed to create whisper state: {e}"))?;

    // 3) 配置推理参数
    // Greedy 通常更快；BeamSearch 更慢但有时更准
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

    // cpu to set
    params.set_n_threads(4);

    // to set translate en srt
    params.set_translate(false);
    params.set_language(Some(language));

    // to close terminal output
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    // 开启 token 级时间戳（配合 DTW 时更有用）
    params.set_token_timestamps(true);

    let reader = WavReader::open(wav_path.as_ref())?;
    // to load wav file information
    let spec = reader.spec();

    // whisper 期望输入是 16kHz
    if spec.sample_rate != 16000 {
        bail!("sample rate must be 16000Hz, got {}", spec.sample_rate);
    }

    // bits_per_sample 是“每个音频采样点用多少位来存”
    if spec.bits_per_sample != 16 {
        bail!("bits per sample must be 16, got {}", spec.bits_per_sample);
    }

    // 把 i16 PCM 样本转成 f32（whisper 需要 f32）
    let samples: Vec<i16> = reader
        .into_samples::<i16>()
        .map(|x| x.map_err(|e| anyhow::anyhow!("invalid wav sample: {e}")))
        .collect::<Result<Vec<_>>>()?;

    let mut audio = vec![0.0f32; samples.len()];
    whisper_rs::convert_integer_to_float_audio(&samples, &mut audio)
        .map_err(|e| anyhow::anyhow!("failed to convert audio to f32: {e}"))?;

    // 如果是双声道，转成单声道（whisper 需要 mono）
    let mono_audio = if spec.channels == 1 {
        audio
    } else if spec.channels == 2 {
        let mut output = vec![0.0f32; audio.len() / 2];
        whisper_rs::convert_stereo_to_mono_audio(&audio, &mut output)
            .map_err(|e| anyhow::anyhow!("failed to convert stereo to mono: {e}"))?;
        output
    } else {
        bail!("unsupported wav channels: {}", spec.channels);
    };

    state
        .full(params, &mono_audio)
        .map_err(|e| anyhow::anyhow!("failed to run whisper inference: {e}"))?;

    // 8) 写出 SRT 文件
    let result_srt_path = wav_path.as_ref().with_extension(format!("{language}.srt"));
    let mut file = File::create(&result_srt_path)?;

    for (idx, segment) in state.as_iter().enumerate() {
        let start = cs_to_srt_timestamp(segment.start_timestamp());
        let end = cs_to_srt_timestamp(segment.end_timestamp());
        let text = segment.to_string().trim().to_string();

        if text.is_empty() {
            continue;
        }

        writeln!(file, "{}", idx + 1)?;
        writeln!(file, "{} --> {}", start, end)?;
        writeln!(file, "{}", text)?;
        writeln!(file)?;
    }

    Ok(result_srt_path)
}

fn cs_to_srt_timestamp(cs: i64) -> String {
    let total_ms = cs * 10;
    let hours = total_ms / 3_600_000;
    let minutes = (total_ms % 3_600_000) / 60_000;
    let seconds = (total_ms % 60_000) / 1_000;
    let millis = total_ms % 1_000;
    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, seconds, millis)
}
