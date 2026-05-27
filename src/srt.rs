use anyhow::{Context, Result, anyhow, bail};
use hound::{SampleFormat, WavReader};
use std::fs::File;
use std::io::Write;
use std::path::Path;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperVadContext,
    WhisperVadContextParams, WhisperVadParams,
};

use crate::translate::{LanguagePair, MarianTranslator};

const VAD_THRESHOLD: f32 = 0.35;
const VAD_MIN_SPEECH_DURATION_MS: i32 = 120;
const VAD_MIN_SILENCE_DURATION_MS: i32 = 60;

#[derive(Debug, Clone)]
struct SrtItem {
    // centiseconds (1/100s)
    start_cs: i64,
    end_cs: i64,
    text: String,
}

pub fn str_translate<W: AsRef<Path>, WM: AsRef<Path>, VM: AsRef<Path>>(
    wav_path: W,
    target_srt_path: W,
    whisper_model: WM,
    vad_model: VM,
    language: &str,
) -> Result<()> {
    let str_item = wav_to_srt(&wav_path, &whisper_model, &vad_model, language)
        .with_context(|| "failed to generate source subtitles".to_string())?;
    if str_item.is_empty() {
        bail!(
            "source subtitle generation produced no entries for {}",
            wav_path.as_ref().display()
        );
    }

    let mut file = File::create(&target_srt_path).with_context(|| {
        format!(
            "failed to create srt file {}",
            target_srt_path.as_ref().display()
        )
    })?;

    let translated_entries = translate_srt_entries(&mut file, &str_item, language)?;
    if translated_entries == 0 {
        bail!(
            "translated subtitle generation produced no entries for {}",
            target_srt_path.as_ref().display()
        );
    }
    Ok(())
}
fn wav_to_srt<W: AsRef<Path>, WM: AsRef<Path>, VM: AsRef<Path>>(
    wav_path: W,
    whisper_model: WM,
    vad_model: VM,
    language: &str,
) -> Result<Vec<SrtItem>> {
    let (samples_i16, sample_rate, _channels) = check_wav_format(wav_path.as_ref())?;

    let mut audio_f32 = vec![0.0f32; samples_i16.len()];
    whisper_rs::convert_integer_to_float_audio(&samples_i16, &mut audio_f32)
        .context("failed to convert i16 wav samples to f32")?;

    let mut vad_ctx_params = WhisperVadContextParams::default();
    vad_ctx_params.set_n_threads(1);
    vad_ctx_params.set_use_gpu(false);

    let vad_model_path = vad_model
        .as_ref()
        .to_str()
        .context("vad model path contains non-utf8 characters")?;
    let mut vad_ctx = WhisperVadContext::new(vad_model_path, vad_ctx_params)
        .context("failed to load vad model")?;

    // Keep VAD boundaries deterministic by disabling extra overlap/padding here.
    let mut vad_params = WhisperVadParams::new();
    vad_params.set_threshold(VAD_THRESHOLD);
    vad_params.set_min_speech_duration(VAD_MIN_SPEECH_DURATION_MS);
    vad_params.set_min_silence_duration(VAD_MIN_SILENCE_DURATION_MS);
    vad_params.set_speech_pad(0);
    vad_params.set_samples_overlap(0.0);

    let segments = vad_ctx
        .segments_from_samples(vad_params, &audio_f32)
        .context("failed to run vad segmentation")?;
    if segments.num_segments() == 0 {
        bail!(
            "audio extraction succeeded for {}, but VAD detected no speech segments; try relaxing VAD parameters or using a different input",
            wav_path.as_ref().display()
        );
    }

    let ctx = WhisperContext::new_with_params(whisper_model, WhisperContextParameters::default())
        .context("failed to load whisper model")?;
    let mut state = ctx
        .create_state()
        .context("failed to create whisper state")?;

    let mut srt_items: Vec<SrtItem> = Vec::new();
    let segment_pad_cs: i64 = 20; // 200ms manual boundary padding
    let language = language.trim();

    for seg in segments {
        let seg_start_cs = seg.start as i64;
        let seg_end_cs = seg.end as i64;

        let clip_start_cs = (seg_start_cs - segment_pad_cs).max(0);
        let clip_end_cs = seg_end_cs + segment_pad_cs;

        let start_idx = ((clip_start_cs as f64 / 100.0) * sample_rate as f64) as usize;
        let end_idx = ((clip_end_cs as f64 / 100.0) * sample_rate as f64) as usize;
        if start_idx >= audio_f32.len() || end_idx <= start_idx {
            continue;
        }

        let end_idx = end_idx.min(audio_f32.len());
        let chunk = &audio_f32[start_idx..end_idx];

        let mut params = FullParams::new(SamplingStrategy::BeamSearch {
            beam_size: 5,
            patience: -1.0,
        });
        if language.is_empty() {
            params.set_language(None);
        } else {
            params.set_language(Some(language));
        }
        params.set_translate(false);
        params.set_no_context(true);
        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        state.full(params, chunk).with_context(|| {
            format!("whisper decode failed for vad segment {seg_start_cs}-{seg_end_cs}cs")
        })?;

        for ws in state.as_iter() {
            let local_start_cs = ws.start_timestamp();
            let local_end_cs = ws.end_timestamp();
            let text = ws.to_string();

            let curr = SrtItem {
                start_cs: clip_start_cs + local_start_cs,
                end_cs: clip_start_cs + local_end_cs,
                text,
            };

            if let Some(prev) = srt_items.last()
                && should_drop(prev, &curr)
            {
                continue;
            }

            srt_items.push(curr);
        }
    }

    let mut file = File::create("./assets/test.en.srt")?;
    let source_entries = write_srt_entries(&mut file, &srt_items)?;
    if source_entries == 0 {
        bail!(
            "whisper produced no subtitle entries for {}; refusing to generate empty subtitles",
            wav_path.as_ref().display()
        );
    }

    Ok(srt_items)
}

// Convert to SRT timestamp format: HH:MM:SS,mmm
fn cs_to_srt_timestamp(cs: i64) -> String {
    let total_ms = cs * 10;
    let hours = total_ms / 3_600_000;
    let minutes = (total_ms % 3_600_000) / 60_000;
    let seconds = (total_ms % 60_000) / 1_000;
    let millis = total_ms % 1_000;
    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, seconds, millis)
}

fn translate_srt_entries<W>(
    writer: &mut W,
    srt_items: &[SrtItem],
    language_type: &str,
) -> Result<usize>
where
    W: Write,
{
    if srt_items.is_empty() {
        bail!("source subtitles are empty; skipping translation");
    }

    let language_pair = match language_type {
        "en" => LanguagePair::EnZh,
        "ja" => LanguagePair::JaZh,
        _ => return Err(anyhow!("no support")),
    };

    let mut translator = MarianTranslator::try_new(language_pair)?;

    let mut sequence = 1;

    for item in srt_items {
        let text = item.text.trim();
        if text.is_empty() || item.end_cs <= item.start_cs {
            continue;
        }

        let translated = translator.translate_text(text)?;
        if translated.is_empty() {
            continue;
        }

        let line = format!(
            "{}\n{} --> {}\n{}\n\n",
            sequence,
            cs_to_srt_timestamp(item.start_cs),
            cs_to_srt_timestamp(item.end_cs),
            translated,
        );
        writer.write_all(line.as_bytes())?;
        sequence += 1;
    }
    Ok(sequence - 1)
}

fn check_wav_format(path: &Path) -> Result<(Vec<i16>, u32, u16)> {
    let reader = WavReader::open(path)?;
    let spec = reader.spec();

    if spec.sample_rate != 16000 {
        bail!("sample rate must be 16000Hz, got {}", spec.sample_rate);
    }
    if spec.channels != 1 {
        bail!("channels must be mono(1), got {}", spec.channels);
    }
    if spec.sample_format != SampleFormat::Int || spec.bits_per_sample != 16 {
        bail!("bits per sample must be 16, got {}", spec.bits_per_sample);
    }

    let samples: Vec<i16> = reader
        .into_samples::<i16>()
        .map(|x| x.map_err(|e| anyhow::anyhow!("invalid wav sample: {e}")))
        .collect::<Result<Vec<_>>>()?;

    Ok((samples, spec.sample_rate, spec.channels))
}

fn norm_text(s: &str) -> String {
    s.trim().to_lowercase()
}

// Drop duplicate segments when text matches exactly, or overlapped text includes each other.
fn should_drop(prev: &SrtItem, curr: &SrtItem) -> bool {
    let p = norm_text(&prev.text);
    let c = norm_text(&curr.text);

    if p.is_empty() || c.is_empty() {
        return false;
    }
    if p == c {
        return true;
    }

    let overlap =
        std::cmp::min(prev.end_cs, curr.end_cs) - std::cmp::max(prev.start_cs, curr.start_cs);
    let has_overlap = overlap > 0;

    has_overlap && (p.contains(&c) || c.contains(&p))
}

fn write_srt_entries<W>(writer: &mut W, srt_items: &[SrtItem]) -> Result<usize>
where
    W: Write,
{
    let mut sequence = 1;

    for item in srt_items {
        let text = item.text.trim();
        if text.is_empty() || item.end_cs <= item.start_cs {
            continue;
        }

        let line = format!(
            "{}\n{} --> {}\n{}\n\n",
            sequence,
            cs_to_srt_timestamp(item.start_cs),
            cs_to_srt_timestamp(item.end_cs),
            text,
        );
        writer.write_all(line.as_bytes())?;
        sequence += 1;
    }

    Ok(sequence - 1)
}

#[cfg(test)]
mod tests {
    use super::{SrtItem, cs_to_srt_timestamp, should_drop, write_srt_entries};
    use anyhow::Result;

    #[test]
    fn timestamp_format_matches_srt_spec() {
        assert_eq!(cs_to_srt_timestamp(372_345), "01:02:03,450");
    }

    #[test]
    fn srt_sequence_remains_contiguous_when_empty_segments_are_skipped() {
        let mut output = Vec::new();
        let items = vec![
            SrtItem {
                start_cs: 0,
                end_cs: 100,
                text: "Hello".to_string(),
            },
            SrtItem {
                start_cs: 100,
                end_cs: 200,
                text: "   ".to_string(),
            },
            SrtItem {
                start_cs: 200,
                end_cs: 300,
                text: "World".to_string(),
            },
        ];
        let written = write_srt_entries(&mut output, &items).unwrap();
        assert_eq!(written, 2);

        let rendered = String::from_utf8(output).unwrap();
        assert!(rendered.contains("1\n00:00:00,000 --> 00:00:01,000\nHello\n\n"));
        assert!(rendered.contains("2\n00:00:02,000 --> 00:00:03,000\nWorld\n\n"));
        assert!(!rendered.contains("3\n"));
    }

    #[test]
    fn write_srt_entries_returns_zero_for_invalid_items() -> Result<()> {
        let mut output = Vec::new();
        let items = vec![SrtItem {
            start_cs: 100,
            end_cs: 100,
            text: "ignored".to_string(),
        }];

        let written = write_srt_entries(&mut output, &items)?;
        assert_eq!(written, 0);
        assert!(output.is_empty());
        Ok(())
    }

    #[test]
    fn should_drop_for_exact_overlap_duplicates() {
        let prev = SrtItem {
            start_cs: 0,
            end_cs: 100,
            text: "Hello world".to_string(),
        };
        let curr = SrtItem {
            start_cs: 60,
            end_cs: 160,
            text: "hello world".to_string(),
        };
        assert!(should_drop(&prev, &curr));
    }

    #[test]
    fn should_drop_for_overlapped_contains_relation() {
        let prev = SrtItem {
            start_cs: 0,
            end_cs: 100,
            text: "hello world".to_string(),
        };
        let curr = SrtItem {
            start_cs: 50,
            end_cs: 160,
            text: "hello".to_string(),
        };
        assert!(should_drop(&prev, &curr));
    }

    #[test]
    fn should_not_drop_for_non_overlapping_segments() {
        let prev = SrtItem {
            start_cs: 0,
            end_cs: 100,
            text: "hello world".to_string(),
        };
        let curr = SrtItem {
            start_cs: 120,
            end_cs: 200,
            text: "hello".to_string(),
        };
        assert!(!should_drop(&prev, &curr));
    }
}
