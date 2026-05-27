use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn burn_subtitles(input_video: &Path, subtitle_srt: &Path) -> Result<PathBuf> {
    let subtitle_filter = subtitles_filter(subtitle_srt);
    let output_video = output_video_path(input_video)?;

    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-i")
        .arg(input_video)
        .arg("-vf")
        .arg(&subtitle_filter)
        .arg("-c:a")
        .arg("copy")
        .arg(&output_video)
        .status()
        .context("Failed to run ffmpeg for subtitle burn-in")?;

    if !status.success() {
        return Err(anyhow!(
            "ffmpeg failed to burn subtitles into {}",
            output_video.display()
        ));
    }

    Ok(output_video)
}

pub fn output_video_path(input_video: &Path) -> Result<PathBuf> {
    let stem = input_video
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| {
            anyhow!(
                "Failed to derive output filename from {}",
                input_video.display()
            )
        })?;

    Ok(input_video.with_file_name(format!("{stem}_cn_bake.mp4")))
}

fn escape_for_subtitles_filter(path: &Path) -> String {
    let raw = path.to_string_lossy();
    raw.replace('\\', "\\\\")
        .replace(':', "\\:")
        .replace('\'', "\\'")
        .replace(',', "\\,")
        .replace('[', "\\[")
        .replace(']', "\\]")
}

fn subtitles_filter(path: &Path) -> String {
    format!("subtitles=filename={}", escape_for_subtitles_filter(path))
}

#[cfg(test)]
mod tests {
    use super::{output_video_path, subtitles_filter};
    use std::path::Path;

    #[test]
    fn output_video_name_uses_expected_suffix() {
        let output = output_video_path(Path::new("/tmp/demo.mp4")).unwrap();
        assert_eq!(output, Path::new("/tmp/demo_cn_bake.mp4"));
    }

    #[test]
    fn subtitle_filter_uses_filename_syntax() {
        let filter = subtitles_filter(Path::new("assets/example-ja.cn.srt"));
        assert_eq!(filter, "subtitles=filename=assets/example-ja.cn.srt");
    }
}
