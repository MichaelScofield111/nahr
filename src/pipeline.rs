use crate::{
    Args,
    bake::burn_subtitles,
    cn_srt, srt, wav,
};
use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

pub fn subtitle_pipeline(args: &Args) -> Result<()> {
    let temp_paths = temp_paths_for_input(&args.input_file, &args.language);

    wav::extract_wav(&args.input_file, &temp_paths.wav_path)
        .with_context(|| format!("failed to extract wav from {}", args.input_file.display()))?;

    let source_srt_path =
        srt::wav_to_srt(&temp_paths.wav_path, &args.whisper_model_path, &args.language)
            .with_context(|| {
                format!(
                    "failed to generate source subtitles from {}",
                    temp_paths.wav_path.display()
                )
            })?;
    debug_assert_eq!(source_srt_path, temp_paths.source_srt_path);

    let cn_srt_path = cn_srt::gen_cnsrt(&source_srt_path, &args.language)
        .with_context(|| format!("failed to translate subtitles from {}", source_srt_path.display()))?;
    debug_assert_eq!(cn_srt_path, temp_paths.cn_srt_path);

    burn_subtitles(&args.input_file, &cn_srt_path)
        .with_context(|| format!("failed to burn subtitles into {}", args.input_file.display()))?;

    if should_cleanup_temp_files(args.keep_temp) {
        cleanup_temp_files(&cleanup_targets(&temp_paths))?;
    }

    Ok(())
}

struct TempPaths {
    wav_path: PathBuf,
    source_srt_path: PathBuf,
    cn_srt_path: PathBuf,
}

fn temp_paths_for_input(input_file: &Path, language: &str) -> TempPaths {
    let wav_path = input_file.with_extension("wav");
    let source_srt_path = wav_path.with_extension(format!("{language}.srt"));
    let cn_srt_path = source_srt_path.with_extension("cn.srt");

    TempPaths {
        wav_path,
        source_srt_path,
        cn_srt_path,
    }
}

fn cleanup_targets(temp_paths: &TempPaths) -> Vec<&Path> {
    vec![
        temp_paths.wav_path.as_path(),
        temp_paths.source_srt_path.as_path(),
        temp_paths.cn_srt_path.as_path(),
    ]
}

fn should_cleanup_temp_files(keep_temp: bool) -> bool {
    !keep_temp
}

fn cleanup_temp_files(paths: &[&Path]) -> Result<()> {
    for path in paths {
        if path.exists() {
            fs::remove_file(path)
                .with_context(|| format!("failed to remove temporary file {}", path.display()))?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{cleanup_targets, cleanup_temp_files, should_cleanup_temp_files, temp_paths_for_input};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn cleanup_temp_files_removes_existing_files() {
        let test_dir = unique_test_dir();
        fs::create_dir_all(&test_dir).unwrap();

        let first = test_dir.join("a.wav");
        let second = test_dir.join("b.srt");
        fs::write(&first, "temp").unwrap();
        fs::write(&second, "temp").unwrap();

        cleanup_temp_files(&[first.as_path(), second.as_path()]).unwrap();

        assert!(!first.exists());
        assert!(!second.exists());

        fs::remove_dir_all(&test_dir).unwrap();
    }

    #[test]
    fn cleanup_temp_files_skips_missing_files() {
        let test_dir = unique_test_dir();
        fs::create_dir_all(&test_dir).unwrap();

        let missing = test_dir.join("missing.srt");
        cleanup_temp_files(&[missing.as_path()]).unwrap();

        fs::remove_dir_all(&test_dir).unwrap();
    }

    #[test]
    fn keep_temp_flag_disables_cleanup() {
        assert!(!should_cleanup_temp_files(true));
        assert!(should_cleanup_temp_files(false));
    }

    #[test]
    fn cleanup_targets_include_all_intermediate_files() {
        let temp_paths = temp_paths_for_input(Path::new("/tmp/demo.mp4"), "en");
        let targets = cleanup_targets(&temp_paths);

        assert_eq!(
            targets,
            vec![
                Path::new("/tmp/demo.wav"),
                Path::new("/tmp/demo.en.srt"),
                Path::new("/tmp/demo.en.cn.srt"),
            ]
        );
    }

    fn unique_test_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        std::env::temp_dir().join(format!("nahr-tests-{nanos}"))
    }
}
