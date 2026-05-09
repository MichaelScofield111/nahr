use anyhow::{Context, Result, anyhow, bail};
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn gen_cnsrt<P: AsRef<Path>, S: AsRef<Path>>(
    source_srt_path: P,
    python_bin: &str,
    translator_script_path: S,
    output_srt_path: &Path,
) -> Result<PathBuf> {
    let source_srt_path = source_srt_path.as_ref();
    let translator_script_path = translator_script_path.as_ref();

    if !source_srt_path.exists() {
        bail!(
            "source srt file does not exist: {}",
            source_srt_path.display()
        );
    }
    if !translator_script_path.exists() {
        bail!(
            "translator script does not exist: {}",
            translator_script_path.display()
        );
    }

    let runner = translator_runner(python_bin, translator_script_path, command_exists("uv"));
    let output = runner
        .build_command(translator_script_path, source_srt_path, output_srt_path)
        .output()
        .with_context(|| {
            format!(
                "failed to run translator script {} via {}",
                translator_script_path.display(),
                runner.describe()
            )
        })?;

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "translator script failed for {} via {} (status: {}). stdout: {} stderr: {}",
            source_srt_path.display(),
            runner.describe(),
            output
                .status
                .code()
                .map(|code| code.to_string())
                .unwrap_or_else(|| "terminated by signal".to_string()),
            stdout.trim(),
            stderr.trim()
        ));
    }

    if !output_srt_path.exists() {
        bail!(
            "translator script completed without creating {}",
            output_srt_path.display()
        );
    }

    Ok(output_srt_path.to_path_buf())
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TranslatorRunner {
    Uv {
        project_dir: PathBuf,
        cache_dir: PathBuf,
    },
    Python {
        interpreter: String,
    },
}

impl TranslatorRunner {
    fn build_command(
        &self,
        translator_script_path: &Path,
        source_srt_path: &Path,
        output_srt_path: &Path,
    ) -> Command {
        match self {
            TranslatorRunner::Uv {
                project_dir,
                cache_dir,
            } => {
                let mut command = Command::new("uv");
                command
                    .arg("run")
                    .arg("--project")
                    .arg(project_dir)
                    .arg("python")
                    .arg(translator_script_path)
                    .arg(source_srt_path)
                    .arg("--output")
                    .arg(output_srt_path)
                    .env("UV_CACHE_DIR", cache_dir);
                command
            }
            TranslatorRunner::Python { interpreter } => {
                let mut command = Command::new(interpreter);
                command
                    .arg(translator_script_path)
                    .arg(source_srt_path)
                    .arg("--output")
                    .arg(output_srt_path);
                command
            }
        }
    }

    fn describe(&self) -> String {
        match self {
            TranslatorRunner::Uv { project_dir, .. } => {
                format!("uv run --project {}", project_dir.display())
            }
            TranslatorRunner::Python { interpreter } => interpreter.clone(),
        }
    }
}

fn translator_runner(
    python_bin: &str,
    translator_script_path: &Path,
    uv_available: bool,
) -> TranslatorRunner {
    if python_bin == "python3"
        && uv_available
        && let Some(project_dir) = uv_project_dir(translator_script_path)
    {
        return TranslatorRunner::Uv {
            cache_dir: project_dir.join(".uv-cache"),
            project_dir,
        };
    }

    TranslatorRunner::Python {
        interpreter: python_bin.to_string(),
    }
}

fn uv_project_dir(translator_script_path: &Path) -> Option<PathBuf> {
    let script_dir = translator_script_path.parent()?;
    let pyproject_path = script_dir.join("pyproject.toml");
    pyproject_path.exists().then(|| script_dir.to_path_buf())
}

fn command_exists(command: &str) -> bool {
    Command::new(command)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::{TranslatorRunner, gen_cnsrt, translator_runner, uv_project_dir};
    use std::path::Path;

    #[test]
    fn missing_source_srt_is_rejected() {
        let err = gen_cnsrt(
            "/tmp/does-not-exist.en.srt",
            "python3",
            Path::new("/tmp/trans.py"),
            Path::new("/tmp/out.cn.srt"),
        )
        .unwrap_err();

        assert!(err.to_string().contains("source srt file does not exist"));
    }

    #[test]
    fn uv_project_dir_detects_script_workspace() {
        let project_dir = uv_project_dir(Path::new("script/trans.py")).unwrap();
        assert!(project_dir.ends_with("script"));
    }

    #[test]
    fn default_python_prefers_uv_when_project_exists() {
        let runner = translator_runner("python3", Path::new("script/trans.py"), true);
        assert_eq!(
            runner,
            TranslatorRunner::Uv {
                project_dir: Path::new("script").to_path_buf(),
                cache_dir: Path::new("script/.uv-cache").to_path_buf(),
            }
        );
    }

    #[test]
    fn explicit_python_keeps_direct_interpreter_execution() {
        let runner = translator_runner("/custom/python", Path::new("script/trans.py"), true);
        assert_eq!(
            runner,
            TranslatorRunner::Python {
                interpreter: "/custom/python".to_string(),
            }
        );
    }
}
