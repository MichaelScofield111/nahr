use crate::{Args, bake::burn_subtitles, cn_srt, srt, wav};
use anyhow::Result;

pub fn subtitle_pipeline(args: &Args) -> Result<()> {
    // split mp4 to wav
    let wav_path = args.input_file.with_extension("wav");
    wav::extract_wav(&args.input_file, &wav_path)?;

    // need to select which base video
    let source_srt_path = srt::wav_to_srt(&wav_path, &args.whisper_model_path, &args.language)?;
    let cn_srt_path = cn_srt::gen_cnsrt(source_srt_path, &args.language)?;

    burn_subtitles(&args.input_file, &cn_srt_path)?;
    Ok(())
}
