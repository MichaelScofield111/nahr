#!/usr/bin/env python3
"""
Translate an existing SRT file into Simplified Chinese.

Supports auto-detecting English or Japanese source subtitles.

Examples:
  python translate_srt.py input.en.srt
  python translate_srt.py input.ja.srt
  python translate_srt.py input.srt -o output.cn.srt
"""

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Subtitle:
    index: int
    start: str
    end: str
    content: str


def parse_srt(text: str) -> List[Subtitle]:
    subtitles = []
    blocks = text.strip().split("\n\n")

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue

        index = int(lines[0].strip())
        timing = lines[1].strip()
        start, end = [part.strip() for part in timing.split("-->")]
        content = "\n".join(lines[2:]).strip()
        subtitles.append(Subtitle(index=index, start=start, end=end, content=content))

    return subtitles


def compose_srt(subtitles: List[Subtitle]) -> str:
    blocks = []
    for sub in subtitles:
        blocks.append(f"{sub.index}\n{sub.start} --> {sub.end}\n{sub.content}")
    return "\n\n".join(blocks) + "\n"


def contains_japanese_kana(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u30ff]", text))


def count_english_letters(text: str) -> int:
    return len(re.findall(r"[A-Za-z]", text))


def detect_source_language(subtitles: List[Subtitle]) -> str:
    sample_text = "\n".join(sub.content for sub in subtitles[:100])

    if contains_japanese_kana(sample_text):
        return "jpn_Jpan"

    english_letters = count_english_letters(sample_text)
    non_space_chars = len(re.findall(r"\S", sample_text))

    if non_space_chars == 0:
        return "eng_Latn"

    if english_letters / non_space_chars >= 0.3:
        return "eng_Latn"

    return "jpn_Jpan"


def proxy_env_summary() -> str:
    keys = [
        "all_proxy",
        "ALL_PROXY",
        "http_proxy",
        "HTTP_PROXY",
        "https_proxy",
        "HTTPS_PROXY",
    ]
    present = [key for key in keys if os.environ.get(key)]
    return ", ".join(present)


class SRTTranslator:
    def __init__(self, model_name="facebook/nllb-200-distilled-600M"):
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Missing Python dependencies. If you use uv, run `uv sync --project script`. Otherwise install them with `pip install torch transformers srt`."
            ) from exc

        self.torch = torch
        self.AutoTokenizer = AutoTokenizer
        self.AutoModelForSeq2SeqLM = AutoModelForSeq2SeqLM

        if self.torch.backends.mps.is_available():
            self.device = "mps"
            print("Using Apple Silicon GPU (Metal)")
        elif self.torch.cuda.is_available():
            self.device = "cuda"
            print("Using NVIDIA GPU (CUDA)")
        else:
            self.device = "cpu"
            print("Using CPU")

        print(f"Loading translation model on {self.device}...")
        try:
            self.tokenizer = self.AutoTokenizer.from_pretrained(
                model_name,
                src_lang="eng_Latn",
                local_files_only=True,
            )
            self.model = self.AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                local_files_only=True,
            ).to(self.device)
        except (OSError, ValueError):
            print("Model not found in local cache, downloading...")
            try:
                self.tokenizer = self.AutoTokenizer.from_pretrained(
                    model_name,
                    src_lang="eng_Latn",
                )
                self.model = self.AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
                    self.device
                )
            except ImportError as exc:
                if "socksio" in str(exc).lower():
                    proxies = proxy_env_summary() or "proxy env vars"
                    raise RuntimeError(
                        f"Downloading the translation model requires SOCKS proxy support because {proxies} is set. Run `uv sync --project script` again after adding proxy dependencies, or install `httpx[socks]` / `socksio`."
                    ) from exc
                raise
            except OSError as exc:
                raise RuntimeError(
                    f"Failed to download or load translation model '{model_name}'. Check your network/proxy setup and whether Hugging Face is reachable."
                ) from exc

    def translate_text(self, text, src_lang="eng_Latn", tgt_lang="zho_Hans"):
        text = text.strip()
        if not text:
            return text

        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        output_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang),
            max_length=512,
        )
        return self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

    def translate_subtitles(self, subtitles, src_lang="eng_Latn", tgt_lang="zho_Hans"):
        translated = []
        total = len(subtitles)

        for i, sub in enumerate(subtitles, start=1):
            print(f"Translating {i}/{total}...", end="\r")
            translated_text = self.translate_text(sub.content, src_lang, tgt_lang)
            translated.append(
                Subtitle(
                    index=sub.index,
                    start=sub.start,
                    end=sub.end,
                    content=translated_text,
                )
            )

        print(f"Translated {total}/{total}")
        return translated


def build_output_path(input_path: Path, output_path: Optional[str]) -> Path:
    if output_path:
        return Path(output_path)

    if input_path.name.endswith(".en.srt"):
        return input_path.with_name(input_path.name[:-7] + ".cn.srt")

    return input_path.with_name(input_path.stem + ".cn.srt")


def main():
    parser = argparse.ArgumentParser(
        description="Translate an SRT file into Simplified Chinese."
    )
    parser.add_argument("input_srt", help="Path to the input .srt file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output path for the translated Chinese .srt file",
    )
    parser.add_argument(
        "--src-lang",
        default="auto",
        help="Source language code for NLLB, or 'auto' to detect English/Japanese (default: auto)",
    )
    parser.add_argument(
        "--tgt-lang",
        default="zho_Hans",
        help="Target language code for NLLB (default: zho_Hans)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_srt)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = build_output_path(input_path, args.output)

    with input_path.open("r", encoding="utf-8") as f:
        subtitles = parse_srt(f.read())

    src_lang = args.src_lang
    if src_lang == "auto":
        src_lang = detect_source_language(subtitles)
        print(f"Auto-detected source language: {src_lang}")

    if src_lang == "en":
        src_lang = "eng_Latn"
    elif src_lang == "ja":
        src_lang = "jpn_Jpan"

    translator = SRTTranslator()
    translated_subtitles = translator.translate_subtitles(
        subtitles,
        src_lang=src_lang,
        tgt_lang=args.tgt_lang,
    )

    with output_path.open("w", encoding="utf-8") as f:
        f.write(compose_srt(translated_subtitles))

    print(f"Saved translated subtitles to: {output_path}")


if __name__ == "__main__":
    main()
