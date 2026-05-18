# Script Purpose

`convert_tok.py` converts the **slow Marian tokenizer** from Hugging Face into two **fast tokenizer JSON** files (one for source language and one for target language).

## What it does

1. Loads the slow tokenizer from `shun89/opus-mt-ja-zh` (`use_fast=False`).
2. Uses a custom `MarianConverter` (based on `SpmConverter`) to:
   - read each SentencePiece model file (`spm_files[0]` and `spm_files[1]`),
   - load `vocab.json` from the same directory,
   - rebuild tokenizer vocabulary order using `vocab.json` indices,
   - warn if SentencePiece byte-fallback is enabled (because fast tokenizer behavior may differ).
3. Exports two fast tokenizer files:
   - `tokenizer-marian-base-ja-zh-src.json` (source side, `index=0`)
   - `tokenizer-marian-base-ja-zh-tgt.json` (target side, `index=1`)

## In short

This script is a conversion utility that creates reusable fast-tokenizer artifacts from a Marian MT tokenizer that is originally provided in slow/SentencePiece form.
