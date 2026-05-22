import json
import warnings
from pathlib import Path

from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.models import Unigram
from tokenizers.pre_tokenizers import Metaspace
from transformers import AutoTokenizer
from transformers.convert_slow_tokenizer import import_protobuf, requires_backends


def build_marian_fast_tokenizer(tokenizer, index: int = 0):
    """
    手动从 spm 文件 + vocab.json 构建 HuggingFace fast tokenizer，
    绕过新版 transformers SpmConverter 对 vocab_file 属性的依赖。
    """
    requires_backends(tokenizer, "protobuf")

    model_pb2 = import_protobuf()
    m = model_pb2.ModelProto()
    spm_path = tokenizer.spm_files[index]
    with open(spm_path, "rb") as f:
        m.ParseFromString(f.read())

    dir_path = Path(tokenizer.spm_files[0]).parent
    with open(dir_path / "vocab.json", "r") as f:
        vocab_map = json.load(f)  # piece -> id

    # 构建 vocab 列表：[(piece, score), ...]，按 id 排序
    vocab_size = max(vocab_map.values()) + 1
    vocab = [("<NIL>", -100.0)] * vocab_size

    for piece in m.pieces:
        idx = vocab_map.get(piece.piece)
        if idx is None:
            print(f"Skipped missing piece: {piece.piece}")
            continue
        vocab[idx] = (piece.piece, piece.score)

    # 用 Unigram 模型构建 tokenizer
    unk_id = vocab_map.get("<unk>", 1)
    fast_tok = Tokenizer(Unigram(vocab, unk_id))

    # Marian 用 Metaspace（即 sentencepiece 的 ▁ 前缀）做 pre_tokenizer / decoder
    fast_tok.pre_tokenizer = Metaspace(prepend_scheme="first")
    fast_tok.decoder = MetaspaceDecoder(prepend_scheme="first")

    # 添加特殊 token
    special_tokens = []
    for tok_str, tok_id in [("</s>", 0), ("<unk>", unk_id), ("<pad>", 65000)]:
        if tok_str in vocab_map:
            special_tokens.append(AddedToken(tok_str, special=True))
    fast_tok.add_special_tokens(special_tokens)

    return fast_tok


# 加载慢分词器
tokenizer = AutoTokenizer.from_pretrained("shun89/opus-mt-ja-zh", use_fast=False)

# 导出 source 侧（日语，index=0）
fast_tokenizer = build_marian_fast_tokenizer(tokenizer, index=0)
fast_tokenizer.save("tokenizer-marian-base-ja-zh-ja.json")
print("Saved ja tokenizer")

# 导出 target 侧（中文，index=1）
fast_tokenizer = build_marian_fast_tokenizer(tokenizer, index=1)
fast_tokenizer.save("tokenizer-marian-base-ja-zh-zh.json")
print("Saved zh tokenizer")
