import warnings
from pathlib import Path

from transformers import AutoTokenizer
from transformers.convert_slow_tokenizer import (
    SpmConverter,
    import_protobuf,
    requires_backends,
)


class MarianConverter(SpmConverter):
    def __init__(self, *args, index: int = 0):
        requires_backends(self, "protobuf")

        super(SpmConverter, self).__init__(*args)

        # from .utils import sentencepiece_model_pb2 as model_pb2
        model_pb2 = import_protobuf()

        m = model_pb2.ModelProto()
        print(self.original_tokenizer.spm_files)
        with open(self.original_tokenizer.spm_files[index], "rb") as f:
            m.ParseFromString(f.read())
        self.proto = m
        print(self.original_tokenizer)
        # with open(self.original_tokenizer.vocab_path, "r") as f:
        dir_path = Path(self.original_tokenizer.spm_files[0]).parents[0]
        with open(dir_path / "vocab.json", "r") as f:
            import json

            self._vocab = json.load(f)

        if self.proto.trainer_spec.byte_fallback:
            if not getattr(self, "handle_byte_fallback", None):
                warnings.warn(
                    "The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option"
                    " which is not implemented in the fast tokenizers. In practice this means that the fast version of the"
                    " tokenizer can produce unknown tokens whereas the sentencepiece version would have converted these "
                    "unknown tokens into a sequence of byte tokens matching the original piece of text."
                )

    def vocab(self, proto):
        vocab_size = max(self._vocab.values()) + 1
        vocab = [("<NIL>", -100) for _ in range(vocab_size)]
        for piece in proto.pieces:
            try:
                index = self._vocab[piece.piece]
            except Exception:
                print(f"Ignored missing piece {piece.piece}")
            vocab[index] = (piece.piece, piece.score)
        return vocab


# 1) 加载慢分词器
tokenizer = AutoTokenizer.from_pretrained("shun89/opus-mt-ja-zh", use_fast=False)

# 2) 导出 index=0（通常是 source 侧）
fast_tokenizer = MarianConverter(tokenizer, index=0).converted()
fast_tokenizer.save("tokenizer-marian-base-ja-zh-src.json")

# 3) 导出 index=1（通常是 target 侧）
fast_tokenizer = MarianConverter(tokenizer, index=1).converted()
fast_tokenizer.save("tokenizer-marian-base-ja-zh-tgt.json")
