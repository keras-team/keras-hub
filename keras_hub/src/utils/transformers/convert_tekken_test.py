import base64
import json
import os
import tempfile

import pytest

from keras_hub.src.models.mistral.mistral_tokenizer import MistralTokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers.convert_tekken import (
    convert_tekken_tokenizer,
)

# A tiktoken-style split pattern, matching the Tekken format.
_SPLIT_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|\p{N}| ?[^\s\p{L}\p{N}]+"
    r"[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def _write_tekken_file(dir_path):
    """Write a tiny synthetic `tekken.json` for offline tests."""
    vocab = []
    # The 256 single bytes are always the base alphabet.
    for i in range(256):
        vocab.append(
            {
                "rank": i,
                "token_bytes": base64.b64encode(bytes([i])).decode(),
                "token_str": None,
            }
        )
    # A handful of merges, added as higher ranks.
    for rank, piece in [
        (256, b"th"),
        (257, b"the"),
        (258, b"in"),
        (259, b" t"),
        (260, b" th"),
    ]:
        vocab.append(
            {
                "rank": rank,
                "token_bytes": base64.b64encode(piece).decode(),
                "token_str": piece.decode("latin-1"),
            }
        )
    special_tokens = [
        {"rank": 0, "token_str": "<unk>", "is_control": True},
        {"rank": 1, "token_str": "<s>", "is_control": True},
        {"rank": 2, "token_str": "</s>", "is_control": True},
        {"rank": 3, "token_str": "<pad>", "is_control": True},
        {"rank": 4, "token_str": "[INST]", "is_control": True},
    ]
    config = {
        "pattern": _SPLIT_PATTERN,
        "num_vocab_tokens": 261,
        "default_vocab_size": 266,
        "default_num_special_tokens": 5,
        "version": "v7",
    }
    path = os.path.join(dir_path, "tekken.json")
    with open(path, "w") as f:
        json.dump(
            {
                "config": config,
                "vocab": vocab,
                "special_tokens": special_tokens,
            },
            f,
        )
    return path


class ConvertTekkenTest(TestCase):
    def setUp(self):
        pytest.importorskip("mistral_common")

    def test_convert_tekken_tokenizer(self):
        with tempfile.TemporaryDirectory() as dir_path:
            path = _write_tekken_file(dir_path)
            vocabulary, merges, special_tokens, split_pattern = (
                convert_tekken_tokenizer(path)
            )
        # 256 bytes + 5 merges + 5 special tokens.
        self.assertEqual(len(vocabulary), 266)
        self.assertEqual(len(merges), 5)
        self.assertEqual(
            special_tokens[:5], ["<unk>", "<s>", "</s>", "<pad>", "[INST]"]
        )
        self.assertEqual(split_pattern, _SPLIT_PATTERN)
        # Special tokens keep their reserved rank as id.
        self.assertEqual(vocabulary["<unk>"], 0)
        self.assertEqual(vocabulary["<s>"], 1)
        self.assertEqual(vocabulary["</s>"], 2)
        self.assertEqual(vocabulary["<pad>"], 3)

    def test_tokenizer_basics(self):
        with tempfile.TemporaryDirectory() as dir_path:
            path = _write_tekken_file(dir_path)
            vocabulary, merges, _, split_pattern = convert_tekken_tokenizer(
                path
            )
        self.run_preprocessing_layer_test(
            cls=MistralTokenizer,
            init_kwargs={
                "vocabulary": vocabulary,
                "merges": merges,
                "split_pattern": split_pattern,
            },
            input_data=["the tin", "in the"],
        )

    def test_tekken_special_token_ids(self):
        with tempfile.TemporaryDirectory() as dir_path:
            path = _write_tekken_file(dir_path)
            vocabulary, merges, _, split_pattern = convert_tekken_tokenizer(
                path
            )
        tokenizer = MistralTokenizer(
            vocabulary=vocabulary,
            merges=merges,
            split_pattern=split_pattern,
        )
        self.assertEqual(tokenizer.start_token_id, 1)
        self.assertEqual(tokenizer.end_token_id, 2)
        self.assertEqual(tokenizer.pad_token_id, 0)
        self.assertEqual(tokenizer.vocabulary_size(), 266)
        # Round-trip a simple string.
        output = tokenizer("the tin")
        self.assertEqual(tokenizer.detokenize(output), "the tin")
