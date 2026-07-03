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


class MistralTokenizerTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            # Generated using create_mistral_test_proto.py
            "proto": os.path.join(
                self.get_test_data_dir(), "mistral_test_vocab.spm"
            )
        }
        self.input_data = ["the quick brown fox", "the earth is round"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=MistralTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[3, 8, 4, 6], [3, 5, 7, 9]],
        )

    def test_tekken_tokenizer_basics(self):
        pytest.importorskip("mistral_common")
        # Magistral-style Tekken (byte-level BPE) vocabulary.
        vocab = [
            {
                "rank": i,
                "token_bytes": base64.b64encode(bytes([i])).decode(),
                "token_str": None,
            }
            for i in range(256)
        ]
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
        tekken_config = {
            "config": {
                "pattern": (
                    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
                    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|\p{N}| ?[^\s\p{L}\p{N}]+"
                    r"[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
                ),
                "num_vocab_tokens": 261,
                "default_vocab_size": 266,
                "default_num_special_tokens": 5,
                "version": "v7",
            },
            "vocab": vocab,
            "special_tokens": [
                {"rank": 0, "token_str": "<unk>", "is_control": True},
                {"rank": 1, "token_str": "<s>", "is_control": True},
                {"rank": 2, "token_str": "</s>", "is_control": True},
                {"rank": 3, "token_str": "<pad>", "is_control": True},
                {"rank": 4, "token_str": "[INST]", "is_control": True},
            ],
        }
        with tempfile.TemporaryDirectory() as dir_path:
            path = os.path.join(dir_path, "tekken.json")
            with open(path, "w") as f:
                json.dump(tekken_config, f)
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

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            MistralTokenizer(
                # Generated using create_no_special_token_proto.py
                proto=os.path.join(
                    self.get_test_data_dir(), "no_special_token_vocab.spm"
                )
            )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=MistralTokenizer,
            preset="mistral_7b_en",
            input_data=["The quick brown fox."],
            expected_output=[[415, 2936, 9060, 285, 1142, 28723]],
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in MistralTokenizer.presets:
            self.run_preset_test(
                cls=MistralTokenizer,
                preset=preset,
                input_data=self.input_data,
            )
