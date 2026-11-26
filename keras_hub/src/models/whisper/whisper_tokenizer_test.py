import json
import os

import keras
import pytest
from keras.src.saving import serialization_lib

from keras_hub.src.models.whisper.whisper_tokenizer import WhisperTokenizer
from keras_hub.src.tests.test_case import TestCase


class WhisperTokenizerTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab += ["!", "<|endoftext|>"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.special_tokens = {
            "<|startoftranscript|>": 31,  # len(self.vocab) == 31 at this point
            "<|endoftext|>": 32,
            "<|notimestamps|>": 33,
            "<|transcribe|>": 34,
            "<|translate|>": 35,
        }
        self.language_tokens = {
            "<|en|>": 36,
            "<|fr|>": 37,
        }
        self.init_kwargs = {
            "vocabulary": self.vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
            "language_tokens": self.language_tokens,
        }
        self.input_data = [
            " airplane at airport<|endoftext|>",
            " airplane airport",
        ]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=WhisperTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[
                [23, 14, 24, 23, 16, 32],
                [23, 14, 23, 16],
            ],
            expected_detokenize_output=[
                " airplane at airport<|endoftext|>",
                " airplane airport",
            ],
        )

    def test_special_tokens(self):
        tokenizer = WhisperTokenizer(**self.init_kwargs)
        self.assertEqual(tokenizer.bos_token_id, 31)
        self.assertEqual(tokenizer.eos_token_id, 32)
        self.assertEqual(tokenizer.pad_token_id, 32)
        self.assertEqual(tokenizer.no_timestamps_token_id, 33)
        self.assertEqual(tokenizer.transcribe_token_id, 34)
        self.assertEqual(tokenizer.translate_token_id, 35)

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            WhisperTokenizer(
                vocabulary=["a", "b", "c"], merges=[], special_tokens={}
            )

    def test_safe_mode_file_paths_disallowed(self):
        temp_dir = self.get_temp_dir()
        vocab_path = os.path.join(temp_dir, "vocab.json")
        merges_path = os.path.join(temp_dir, "merges.txt")
        special_tokens_path = os.path.join(temp_dir, "special_tokens.json")
        language_tokens_path = os.path.join(temp_dir, "language_tokens.json")

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f)
        with open(merges_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.merges))
        with open(special_tokens_path, "w", encoding="utf-8") as f:
            json.dump(self.special_tokens, f)
        with open(language_tokens_path, "w", encoding="utf-8") as f:
            json.dump(self.language_tokens, f)

        with serialization_lib.SafeModeScope(True):
            with self.assertRaisesRegex(
                ValueError,
                r"Requested the loading of a tokenizer asset file outside of "
                r"the model archive.*Asset file: .*special_tokens\.json",
            ):
                WhisperTokenizer(
                    vocabulary=self.vocab,
                    merges=self.merges,
                    special_tokens=special_tokens_path,
                )

            with self.assertRaisesRegex(
                ValueError,
                r"Requested the loading of a tokenizer asset file outside of "
                r"the model archive.*Asset file: .*language_tokens\.json",
            ):
                WhisperTokenizer(
                    vocabulary=self.vocab,
                    merges=self.merges,
                    special_tokens=self.special_tokens,
                    language_tokens=language_tokens_path,
                )

            tokenizer = WhisperTokenizer(**self.init_kwargs)
            with self.assertRaisesRegex(
                ValueError,
                r"Requested the loading of a tokenizer asset file outside of "
                r"the model archive.*Asset file: .*vocab\.json",
            ):
                tokenizer.set_vocabulary_and_merges(vocab_path, merges_path)

        serialized = keras.saving.serialize_keras_object(
            WhisperTokenizer(**self.init_kwargs)
        )
        serialized_cases = [
            (
                {
                    **serialized,
                    "config": {
                        **serialized["config"],
                        "language_tokens": language_tokens_path,
                    },
                },
                r"Requested the loading of a tokenizer asset file outside of "
                r"the model archive.*Asset file: .*language_tokens\.json",
            ),
            (
                {
                    **serialized,
                    "config": {
                        **serialized["config"],
                        "special_tokens": special_tokens_path,
                    },
                },
                r"Requested the loading of a tokenizer asset file outside of "
                r"the model archive.*Asset file: .*special_tokens\.json",
            ),
        ]

        for serialized_config, error_pattern in serialized_cases:
            with self.assertRaisesRegex(TypeError, error_pattern):
                keras.saving.deserialize_keras_object(serialized_config)

        # Unscoped construction from file paths should remain allowed.
        tokenizer = WhisperTokenizer(
            vocabulary=vocab_path,
            merges=merges_path,
            special_tokens=special_tokens_path,
            language_tokens=language_tokens_path,
        )
        self.assertAllEqual(
            tokenizer(self.input_data),
            [[23, 14, 24, 23, 16, 32], [23, 14, 23, 16]],
        )

        with serialization_lib.SafeModeScope(False):
            tokenizer = WhisperTokenizer(
                vocabulary=vocab_path,
                merges=merges_path,
                special_tokens=special_tokens_path,
                language_tokens=language_tokens_path,
            )
            self.assertAllEqual(
                tokenizer(self.input_data),
                [[23, 14, 24, 23, 16, 32], [23, 14, 23, 16]],
            )

        for serialized_config, _ in serialized_cases:
            restored = keras.saving.deserialize_keras_object(
                serialized_config, safe_mode=False
            )
            self.assertEqual(restored.language_tokens, self.language_tokens)
            self.assertEqual(restored._special_token_dict, self.special_tokens)

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=WhisperTokenizer,
            preset="whisper_tiny_en",
            input_data=["The quick brown fox."],
            expected_output=[[464, 2068, 7586, 21831, 13]],
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in WhisperTokenizer.presets:
            self.run_preset_test(
                cls=WhisperTokenizer,
                preset=preset,
                input_data=self.input_data,
            )
