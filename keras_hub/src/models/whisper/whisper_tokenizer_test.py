import pytest

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
