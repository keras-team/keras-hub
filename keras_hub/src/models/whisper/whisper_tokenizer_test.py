import pytest

from keras_hub.src.models.whisper.whisper_tokenizer import WhisperTokenizer
from keras_hub.src.tests.test_case import TestCase


class WhisperTokenizerTest(TestCase):
    def setUp(self):
        self.vocab = ["!", "air", "Ġair", "plane", "Ġat", "port"]
        self.vocab += ["<|endoftext|>"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.special_tokens = {
            "<|startoftranscript|>": 9,
            "<|endoftext|>": 10,
            "<|notimestamps|>": 11,
            "<|transcribe|>": 12,
            "<|translate|>": 13,
        }
        self.language_tokens = {
            "<|en|>": 14,
            "<|fr|>": 15,
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
            expected_output=[[2, 3, 4, 2, 5, 10], [2, 3, 2, 5]],
        )

    def test_special_tokens(self):
        tokenizer = WhisperTokenizer(**self.init_kwargs)
        self.assertEqual(tokenizer.bos_token_id, 9)
        self.assertEqual(tokenizer.eos_token_id, 10)
        self.assertEqual(tokenizer.pad_token_id, 10)
        self.assertEqual(tokenizer.no_timestamps_token_id, 11)
        self.assertEqual(tokenizer.translate_token_id, 13)
        self.assertEqual(tokenizer.transcribe_token_id, 12)

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            WhisperTokenizer(
                vocabulary=["a", "b", "c"], merges=[], special_tokens={}
            )

    @pytest.mark.large
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
