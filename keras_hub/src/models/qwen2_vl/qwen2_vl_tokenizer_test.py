import pytest

from keras_hub.src.models.qwen2_vl.qwen2_vl_tokenizer import (
    Qwen2VLTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen2VLTokenizerTest(TestCase):
    def setUp(self):
        self.vocab = ["!", "air", "\u0120air", "plane", "\u0120at", "port"]
        self.vocab += ["<|endoftext|>"]
        self.vocab += ["<|eot_id|>"]
        self.vocab += ["<|vision_start|>"]
        self.vocab += ["<|vision_end|>"]
        self.vocab += ["<|image_pad|>"]
        self.vocab = dict(
            [(token, i) for i, token in enumerate(self.vocab)]
        )
        self.merges = [
            "\u0120 a", "\u0120 t", "\u0120 i", "\u0120 b",
            "a i", "p l", "n e",
        ]
        self.merges += [
            "\u0120a t", "p o", "r t", "\u0120t h", "ai r",
            "pl a", "po rt",
        ]
        self.merges += ["\u0120ai r", "\u0120a i", "pla ne"]
        self.init_kwargs = {
            "vocabulary": self.vocab,
            "merges": self.merges,
        }
        self.input_data = [
            " airplane at airport<|endoftext|>",
            " airplane airport",
        ]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=Qwen2VLTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[2, 3, 4, 2, 5, 6], [2, 3, 2, 5]],
        )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            Qwen2VLTokenizer(vocabulary=["a", "b", "c"], merges=[])

    def test_vision_special_tokens(self):
        """Verify vision special tokens are registered."""
        tokenizer = Qwen2VLTokenizer(**self.init_kwargs)
        self.assertIsNotNone(tokenizer.vision_start_token_id)
        self.assertIsNotNone(tokenizer.vision_end_token_id)
        self.assertIsNotNone(tokenizer.image_pad_token_id)
        # Check they map to the right vocabulary ids.
        self.assertEqual(tokenizer.vision_start_token_id, 8)
        self.assertEqual(tokenizer.vision_end_token_id, 9)
        self.assertEqual(tokenizer.image_pad_token_id, 10)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Qwen2VLTokenizer.presets:
            self.run_preset_test(
                cls=Qwen2VLTokenizer,
                preset=preset,
                input_data=self.input_data,
            )
