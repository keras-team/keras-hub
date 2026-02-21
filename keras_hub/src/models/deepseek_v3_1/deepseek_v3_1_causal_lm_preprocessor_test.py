import os
import pytest

from keras_hub.src.models.deepseek_v3_1.deepseek_v3_1_causal_lm_preprocessor import (
    DeepSeekV3_1CausalLMPreprocessor,
)
from keras_hub.src.models.deepseek_v3_1.deepseek_v3_1_tokenizer import (
    DeepSeekV3_1Tokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class DeepSeekV3_1CausalLMPreprocessorTest(TestCase):
    def setUp(self):
        # FIX: The BytePairTokenizer StaticHashTable rejects duplicate keys.
        # "Ġ" maps to 6, and " " maps to 7 to maintain a valid 1:1 mapping.
        self.vocab = {
            "<｜begin▁of▁sentence｜>": 151646,
            "<｜end▁of▁sentence｜>": 151643,
            "a": 2,
            "b": 3,
            "c": 4,
            "d": 5,
            "Ġ": 6,
            " ": 7,
        }
        self.merges = []
        self.tokenizer = DeepSeekV3_1Tokenizer(
            vocabulary=self.vocab, merges=self.merges
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = (["a b"],)

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=DeepSeekV3_1CausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[151646, 2, 6, 3, 151643, 0, 0, 0]],
                    "padding_mask": [[1, 1, 1, 1, 1, 0, 0, 0]],
                },
                [[2, 6, 3, 151643, 0, 0, 0, 0]],  # Pass through labels.
                [[1, 1, 1, 1, 0, 0, 0, 0]],  # Pass through sample_weights.
            ),
        )

    def test_no_start_end_token(self):
        input_data = ["a b"] * 4

        preprocessor = DeepSeekV3_1CausalLMPreprocessor(
            **self.init_kwargs,
            add_start_token=False,
            add_end_token=False,
        )
        x, y, sw = preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[2, 6, 3, 0, 0, 0, 0, 0]] * 4)

    def test_generate_preprocess(self):
        preprocessor = DeepSeekV3_1CausalLMPreprocessor(**self.init_kwargs)
        preprocessed = preprocessor.generate_preprocess(["a b"])
        self.assertIn("token_ids", preprocessed)
        self.assertIn("padding_mask", preprocessed)

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=DeepSeekV3_1CausalLMPreprocessor,
            preset="deepseek_v3_1_base",
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in DeepSeekV3_1CausalLMPreprocessor.presets:
            self.run_preset_test(
                cls=DeepSeekV3_1CausalLMPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
