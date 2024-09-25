import os

import pytest

from keras_hub.src.models.gemma.gemma_causal_lm_preprocessor import (
    GemmaCausalLMPreprocessor,
)
from keras_hub.src.models.gemma.gemma_tokenizer import GemmaTokenizer
from keras_hub.src.tests.test_case import TestCase


class GemmaCausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = GemmaTokenizer(
            proto=os.path.join(
                self.get_test_data_dir(), "gemma_test_vocab.spm"
            ),
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = ["the quick brown fox"]

    def test_preprocessor_basics(self):
        self.run_preprocessing_layer_test(
            cls=GemmaCausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[1, 4, 9, 5, 7, 2, 0, 0]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0]],
                },
                [[4, 9, 5, 7, 2, 0, 0, 0]],  # Labels shifted.
                [[1, 1, 1, 1, 1, 0, 0, 0]],  # Zero out unlabeled examples.
            ),
        )

    def test_no_start_end_token(self):
        input_data = ["the quick brown fox"] * 4

        preprocessor = GemmaCausalLMPreprocessor(
            **self.init_kwargs,
            add_start_token=False,
            add_end_token=False,
        )
        x, y, sw = preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[4, 9, 5, 7, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(y, [[9, 5, 7, 0, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 0, 0, 0, 0, 0]] * 4)

    def test_generate_preprocess(self):
        input_data = "the quick brown fox"
        preprocessor = GemmaCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(x["token_ids"], [1, 4, 9, 5, 7, 0, 0, 0])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 0, 0, 0])

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": [1, 4, 9, 5, 7, 2, 0, 0],
            "padding_mask": [1, 1, 1, 1, 1, 1, 0, 0],
        }
        preprocessor = GemmaCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, "the quick brown fox")

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in GemmaCausalLMPreprocessor.presets:
            self.run_preset_test(
                cls=GemmaCausalLMPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
