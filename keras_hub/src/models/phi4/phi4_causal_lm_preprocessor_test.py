import os

import pytest

from keras_hub.src.models.phi4.phi4_causal_lm_preprocessor import (
    Phi4CausalLMPreprocessor,
)
from keras_hub.src.models.phi4.phi4_tokenizer import Phi4Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Phi4CausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = Phi4Tokenizer(
            # Generated using create_phi4_test_proto.py
            proto=os.path.join(self.get_test_data_dir(), "phi4_test_vocab.spm")
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 10,
        }
        # [3, 5, 6, 4, 3, 9, 7, 11]
        self.input_data = (["the fox"],)

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=Phi4CausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[1, 3, 5, 6, 4, 3, 9, 7, 11, 15]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                },
                [[3, 5, 6, 4, 3, 9, 7, 11, 15, 0]],
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 0]],
            ),
        )

    def test_no_start_end_token(self):
        input_data = ["the fox"] * 4

        preprocessor = Phi4CausalLMPreprocessor(
            **self.init_kwargs,
            add_start_token=False,
            add_end_token=False,
        )
        x, y, sw = preprocessor(input_data)
        self.assertAllEqual(
            x["token_ids"], [[3, 5, 6, 4, 3, 9, 7, 11, 0, 0]] * 4
        )
        self.assertAllEqual(
            x["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 4
        )
        self.assertAllEqual(y, [[5, 6, 4, 3, 9, 7, 11, 0, 0, 0]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]] * 4)

    def test_generate_preprocess(self):
        input_data = "the fox"
        preprocessor = Phi4CausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(x["token_ids"], [1, 3, 5, 6, 4, 3, 9, 7, 11, 0])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 0])

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": [1, 3, 5, 6, 4, 3, 9, 7, 11, 0],
            "padding_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        }
        preprocessor = Phi4CausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, "the fox")

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Phi4CausalLMPreprocessor.presets:
            self.run_preset_test(
                cls=Phi4CausalLMPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
