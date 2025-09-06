import os

import pytest

from keras_hub.src.models.gpt_oss.gpt_oss_causal_lm_preprocessor import (
    GptOssCausalLMPreprocessor,
)
from keras_hub.src.models.gpt_oss.gpt_oss_tokenizer import GptOssTokenizer
from keras_hub.src.tests.test_case import TestCase


class GptOssCausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = GptOssTokenizer(
            # Generated using create_gpt_oss_test_proto.py (hypothetical script)
            proto=os.path.join(
                self.get_test_data_dir(), "gpt_oss_test_vocab.spm"
            )
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = (["the quick brown fox"],)

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=GptOssCausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [
                        [1, 3, 8, 4, 6, 2, 0, 0]
                    ],  # Start, the, quick, brown, fox, end, pad, pad
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0]],
                },
                [
                    [3, 8, 4, 6, 2, 0, 0, 0]
                ],  # Labels: the, quick, brown, fox, end, pad, pad, pad (shifted)
                [[1, 1, 1, 1, 1, 0, 0, 0]],  # Sample weights for labels
            ),
        )

    def test_no_start_end_token(self):
        input_data = ["the quick brown fox"] * 4

        preprocessor = GptOssCausalLMPreprocessor(
            **self.init_kwargs,
            add_start_token=False,
            add_end_token=False,
        )
        x, y, sw = preprocessor(input_data)
        # No start/end tokens, just the content and padding
        self.assertAllEqual(x["token_ids"], [[3, 8, 4, 6, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 0, 0, 0, 0]] * 4)
        # Labels shifted, no start token to predict
        self.assertAllEqual(y, [[8, 4, 6, 0, 0, 0, 0, 0]] * 4)
        # Sample weights for labels
        self.assertAllEqual(sw, [[1, 1, 1, 0, 0, 0, 0, 0]] * 4)

    def test_generate_preprocess(self):
        input_data = "the quick brown fox"
        preprocessor = GptOssCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        # Generate preprocess adds start token, but not end token, and pads
        self.assertAllEqual(x["token_ids"], [1, 3, 8, 4, 6, 0, 0, 0])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 0, 0, 0])

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": [1, 3, 8, 4, 6, 0, 0, 0],
            "padding_mask": [1, 1, 1, 1, 1, 0, 0, 0],
        }
        preprocessor = GptOssCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        # Postprocess should decode the tokens back to the original string
        self.assertAllEqual(x, "the quick brown fox")

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in GptOssCausalLMPreprocessor.presets:
            self.run_preset_test(
                cls=GptOssCausalLMPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
