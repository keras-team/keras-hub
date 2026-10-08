import pytest

from keras_hub.src.models.phi4.phi4_causal_lm_preprocessor import (
    Phi4CausalLMPreprocessor,
)
from keras_hub.src.models.phi4.phi4_tokenizer import Phi4Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Phi4CausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.vocab = ["!", "air", "Ġair", "plane", "Ġat", "port"]
        self.vocab += [
            "<s>",
            "</s>",
            "<pad>",
            "<im_start>",
            "<im_sep>",
            "<im_end>",
        ]
        self.vocab += ["<fim_prefix>", "<fim_middle>", "<fim_suffix>"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.tokenizer = Phi4Tokenizer(
            vocabulary=self.vocab, merges=self.merges
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 10,
        }
        # [1, 3, 4, 2, 5]
        self.input_data = (["airplane at airport"],)

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=Phi4CausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[6, 1, 3, 4, 2, 5, 0, 0, 0, 0]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                },
                [[1, 3, 4, 2, 5, 0, 0, 0, 0, 7]],
                [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
            ),
        )

    def test_no_start_end_token(self):
        input_data = ["airplane at airport"] * 4

        preprocessor = Phi4CausalLMPreprocessor(
            **self.init_kwargs,
            add_start_token=False,
            add_end_token=False,
        )
        x, y, sw = preprocessor(input_data)
        self.assertAllEqual(
            x["token_ids"], [[1, 3, 4, 2, 5, 0, 0, 0, 0, 0]] * 4
        )
        self.assertAllEqual(
            x["padding_mask"], [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] * 4
        )
        self.assertAllEqual(y, [[3, 4, 2, 5, 0, 0, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]] * 4)

    def test_generate_preprocess(self):
        input_data = "airplane at airport"
        preprocessor = Phi4CausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(x["token_ids"], [6, 1, 3, 4, 2, 5, 0, 0, 0, 0])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": [1, 3, 4, 2, 5, 3, 9, 7, 11, 0],
            "padding_mask": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        }
        preprocessor = Phi4CausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, "airplane at airport")

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Phi4CausalLMPreprocessor.presets:
            self.run_preset_test(
                cls=Phi4CausalLMPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
