import pytest

from keras_hub.src.models.bloom.bloom_causal_lm_preprocessor import (
    BloomCausalLMPreprocessor,
)
from keras_hub.src.models.bloom.bloom_tokenizer import BloomTokenizer
from keras_hub.src.tests.test_case import TestCase


class BloomCausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.vocab = ["<pad>", "<s>", "</s>"]
        self.vocab += ["!", "air", "Ġair", "plane", "Ġat", "port"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.tokenizer = BloomTokenizer(
            vocabulary=self.vocab,
            merges=self.merges,
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = ["airplane at airport"]

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=BloomCausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[1, 4, 6, 7, 5, 8, 2, 0]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 1, 0]],
                },
                [[4, 6, 7, 5, 8, 2, 0, 0]],  # Pass through labels.
                [[1, 1, 1, 1, 1, 1, 0, 0]],  # Pass through sample_weights.
            ),
        )

    def test_no_start_end_token(self):
        input_data = ["airplane at airport"] * 4

        preprocessor = BloomCausalLMPreprocessor(
            **self.init_kwargs,
            add_start_token=False,
            add_end_token=False,
        )
        x, y, sw = preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[4, 6, 7, 5, 8, 0, 0, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 0, 0, 0]] * 4)
        self.assertAllEqual(y, [[6, 7, 5, 8, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 1, 0, 0, 0, 0]] * 4)

    def test_generate_preprocess(self):
        input_data = "airplane at airport"
        preprocessor = BloomCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(x["token_ids"], [1, 4, 6, 7, 5, 8, 0, 0])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 1, 0, 0])

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": [1, 4, 6, 7, 5, 8, 0, 0],
            "padding_mask": [1, 1, 1, 1, 1, 1, 0, 0],
        }
        preprocessor = BloomCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, "airplane at airport")

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in BloomCausalLMPreprocessor.presets:
            self.run_preset_test(
                cls=BloomCausalLMPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
