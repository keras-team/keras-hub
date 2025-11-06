import numpy as np

from keras_hub.src.models.rwkv7.rwkv7_causal_lm_preprocessor import (
    RWKV7CausalLMPreprocessor,
)
from keras_hub.src.models.rwkv7.rwkv7_tokenizer import RWKVTokenizer
from keras_hub.src.tests.test_case import TestCase


class RWKV7CausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = RWKVTokenizer(
            ["1 ' ' 1", "2 '\\n' 1", "3 'the' 3", "4 'hello' 5", "5 'world' 5"]
        )
        self.preprocessor = RWKV7CausalLMPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=15,
        )

    def test_preprocessor_basics(self):
        result = self.preprocessor(x=["hello world hello world hello world"])
        self.assertAllEqual(
            result[0]["token_ids"],
            [[0, 0, 0, 0, 0, 0, 4, 1, 5, 1, 4, 1, 5, 1, 4, 1]],
        )
        self.assertAllEqual(
            result[1], [[0, 0, 0, 0, 0, 4, 1, 5, 1, 4, 1, 5, 1, 4, 1, 5]]
        )
        self.assertAllEqual(
            result[2],
            [
                [
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ]
            ],
        )

    def test_generate_preprocess(self):
        result = self.preprocessor.generate_preprocess(
            ["hello world hello world hello world"], 16
        )

        self.assertAllEqual(
            result["token_ids"],
            [[0, 0, 0, 0, 0, 0, 4, 1, 5, 1, 4, 1, 5, 1, 4, 1]],
        )
        self.assertAllEqual(
            result["input_padding_mask"],
            [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        )
        self.assertAllEqual(
            result["padding_mask"],
            [
                [
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                ]
            ],
        )
        self.assertAllEqual(
            result["predict_token_ids"],
            [[5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        )

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": np.array(
                [[3, 2, 4, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
            ),
            "padding_mask": np.array(
                [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
            ),
        }
        result = self.preprocessor.generate_postprocess(input_data)
        self.assertEqual(result, ["the\nhellothe"])
