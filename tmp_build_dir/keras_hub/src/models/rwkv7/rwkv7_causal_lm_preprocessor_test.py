import numpy as np

from keras_hub.src.models.rwkv7.rwkv7_causal_lm_preprocessor import (
    RWKV7CausalLMPreprocessor,
)
from keras_hub.src.models.rwkv7.rwkv7_tokenizer import RWKVTokenizer
from keras_hub.src.tests.test_case import TestCase


class RWKV7CausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.vocab = [
            "1 ' ' 1",
            "2 '\\n' 1",
            "3 'the' 3",
            "4 'hello' 5",
            "5 'world' 5",
            "6 'python' 6",
            "7 'code' 4",
            "8 'def' 3",
            "9 'function' 8",
            "10 'return' 6",
        ]
        self.tokenizer = RWKVTokenizer(vocabulary=self.vocab)
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 16,
        }
        self.input_data = ["the python code"]

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=RWKV7CausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 6, 1]
                    ],
                    "padding_mask": [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
                    ],
                },
                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 6, 1, 7]],
                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]],
            ),
        )

    def test_generate_preprocess(self):
        preprocessor = RWKV7CausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(
            self.input_data, sequence_length=14
        )
        self.assertAllEqual(
            x["token_ids"], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 6, 1]]
        )
        self.assertAllEqual(
            x["input_padding_mask"],
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]],
        )
        self.assertAllEqual(
            x["padding_mask"], [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        )
        self.assertAllEqual(
            x["predict_token_ids"],
            [[7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        )

    def test_generate_postprocess(self):
        input_tokens = {
            "token_ids": np.array(
                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 6, 1, 7]]
            ),
            "padding_mask": np.array(
                [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
            ),
        }
        preprocessor = RWKV7CausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_tokens)
        self.assertEqual(x[0], "the python code")
