from keras_hub.src.models.qwen3.qwen3_causal_lm_preprocessor import (
    Qwen3CausalLMPreprocessor,
)
from keras_hub.src.models.qwen3.qwen3_tokenizer import Qwen3Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Qwen3CausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["<|im_end|>", "<|endoftext|>", "!"]
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.tokenizer = Qwen3Tokenizer(
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
            cls=Qwen3CausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[5, 17, 27, 26, 19, 2, 1, 1]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0]],
                },
                [[17, 27, 26, 19, 2, 1, 1, 1]],
                [[1, 1, 1, 1, 1, 0, 0, 0]],
            ),
        )

    def test_with_start_end_token(self):
        input_data = ["airplane at airport"] * 4
        preprocessor = Qwen3CausalLMPreprocessor(
            **self.init_kwargs,
            add_start_token=True,
            add_end_token=True,
        )
        x, y, sw = preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[5, 17, 27, 26, 19, 2, 1, 1]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 1, 0, 0]] * 4)
        self.assertAllEqual(y, [[17, 27, 26, 19, 2, 1, 1, 1]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 1, 1, 0, 0, 0]] * 4)

    def test_generate_preprocess(self):
        input_data = "airplane at airport"
        preprocessor = Qwen3CausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(x["token_ids"], [5, 17, 27, 26, 19, 1, 1, 1])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 0, 0, 0])

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": [5, 17, 27, 26, 19, 1, 1, 1],
            "padding_mask": [1, 1, 1, 1, 1, 0, 0, 0],
        }
        preprocessor = Qwen3CausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, "airplane at airport")
