from keras_hub.src.models.qwen3_5_moe.qwen3_5_moe_causal_lm_preprocessor import (  # noqa: E501
    Qwen3_5MoeCausalLMPreprocessor,
)
from keras_hub.src.models.qwen3_5_moe.qwen3_5_moe_tokenizer import (
    Qwen3_5MoeTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3_5MoeCausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.vocab = ["!", "air", "\u0120air", "plane", "\u0120at", "port"]
        self.vocab += ["<|im_end|>", "<|endoftext|>"]
        self.vocab += [
            "<|im_start|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "<|video_pad|>",
        ]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = [
            "\u0120 a",
            "\u0120 t",
            "\u0120 i",
            "\u0120 b",
            "a i",
            "p l",
            "n e",
        ]
        self.merges += [
            "\u0120a t",
            "p o",
            "r t",
            "\u0120t h",
            "ai r",
            "pl a",
            "po rt",
        ]
        self.merges += ["\u0120ai r", "\u0120a i", "pla ne"]
        self.tokenizer = Qwen3_5MoeTokenizer(
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
            cls=Qwen3_5MoeCausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[1, 3, 4, 2, 5, 6, 7, 7]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0]],
                },
                [[3, 4, 2, 5, 6, 7, 7, 7]],
                [[1, 1, 1, 1, 1, 0, 0, 0]],
            ),
        )

    def test_with_start_end_token(self):
        input_data = ["airplane at airport"] * 4
        preprocessor = Qwen3_5MoeCausalLMPreprocessor(
            **self.init_kwargs,
            add_start_token=True,
            add_end_token=True,
        )
        x, y, sw = preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[1, 3, 4, 2, 5, 6, 7, 7]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 1, 0, 0]] * 4)
        self.assertAllEqual(y, [[3, 4, 2, 5, 6, 7, 7, 7]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 1, 1, 0, 0, 0]] * 4)

    def test_generate_preprocess(self):
        input_data = "airplane at airport"
        preprocessor = Qwen3_5MoeCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(x["token_ids"], [1, 3, 4, 2, 5, 7, 7, 7])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 0, 0, 0])

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": [1, 3, 4, 2, 5, 7, 7, 7],
            "padding_mask": [1, 1, 1, 1, 1, 0, 0, 0],
        }
        preprocessor = Qwen3_5MoeCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, "airplane at airport")
