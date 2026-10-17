from keras_hub.src.models.flux2.flux2_text_to_image_preprocessor import (
    Flux2TextToImagePreprocessor,
)
from keras_hub.src.models.qwen3.qwen3_tokenizer import Qwen3Tokenizer
from keras_hub.src.tests.test_case import TestCase

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
]


class Flux2TextToImagePreprocessorTest(TestCase):
    def setUp(self):
        merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        merges += ["Ġai r", "Ġa i", "pla ne"]
        # The chat template's words and newlines ("Ċ").
        merges += ["u s", "e r", "us er", "a s", "as s", "i s", "ass is"]
        merges += ["t a", "n t", "ta nt", "assis tant", "Ċ Ċ"]
        vocab = []
        for merge in merges:
            a, b = merge.split(" ")
            vocab.extend([a, b, a + b])
        vocab = sorted(set(vocab)) + SPECIAL_TOKENS
        self.vocab = {token: i for i, token in enumerate(vocab)}
        self.merges = merges
        self.tokenizer = Qwen3Tokenizer(
            vocabulary=self.vocab,
            merges=self.merges,
            unsplittable_tokens=SPECIAL_TOKENS,
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 20,
        }
        self.input_data = ["airplane at airport"]
        # "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        # "<think>\n\n</think>\n\n", as tokenized by the Qwen3 tokenizer.
        self.expected_tokens = [
            "<|im_start|>",
            "user",
            "Ċ",
            "air",
            "plane",
            "Ġat",
            "Ġair",
            "port",
            "<|im_end|>",
            "Ċ",
            "<|im_start|>",
            "assistant",
            "Ċ",
            "<think>",
            "ĊĊ",
            "</think>",
            "ĊĊ",
        ]

    def token_ids(self, tokens):
        return [self.vocab[token] for token in tokens]

    def test_preprocessor_basics(self):
        pad = "<|endoftext|>"
        self.run_preprocessor_test(
            cls=Flux2TextToImagePreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output={
                "token_ids": [self.token_ids(self.expected_tokens + [pad] * 3)],
                "padding_mask": [[1] * 17 + [0] * 3],
            },
        )

    def test_generate_preprocess(self):
        preprocessor = Flux2TextToImagePreprocessor(**self.init_kwargs)
        # Scalar input.
        x = preprocessor.generate_preprocess("airplane at airport")
        self.assertAllEqual(
            x["token_ids"],
            self.token_ids(self.expected_tokens + ["<|endoftext|>"] * 3),
        )
        self.assertAllEqual(x["padding_mask"], [1] * 17 + [0] * 3)
        # Batched input.
        x = preprocessor.generate_preprocess(["airplane at airport"] * 2)
        self.assertEqual(tuple(x["token_ids"].shape), (2, 20))

    def test_truncation(self):
        # Like Hugging Face's `truncation=True`, the templated sequence is
        # cut on the right and no end token is appended.
        preprocessor = Flux2TextToImagePreprocessor(
            self.tokenizer, sequence_length=5
        )
        x = preprocessor(self.input_data)
        self.assertAllEqual(
            x["token_ids"], [self.token_ids(self.expected_tokens[:5])]
        )
        self.assertAllEqual(x["padding_mask"], [[1] * 5])

    def test_requires_unsplittable_template_tokens(self):
        tokenizer = Qwen3Tokenizer(vocabulary=self.vocab, merges=self.merges)
        preprocessor = Flux2TextToImagePreprocessor(tokenizer)
        with self.assertRaisesRegex(ValueError, "unsplittable"):
            preprocessor(self.input_data)
