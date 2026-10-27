from keras_hub.src.models.qwen3.qwen3_text_embedder_preprocessor import (
    Qwen3TextEmbedderPreprocessor,
)
from keras_hub.src.models.qwen3.qwen3_tokenizer import Qwen3Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Qwen3TextEmbedderPreprocessorTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab += ["<|im_end|>", "<|endoftext|>", "!"]
        self.vocab = sorted(set(self.vocab))
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
            cls=Qwen3TextEmbedderPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output={
                "token_ids": [[5, 17, 27, 26, 19, 2, 1, 1]],
                "padding_mask": [
                    [True, True, True, True, True, True, False, False]
                ],
            },
        )

    def test_output_keys(self):
        preprocessor = Qwen3TextEmbedderPreprocessor(**self.init_kwargs)
        # Without labels, the preprocessor returns the feature dict directly.
        output = preprocessor(self.input_data)
        self.assertIn("token_ids", output)
        self.assertIn("padding_mask", output)
        self.assertNotIn("segment_ids", output)

    def test_padding_mask_matches_non_pad_tokens(self):
        preprocessor = Qwen3TextEmbedderPreprocessor(**self.init_kwargs)
        # Without labels, the preprocessor returns the feature dict directly.
        output = preprocessor(self.input_data)
        pad_id = self.tokenizer.pad_token_id
        expected_mask = output["token_ids"] != pad_id
        self.assertAllEqual(output["padding_mask"], expected_mask)
