from keras_hub.src.models.qwen3_vl.qwen3_vl_tokenizer import Qwen3_VLTokenizer
from keras_hub.src.tests.test_case import TestCase


class Qwen3_VLTokenizerTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab = sorted(set(self.vocab))
        self.vocab += [
            "!",
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "<|video_pad|>",
            "<think>",
        ]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.init_kwargs = {"vocabulary": self.vocab, "merges": self.merges}
        self.input_data = [
            "<|im_start|>airplane at airport<|im_end|>",
            " airplane airport",
        ]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=Qwen3_VLTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[31, 2, 14, 24, 23, 16, 32], [23, 14, 23, 16]],
        )

    def test_special_token_ids(self):
        tokenizer = Qwen3_VLTokenizer(**self.init_kwargs)
        for name, token in [
            ("end_token", "<|im_end|>"),
            ("pad_token", "<|endoftext|>"),
            ("im_start_token", "<|im_start|>"),
            ("vision_start_token", "<|vision_start|>"),
            ("vision_end_token", "<|vision_end|>"),
            ("image_token", "<|image_pad|>"),
            ("video_token", "<|video_pad|>"),
        ]:
            self.assertEqual(getattr(tokenizer, name), token)
            self.assertEqual(
                getattr(tokenizer, f"{name}_id"), self.vocab[token]
            )

    def test_errors_missing_special_tokens(self):
        with self.assertRaises(ValueError):
            Qwen3_VLTokenizer(vocabulary=["a"], merges=["Ġ a"])
