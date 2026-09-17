import pytest

from keras_hub.src.models.qwen2_vl.qwen2_vl_tokenizer import Qwen2VLTokenizer
from keras_hub.src.tests.test_case import TestCase


class Qwen2VLTokenizerTest(TestCase):
    def setUp(self):
        merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i"]
        merges += ["p l", "n e", "Ġa t", "p o", "r t", "Ġt h"]
        merges += ["ai r", "pl a", "po rt", "Ġai r", "Ġa i"]
        merges += ["pla ne"]
        vocab = []
        for merge in merges:
            a, b = merge.split(" ")
            vocab.extend([a, b, a + b])
        vocab += ["<|endoftext|>", "<|im_end|>", "<|im_start|>"]
        vocab += ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>"]
        vocab += ["<|video_pad|>", "!"]
        self.vocab = sorted(set(vocab))
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = merges
        self.init_kwargs = {
            "vocabulary": self.vocab,
            "merges": self.merges,
        }
        self.input_data = ["airplane at airport"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=Qwen2VLTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[10, 22, 32, 31, 24]],
        )

    def test_special_token_ids(self):
        tokenizer = Qwen2VLTokenizer(**self.init_kwargs)
        # Vision tokens resolve to their vocabulary IDs.
        for tok in (
            "<|im_start|>",
            "<|im_end|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "<|video_pad|>",
        ):
            self.assertIn(tok, self.vocab)
        self.assertEqual(tokenizer.image_token_id, self.vocab["<|image_pad|>"])
        self.assertEqual(tokenizer.video_token_id, self.vocab["<|video_pad|>"])
        self.assertEqual(
            tokenizer.vision_start_token_id, self.vocab["<|vision_start|>"]
        )
        self.assertEqual(
            tokenizer.vision_end_token_id, self.vocab["<|vision_end|>"]
        )
        self.assertEqual(tokenizer.end_token_id, self.vocab["<|im_end|>"])
        self.assertEqual(tokenizer.pad_token_id, self.vocab["<|endoftext|>"])

    def test_vision_tokens_not_split(self):
        """Special tokens must tokenize to a single ID, not sub-word pieces."""
        tokenizer = Qwen2VLTokenizer(**self.init_kwargs)
        ids = tokenizer("<|vision_start|><|image_pad|><|vision_end|>")
        ids = list(ids)
        self.assertEqual(
            ids,
            [
                self.vocab["<|vision_start|>"],
                self.vocab["<|image_pad|>"],
                self.vocab["<|vision_end|>"],
            ],
        )

    def test_no_start_token(self):
        tokenizer = Qwen2VLTokenizer(**self.init_kwargs)
        self.assertIsNone(tokenizer.start_token)
        self.assertIsNone(tokenizer.start_token_id)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Qwen2VLTokenizer.presets:
            self.run_preset_test(
                cls=Qwen2VLTokenizer,
                preset=preset,
                input_data=self.input_data,
            )
