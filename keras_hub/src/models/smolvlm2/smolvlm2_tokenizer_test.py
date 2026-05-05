import pytest

from keras_hub.src.models.smolvlm2.smolvlm2_tokenizer import SmolVLM2Tokenizer
from keras_hub.src.tests.test_case import TestCase


class SmolVLM2TokenizerTest(TestCase):
    def setUp(self):
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += [
            "Ġa t",
            "p o",
            "r t",
            "Ġt h",
            "ai r",
            "pl a",
            "po rt",
        ]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.vocab = []
        for merge in self.merges:
            a, b = merge.split(" ")
            self.vocab.extend([a, b, a + b])
        self.vocab = sorted(set(self.vocab))  # Remove duplicates
        self.vocab += ["!"]
        self.vocab += ["<|begin_of_text|>"]
        self.vocab += ["<|end_of_text|>"]
        self.vocab += ["<image>"]
        self.vocab += ["<end_of_utterance>"]
        self.vocab += ["<|im_start|>"]
        self.vocab += ["<|im_end|>"]
        self.vocab += ["<fake_token_around_image>"]
        self.vocab += ["<global-img>"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.init_kwargs = {
            "vocabulary": self.vocab,
            "merges": self.merges,
        }
        self.input_data = [" airplane at airport"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=SmolVLM2Tokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[23, 14, 24, 23, 16]],
        )

    def test_special_tokens(self):
        """Verify all special tokens are registered and have valid IDs."""
        tokenizer = SmolVLM2Tokenizer(**self.init_kwargs)

        # Start/end tokens.
        self.assertEqual(tokenizer.start_token, "<|im_start|>")
        self.assertEqual(tokenizer.end_token, "<|im_end|>")
        self.assertIsNotNone(tokenizer.start_token_id)
        self.assertIsNotNone(tokenizer.end_token_id)

        # Image-related tokens.
        self.assertEqual(tokenizer.image_token, "<image>")
        self.assertIsNotNone(tokenizer.image_token_id)
        self.assertEqual(
            tokenizer.fake_image_token, "<fake_token_around_image>"
        )
        self.assertIsNotNone(tokenizer.fake_image_token_id)
        self.assertEqual(tokenizer.global_image_token, "<global-img>")
        self.assertIsNotNone(tokenizer.global_image_token_id)

        # End-of-utterance token.
        self.assertEqual(tokenizer.end_of_utterance_token, "<end_of_utterance>")
        self.assertIsNotNone(tokenizer.end_of_utterance_token_id)

    def test_pad_token(self):
        tokenizer = SmolVLM2Tokenizer(**self.init_kwargs)
        self.assertEqual(tokenizer.pad_token_id, 0)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in SmolVLM2Tokenizer.presets:
            self.run_preset_test(
                cls=SmolVLM2Tokenizer,
                preset=preset,
                input_data=self.input_data,
            )
