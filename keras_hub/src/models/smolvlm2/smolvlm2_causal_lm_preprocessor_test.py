"""Tests for SmolVLM2CausalLMPreprocessor."""

import numpy as np
from keras import ops

from keras_hub.src.models.smolvlm2.smolvlm2_causal_lm_preprocessor import (
    SmolVLM2CausalLMPreprocessor,
)
from keras_hub.src.models.smolvlm2.smolvlm2_causal_lm_preprocessor import (
    _get_image_prompt_string,
)
from keras_hub.src.models.smolvlm2.smolvlm2_image_converter import (
    SmolVLM2ImageConverter,
)
from keras_hub.src.models.smolvlm2.smolvlm2_tokenizer import SmolVLM2Tokenizer
from keras_hub.src.models.smolvlm2.smolvlm2_video_converter import (
    SmolVLM2VideoConverter,
)
from keras_hub.src.tests.test_case import TestCase


class SmolVLM2CausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.vocab = ["!", "air", "\u0120air", "plane", "\u0120at", "port"]
        self.vocab += ["<|begin_of_text|>"]
        self.vocab += ["<|end_of_text|>"]
        self.vocab += ["<image>"]
        self.vocab += ["<end_of_utterance>"]
        self.vocab += ["<|im_start|>"]
        self.vocab += ["<|im_end|>"]
        self.vocab += ["<fake_token_around_image>"]
        self.vocab += ["<global-img>"]
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
        self.tokenizer = SmolVLM2Tokenizer(
            vocabulary=self.vocab,
            merges=self.merges,
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = [" airplane at airport"]

    def test_preprocessor_basics(self):
        # " airplane at airport" tokenizes to [2, 3, 4, 2, 5].
        # call() packs (prompts, responses) = same text duplicated.
        # Packer with seq_length=9 (8+1): [2,3,4,2, 2,3,4,2, 11] → truncated.
        # token_ids[:-1] = [2,3,4,2, 2,3,4,2].
        preprocessor = SmolVLM2CausalLMPreprocessor(**self.init_kwargs)
        x, y, sw = preprocessor(self.input_data)

        self.assertAllEqual(x["token_ids"], [[2, 3, 4, 2, 2, 3, 4, 2]])
        self.assertIn("padding_mask", x)

    def test_with_start_end_token(self):
        input_data = [" airplane at airport"] * 4
        preprocessor = SmolVLM2CausalLMPreprocessor(
            **self.init_kwargs,
            add_start_token=True,
            add_end_token=True,
        )
        x, y, sw = preprocessor(input_data)
        # start=10, [2,3,4,2, 2,3,4] truncated to 8 positions.
        self.assertAllEqual(x["token_ids"], [[10, 2, 3, 4, 2, 2, 3, 4]] * 4)

    def test_generate_preprocess_text_only(self):
        input_data = " airplane at airport"
        preprocessor = SmolVLM2CausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertIn("token_ids", x)
        self.assertIn("padding_mask", x)
        # Text-only should NOT have vision keys.
        self.assertNotIn("pixel_values", x)
        self.assertNotIn("vision_indices", x)

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": [2, 3, 4, 2, 5, 0, 0, 0],
            "padding_mask": [1, 1, 1, 1, 1, 0, 0, 0],
        }
        preprocessor = SmolVLM2CausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, " airplane at airport")

    def test_generate_preprocess_with_image(self):
        """Multimodal prompt with an image produces vision keys."""
        image_converter = SmolVLM2ImageConverter(
            max_image_size=32,
            size=64,
            do_image_splitting=False,
            scale=[1 / 255.0] * 3,
            offset=[0.0] * 3,
            interpolation="bicubic",
        )
        preprocessor = SmolVLM2CausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=image_converter,
            sequence_length=128,
            image_seq_len=4,  # Small for testing.
        )

        img = np.random.randint(0, 256, size=(20, 20, 3)).astype("uint8")
        prompt = (
            "<|im_start|>User:<image>describe<end_of_utterance>\nAssistant:"
        )

        x = preprocessor.generate_preprocess({"prompts": prompt, "images": img})

        self.assertIn("token_ids", x)
        self.assertIn("padding_mask", x)
        self.assertIn("pixel_values", x)
        self.assertIn("vision_indices", x)

        # pixel_values should be (1, 32, 32, 3) — single crop, no splitting.
        pixel_values = ops.convert_to_numpy(x["pixel_values"])
        self.assertEqual(pixel_values.shape[1], 32)
        self.assertEqual(pixel_values.shape[2], 32)
        self.assertEqual(pixel_values.shape[3], 3)

        # vision_indices should contain positions where <image> tokens are.
        vision_indices = ops.convert_to_numpy(x["vision_indices"])
        self.assertGreater(len(vision_indices), 0)

    def test_prompt_expansion_unsplit(self):
        """Unsplit image produces <fake><global-img><image>×N<fake> format."""
        result = _get_image_prompt_string(
            image_seq_len=3,
            image_rows=0,
            image_cols=0,
            fake_token_around_image="<fake_token_around_image>",
            image_token="<image>",
            global_image_token="<global-img>",
        )
        # Should be: <fake><global-img><image><image><image><fake>
        self.assertIn("<fake_token_around_image>", result)
        self.assertIn("<global-img>", result)
        self.assertEqual(result.count("<image>"), 3)

    def test_prompt_expansion_split(self):
        """Split image produces per-patch <row_R_col_C> + global view."""
        result = _get_image_prompt_string(
            image_seq_len=2,
            image_rows=2,
            image_cols=3,
            fake_token_around_image="<fake_token_around_image>",
            image_token="<image>",
            global_image_token="<global-img>",
        )
        # 2×3 = 6 patches + 1 global = 7 sub-images × 2 tokens = 14.
        self.assertEqual(result.count("<image>"), 14)
        # Should contain row/col tags.
        self.assertIn("<row_1_col_1>", result)
        self.assertIn("<row_2_col_3>", result)
        # Should contain global tag.
        self.assertIn("<global-img>", result)

    def test_special_token_tokenization(self):
        """_tokenize_with_special_tokens preserves special tokens."""
        preprocessor = SmolVLM2CausalLMPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=32,
        )
        if not preprocessor.built:
            preprocessor.build(None)

        text = "<|im_start|> air<end_of_utterance>"
        ids = preprocessor._tokenize_with_special_tokens(text)

        # <|im_start|> should be a single ID.
        start_id = self.tokenizer.start_token_id
        eou_id = self.tokenizer.end_of_utterance_token_id
        self.assertEqual(ids[0], start_id)
        self.assertEqual(ids[-1], eou_id)

    def test_generate_preprocess_with_video(self):
        """Video prompt with <video> produces vision keys."""
        video_converter = SmolVLM2VideoConverter(
            max_image_size=32,
            size=64,
            num_frames=3,
            fps=1,
            scale=[1 / 255.0] * 3,
            offset=[0.0] * 3,
            interpolation="bicubic",
        )
        preprocessor = SmolVLM2CausalLMPreprocessor(
            tokenizer=self.tokenizer,
            video_converter=video_converter,
            sequence_length=512,
            image_seq_len=4,
        )

        # Fake video: 6 frames of 48x64.
        video = np.random.randint(0, 256, size=(6, 48, 64, 3)).astype("uint8")
        prompt = (
            "<|im_start|>User:<video>describe<end_of_utterance>\nAssistant:"
        )

        x = preprocessor.generate_preprocess(
            {"prompts": prompt, "videos": video}
        )

        self.assertIn("token_ids", x)
        self.assertIn("padding_mask", x)
        self.assertIn("pixel_values", x)
        self.assertIn("vision_indices", x)

        # pixel_values should be (3, 32, 32, 3) — 3 sampled frames.
        pixel_values = ops.convert_to_numpy(x["pixel_values"])
        self.assertEqual(pixel_values.shape[0], 3)
        self.assertEqual(pixel_values.shape[1], 32)
        self.assertEqual(pixel_values.shape[2], 32)
        self.assertEqual(pixel_values.shape[3], 3)

        # vision_indices should contain <image> token positions.
        vision_indices = ops.convert_to_numpy(x["vision_indices"])
        self.assertGreater(len(vision_indices), 0)

    def test_video_prompt_expansion(self):
        """Video prompt string has per-frame timestamps."""
        preprocessor = SmolVLM2CausalLMPreprocessor(
            tokenizer=self.tokenizer,
            sequence_length=32,
            image_seq_len=2,
        )
        if not preprocessor.built:
            preprocessor.build(None)

        prompt = preprocessor._get_video_prompt_string(
            num_frames=3,
            metadata={"fps": 1, "duration": 3},
        )

        # Should contain video intro.
        self.assertIn("3 frames", prompt)
        self.assertIn("[H:MM:SS]", prompt)

        # Should have per-frame timestamps.
        self.assertIn("Frame from 00:00:", prompt)
        self.assertIn("Frame from 00:01:", prompt)
        self.assertIn("Frame from 00:02:", prompt)

        # Each frame gets image_seq_len=2 <image> tokens.
        self.assertEqual(prompt.count("<image>"), 6)  # 3 frames × 2

        # Each frame wrapped with <fake_token_around_image>.
        self.assertIn("<fake_token_around_image>", prompt)
        self.assertIn("<global-img>", prompt)
