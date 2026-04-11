import numpy as np
import pytest
import tensorflow as tf

from keras_hub.src.models.gemma4.gemma4_causal_lm_preprocessor import (
    Gemma4CausalLMPreprocessor,
)
from keras_hub.src.models.gemma4.gemma4_image_converter import (
    Gemma4ImageConverter,
)
from keras_hub.src.models.gemma4.gemma4_video_converter import (
    Gemma4VideoConverter,
)
from keras_hub.src.tests.mocks.mock_gemma4_tokenizer import MockGemma4Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Gemma4CausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = MockGemma4Tokenizer()

        # === Text Preprocessor ===
        self.init_text_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": None,
            "sequence_length": 8,
            "max_images_per_prompt": 0,
            "num_vision_tokens_per_image": 0,
        }
        self.text_preprocessor = Gemma4CausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=None,
            sequence_length=100,
            max_images_per_prompt=0,
            num_vision_tokens_per_image=0,
        )

        # === Text + Image Preprocessor ===
        self.image_converter = Gemma4ImageConverter(
            image_size=(4, 4),
            patch_size=4,
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": self.image_converter,
            "sequence_length": 20,
            "max_images_per_prompt": 2,
            "num_vision_tokens_per_image": 5,
        }

    def test_text_preprocessor_basics(self):
        input_data = {
            "prompts": ["the quick brown fox"],
            "responses": ["round"],
        }
        preprocessor = Gemma4CausalLMPreprocessor(**self.init_text_kwargs)
        self.run_serialization_test(preprocessor)

        output = preprocessor(input_data)

        expected_output_batched = (
            {
                "token_ids": [[1, 9, 14, 10, 12, 15, 2, 0]],
                "padding_mask": [[1, 1, 1, 1, 1, 1, 1, 0]],
                "position_ids": [[0, 1, 2, 3, 4, 5, 6, 7]],
            },
            [[9, 14, 10, 12, 15, 2, 0, 0]],  # Labels shifted.
            [[0, 0, 0, 0, 1, 1, 0, 0]],  # Zero out unlabeled examples.
        )

        self.assertAllClose(output, expected_output_batched)

    def test_preprocessor_basics(self):
        input_data = {
            "prompts": ["the quick brown fox <|image|>"],
            "responses": ["round"],
            "pixel_values": np.ones((1, 2, 1, 48), dtype="float32"),
            "pixel_position_ids": np.ones((1, 2, 1, 2), dtype="int32"),
        }
        output = self.run_preprocessing_layer_test(
            cls=Gemma4CausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=input_data,
            return_output=True,
        )

        expected_output = [
            {
                "vision_indices": [list(range(6, 11)) + [0] * 5],
                "vision_mask": [[0] * 6 + [1] * 5 + [0] * 9],
                "token_ids": [
                    [1, 9, 14, 10, 12, 4] + [8] * 5 + [5, 15, 2] + [0] * 6
                ],
                "padding_mask": [[1] * 14 + [0] * 6],
            },
            [
                [9, 14, 10, 12, 4] + [8] * 5 + [5, 15, 2] + [0] * 7
            ],  # Labels shifted.
            [[0] * 11 + [1] * 2 + [0] * 7],  # Zero out unlabeled examples.
        ]

        # Check pixel_values shape.
        self.assertAllEqual(output[0]["pixel_values"].shape, [1, 2, 1, 48])

        # Check the remaining values.
        del output[0]["pixel_values"]
        del output[0]["pixel_position_ids"]
        for key in expected_output[0].keys():
            self.assertAllEqual(output[0][key], expected_output[0][key])
        self.assertAllEqual(output[1], expected_output[1])
        self.assertAllEqual(output[2], expected_output[2])

    def test_preprocessor_images_input(self):
        # Passes `images` explicitly to ensure ImageConverter integration works
        input_data = {
            "prompts": ["the quick brown fox <|image|>"],
            "responses": ["round"],
            # Since mock image converter simulates patches cleanly, we pass a
            # raw 4D image stack mapping:
            # max_images=2, image_size = (4, 4), rgb = 3
            "images": np.ones((1, 2, 4, 4, 3), dtype="float32"),
        }
        output = self.run_preprocessing_layer_test(
            cls=Gemma4CausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=input_data,
            return_output=True,
        )

        # Verify the converter created 'pixel_values' automatically
        # representing 4D patches (batch, 2, 1, 48)
        self.assertEqual(len(output[0]["pixel_values"].shape), 4)
        self.assertEqual(
            output[0]["pixel_values"].shape[-1], 48
        )  # patch_dim (4*4*3 = 48)

    def test_text_no_start_end_token(self):
        input_data = {
            "prompts": ["the quick brown fox"] * 4,
            "responses": ["round"] * 4,
        }
        preprocessor = Gemma4CausalLMPreprocessor(
            **self.init_text_kwargs,
            add_start_token=False,
            add_end_token=False,
        )
        x, y, sw = preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[9, 14, 10, 12, 15, 0, 0, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 0, 0, 0]] * 4)
        self.assertAllEqual(y, [[14, 10, 12, 15, 0, 0, 0, 0]] * 4)
        self.assertAllEqual(sw, [[0, 0, 0, 1, 0, 0, 0, 0]] * 4)

    def test_text_generate_preprocess(self):
        input_data = "the quick brown fox"
        preprocessor = Gemma4CausalLMPreprocessor(**self.init_text_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(x["token_ids"], [1, 9, 14, 10, 12, 0, 0, 0])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 0, 0, 0])

    def test_generate_preprocess(self):
        input_data = {
            "prompts": "the quick brown fox <|image|>",
            "pixel_values": np.ones((2, 1, 48), dtype="float32"),
            "pixel_position_ids": np.ones((2, 1, 2), dtype="int32"),
        }
        preprocessor = Gemma4CausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(
            x["token_ids"],
            [1, 9, 14, 10, 12, 4] + [8] * 5 + [5] + [0] * 8,
        )
        self.assertAllEqual(x["padding_mask"], [1] * 12 + [0] * 8)
        self.assertAllEqual(x["vision_indices"], list(range(6, 11)) + [0] * 5)
        self.assertAllEqual(x["vision_mask"], [0] * 6 + [1] * 5 + [0] * 9)
        self.assertAllEqual(x["pixel_values"].shape, [2, 1, 48])

    def test_text_generate_postprocess(self):
        input_data = {
            "token_ids": [1, 9, 14, 10, 12, 0, 0, 0],
            "padding_mask": [1, 1, 1, 1, 1, 0, 0, 0],
        }
        preprocessor = Gemma4CausalLMPreprocessor(**self.init_text_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, "the quick brown fox")

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": [1, 9, 14, 10, 12, 16, 4]
            + [8] * 5
            + [5, 16]
            + [0] * 6,
            "padding_mask": [1] * 14 + [0] * 6,
        }
        preprocessor = Gemma4CausalLMPreprocessor(**self.init_text_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, "the quick brown fox \n\n <|image>")

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            input_data = {
                "prompts": ["hello world", "this is testing"],
                "responses": [""],
            }
            self.text_preprocessor(input_data)
        with self.assertRaises(ValueError):
            input_data = {
                "prompts": ["hello world", "this is testing"],
                "responses": ["hello", "", ""],
            }
            self.text_preprocessor(input_data)

    def test_text_input_dummy_pixel_values_shape(self):
        # When an image-capable preprocessor receives text-only input (no
        # images / pixel_values), it must produce dummy pixel_values whose
        # last dimension equals 3 * patch_size ** 2 (derived from the
        # image_converter.
        preprocessor = Gemma4CausalLMPreprocessor(**self.init_kwargs)
        patch_size = self.image_converter.patch_size  # 4 in the test fixture
        expected_patch_dim = 3 * patch_size**2  # 48

        input_data = {
            "prompts": ["the quick brown fox"],
            "responses": ["round"],
        }
        output = preprocessor(input_data)
        pixel_values = output[0]["pixel_values"]
        pixel_position_ids = output[0]["pixel_position_ids"]

        # Shape: (batch=1, num_images=0, num_patches=1, patch_dim)
        self.assertEqual(pixel_values.shape[1], 0)
        self.assertEqual(pixel_values.shape[3], expected_patch_dim)
        # pixel_position_ids last dim is always 2 (row, col coordinates).
        self.assertEqual(pixel_position_ids.shape[1], 0)
        self.assertEqual(pixel_position_ids.shape[3], 2)

    def test_video_preprocessor_basics(self):
        video_converter = Gemma4VideoConverter(
            patch_size=4,
            num_frames=2,
            max_soft_tokens=2,
        )
        init_video_kwargs = {
            "tokenizer": self.tokenizer,
            "video_converter": video_converter,
            "sequence_length": 30,
            "num_frames_per_video": 2,
            "num_vision_tokens_per_frame": 2,
        }

        input_data = {
            "prompts": ["the quick brown fox <|video|>"],
            "responses": ["round"],
            "videos": np.ones((1, 2, 4, 4, 3), dtype="float32"),
        }

        preprocessor = Gemma4CausalLMPreprocessor(**init_video_kwargs)

        output = preprocessor(input_data)

        self.assertIn("pixel_values", output[0])
        self.assertIn("pixel_position_ids", output[0])
        self.assertIn("vision_mask", output[0])

        import keras

        vision_mask = output[0]["vision_mask"]
        # 2 frames × 1 token per frame: a 4×4 frame with patch_size=4 and
        # max_soft_tokens=2 produces 1 visible token per frame.
        self.assertEqual(int(keras.ops.sum(vision_mask)), 2)

    def test_video_metadata_timestamps(self):
        """video_metadata attribute produces correct per-frame timestamps."""
        video_converter = Gemma4VideoConverter(
            patch_size=4,
            num_frames=2,
            max_soft_tokens=2,
        )
        # Use fps=1.0 so that integer frame indices map directly to seconds,
        # making the expected "MM:SS" values easy to reason about.
        preprocessor = Gemma4CausalLMPreprocessor(
            tokenizer=self.tokenizer,
            video_converter=video_converter,
            sequence_length=100,
            num_frames_per_video=2,
            num_vision_tokens_per_frame=1,
            video_fps=1.0,
        )

        prompts = tf.constant(["<|video|>"])

        # === Default (no metadata): sequential [0, 1] at fps=1.0 ===
        # Frame 0 → 0.0 s → "00:00", frame 1 → 1.0 s → "00:01".
        expanded_default = preprocessor._expand_video_prompt(prompts, None)
        expanded_default_str = expanded_default.numpy()[0].decode("utf-8")
        self.assertIn("00:00", expanded_default_str)
        self.assertIn("00:01", expanded_default_str)

        # === With video_metadata: frames_indices=[0, 60], fps=1.0 ===
        # Frame 0 → 0.0 s → "00:00", frame 60 → 60.0 s → "01:00".
        preprocessor.video_metadata = [{"frames_indices": [0, 60], "fps": 1.0}]
        expanded_meta = preprocessor._expand_video_prompt(prompts, None)
        expanded_meta_str = expanded_meta.numpy()[0].decode("utf-8")
        self.assertIn("00:00", expanded_meta_str)
        self.assertIn("01:00", expanded_meta_str)
        # The non-metadata default "00:01" must NOT appear when frame 60
        # produces "01:00" instead.
        self.assertNotIn("00:01", expanded_meta_str)

        # === Fallback restores default after clearing video_metadata ===
        preprocessor.video_metadata = None
        expanded_cleared = preprocessor._expand_video_prompt(prompts, None)
        self.assertEqual(
            expanded_cleared.numpy()[0], expanded_default.numpy()[0]
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        text_input_data = {
            "prompts": ["the quick brown fox"],
            "responses": ["round"],
        }
        vision_text_input_data = {
            "prompts": ["the quick brown fox <|image>"],
            "responses": ["round"],
            "pixel_values": np.ones((1, 2, 1, 48), dtype="float32"),
            "pixel_position_ids": np.ones((1, 2, 1, 2), dtype="int32"),
        }

        for preset in Gemma4CausalLMPreprocessor.presets:
            if "_text" in preset:
                input_data = text_input_data
            else:
                input_data = vision_text_input_data
            self.run_preset_test(
                cls=Gemma4CausalLMPreprocessor,
                preset=preset,
                input_data=input_data,
            )
