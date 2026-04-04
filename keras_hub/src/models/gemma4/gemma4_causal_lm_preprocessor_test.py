import numpy as np
import pytest

from keras_hub.src.models.gemma4.gemma4_causal_lm_preprocessor import (
    Gemma4CausalLMPreprocessor,
)
from keras_hub.src.models.gemma4.gemma4_image_converter import (
    Gemma4ImageConverter,
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
        self.run_preprocessing_layer_test(
            cls=Gemma4CausalLMPreprocessor,
            init_kwargs=self.init_text_kwargs,
            input_data=input_data,
            expected_output=(
                {
                    "token_ids": [[1, 9, 14, 10, 12, 15, 2, 0]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 1, 0]],
                },
                [[9, 14, 10, 12, 15, 2, 0, 0]],  # Labels shifted.
                [[0, 0, 0, 0, 1, 1, 0, 0]],  # Zero out unlabeled examples.
            ),
        )

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
