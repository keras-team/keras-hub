import numpy as np
import pytest

from keras_hub.src.models.gemma3.gemma3_causal_lm_preprocessor import (
    Gemma3CausalLMPreprocessor,
)
from keras_hub.src.models.gemma3.gemma3_image_converter import (
    Gemma3ImageConverter,
)
from keras_hub.src.tests.mocks.mock_gemma3_tokenizer import MockGemma3Tokenizer
from keras_hub.src.tests.test_case import TestCase


class Gemma3CausalLMPreprocessorTest(TestCase):
    def setUp(self):
        # Easier to use a mock here, instead of trying to figure out why
        # SentencePiece cannot tokenize and detokenize special tokens
        # properly.
        self.tokenizer = MockGemma3Tokenizer()

        # === Text Preprocessor ===
        self.init_text_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": None,
            "sequence_length": 8,
            "max_images_per_prompt": 0,
            "num_vision_tokens_per_image": 0,
        }
        self.text_preprocessor = Gemma3CausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=None,
            sequence_length=100,
            max_images_per_prompt=0,
            num_vision_tokens_per_image=0,
        )

        # === Text + Image Preprocessor ===
        self.image_converter = Gemma3ImageConverter(
            image_size=(4, 4),
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
            cls=Gemma3CausalLMPreprocessor,
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
            "prompts": ["the quick brown fox <start_of_image>"],
            "responses": ["round"],
            "images": [[np.ones((8, 8, 3))]],
        }
        output = self.run_preprocessing_layer_test(
            cls=Gemma3CausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=input_data,
            return_output=True,
        )

        expected_output = [
            {
                "vision_indices": [list(range(7, 12)) + [0] * 5],
                "vision_mask": [[0] * 7 + [1] * 5 + [0] * 8],
                "token_ids": [
                    [1, 9, 14, 10, 12, 16, 4]
                    + [8] * 5
                    + [5, 16, 15, 2]
                    + [0] * 4
                ],
                "padding_mask": [[1] * 16 + [0] * 4],
            },
            [
                [9, 14, 10, 12, 16, 4] + [8] * 5 + [5, 16, 15, 2] + [0] * 5
            ],  # Labels shifted.
            [[0] * 13 + [1] * 2 + [0] * 5],  # Zero out unlabeled examples.
        ]

        # Check shape for images.
        self.assertAllEqual(output[0]["images"].shape, [1, 2, 4, 4, 3])

        # For everything else, let's check the actual values.
        del output[0]["images"]
        for key in expected_output[0].keys():
            self.assertAllEqual(output[0][key], expected_output[0][key])
        self.assertAllEqual(output[1], expected_output[1])
        self.assertAllEqual(output[2], expected_output[2])

    def test_text_no_start_end_token(self):
        input_data = {
            "prompts": ["the quick brown fox"] * 4,
            "responses": ["round"] * 4,
        }

        preprocessor = Gemma3CausalLMPreprocessor(
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
        preprocessor = Gemma3CausalLMPreprocessor(**self.init_text_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(x["token_ids"], [1, 9, 14, 10, 12, 0, 0, 0])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 0, 0, 0])

    def test_generate_preprocess(self):
        input_data = {
            "prompts": "the quick brown fox <start_of_image>",
            "images": np.ones((8, 8, 3)),
        }
        preprocessor = Gemma3CausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(
            x["token_ids"],
            [1, 9, 14, 10, 12, 16, 4] + [8] * 5 + [5, 16] + [0] * 6,
        )
        self.assertAllEqual(x["padding_mask"], [1] * 14 + [0] * 6)
        self.assertAllEqual(x["vision_indices"], list(range(7, 12)) + [0] * 5)
        self.assertAllEqual(x["vision_mask"], [0] * 7 + [1] * 5 + [0] * 8)
        self.assertAllEqual(x["images"].shape, [2, 4, 4, 3])

    def test_text_generate_postprocess(self):
        input_data = {
            "token_ids": [1, 9, 14, 10, 12, 0, 0, 0],
            "padding_mask": [1, 1, 1, 1, 1, 0, 0, 0],
        }
        preprocessor = Gemma3CausalLMPreprocessor(**self.init_text_kwargs)
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
        preprocessor = Gemma3CausalLMPreprocessor(**self.init_text_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, "the quick brown fox \n\n <start_of_image>")

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
            "prompts": ["the quick brown fox <start_of_image>"],
            "responses": ["round"],
            "images": [[np.ones((8, 8, 3))]],
        }

        for preset in Gemma3CausalLMPreprocessor.presets:
            if "1b" in preset or "_text" in preset:
                input_data = text_input_data
            else:
                input_data = vision_text_input_data

            self.run_preset_test(
                cls=Gemma3CausalLMPreprocessor,
                preset=preset,
                input_data=input_data,
            )
