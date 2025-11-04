import numpy as np

from keras_hub.src.models.gemma3n.gemma3n_audio_converter import (
    Gemma3nAudioConverter,
)
from keras_hub.src.models.gemma3n.gemma3n_causal_lm_preprocessor import (
    Gemma3nCausalLMPreprocessor,
)
from keras_hub.src.models.gemma3n.gemma3n_image_converter import (
    Gemma3nImageConverter,
)
from keras_hub.src.tests.mocks.mock_gemma3n_tokenizer import (
    MockGemma3nTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class Gemma3nCausalLMPreprocessorTest(TestCase):
    def setUp(self):
        # Easier to use a mock here, instead of trying to figure out why
        # SentencePiece cannot tokenize and detokenize special tokens
        # properly.
        self.tokenizer = MockGemma3nTokenizer()

        # === Text Preprocessor ===
        self.init_text_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": None,
            "audio_converter": None,
            "sequence_length": 8,
            "max_images_per_prompt": 0,
            "num_vision_tokens_per_image": 0,
            "max_audios_per_prompt": 0,
            "num_audio_tokens_per_audio": 0,
        }
        self.text_preprocessor = Gemma3nCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=None,
            audio_converter=None,
            sequence_length=100,
            max_images_per_prompt=0,
            num_vision_tokens_per_image=0,
            max_audios_per_prompt=0,
            num_audio_tokens_per_audio=0,
        )

        # === Text + Image Preprocessor ===
        self.image_converter = Gemma3nImageConverter(
            image_size=(4, 4),
        )
        self.init_vision_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": self.image_converter,
            "audio_converter": None,
            "sequence_length": 20,
            "max_images_per_prompt": 2,
            "num_vision_tokens_per_image": 5,
            "max_audios_per_prompt": 0,
            "num_audio_tokens_per_audio": 0,
        }

        # === Text + Audio Preprocessor ===
        self.audio_converter = Gemma3nAudioConverter(
            feature_size=16,
            sampling_rate=16000,
        )
        self.init_audio_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": None,
            "audio_converter": self.audio_converter,
            "sequence_length": 20,
            "max_images_per_prompt": 0,
            "num_vision_tokens_per_image": 0,
            "max_audios_per_prompt": 2,
            "num_audio_tokens_per_audio": 3,
        }

        # === Text + Image + Audio Preprocessor ===
        self.init_multimodal_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": self.image_converter,
            "audio_converter": self.audio_converter,
            "sequence_length": 30,
            "max_images_per_prompt": 2,
            "num_vision_tokens_per_image": 5,
            "max_audios_per_prompt": 2,
            "num_audio_tokens_per_audio": 3,
        }

    def test_text_preprocessor_basics(self):
        input_data = {
            "prompts": ["the quick brown fox"],
            "responses": ["round"],
        }
        self.run_preprocessing_layer_test(
            cls=Gemma3nCausalLMPreprocessor,
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

    def test_vision_preprocessor_basics(self):
        input_data = {
            "prompts": ["the quick brown fox <start_of_image>"],
            "responses": ["round"],
            "images": [[np.ones((8, 8, 3))]],
        }
        output = self.run_preprocessing_layer_test(
            cls=Gemma3nCausalLMPreprocessor,
            init_kwargs=self.init_vision_kwargs,
            input_data=input_data,
            return_output=True,
        )
        expected_output = [
            {
                "vision_indices": [list(range(7, 12)) + [0] * 5],
                "audio_indices": [[]],
                "vision_mask": [[0] * 7 + [1] * 5 + [0] * 8],
                "audio_mask": [[0] * 20],
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
        # Check shape for audios (should be empty).
        self.assertAllEqual(output[0]["audios"].shape, [1, 0, 0])
        self.assertAllEqual(output[0]["input_features"].shape, [1, 0, 0, 128])
        self.assertAllEqual(output[0]["input_features_mask"].shape, [1, 0, 0])
        # For everything else, check the actual values.
        del output[0]["images"]
        del output[0]["audios"]
        del output[0]["input_features"]
        del output[0]["input_features_mask"]
        for key in expected_output[0].keys():
            self.assertAllEqual(output[0][key], expected_output[0][key])
        self.assertAllEqual(output[1], expected_output[1])
        self.assertAllEqual(output[2], expected_output[2])

    def test_audio_preprocessor_basics(self):
        input_data = {
            "prompts": ["the quick <start_of_audio>"],
            "responses": ["brown"],
            "audios": [[np.ones((16000,))]],
        }
        preprocessor = Gemma3nCausalLMPreprocessor(**self.init_audio_kwargs)
        output = preprocessor(input_data)
        # Check that we have the right keys.
        self.assertIn("token_ids", output[0])
        self.assertIn("vision_indices", output[0])
        self.assertIn("audio_indices", output[0])
        self.assertIn("vision_mask", output[0])
        self.assertIn("audio_mask", output[0])
        self.assertIn("padding_mask", output[0])
        self.assertIn("images", output[0])
        self.assertIn("audios", output[0])
        self.assertIn("input_features", output[0])
        self.assertIn("input_features_mask", output[0])
        # Check shapes for images (should be empty).
        self.assertAllEqual(output[0]["images"].shape[0:2], [1, 0])
        # Check shapes for audios (should have data).
        self.assertAllEqual(output[0]["audios"].shape[0:2], [1, 2])
        self.assertEqual(output[0]["input_features"].shape[0], 1)
        self.assertEqual(output[0]["input_features_mask"].shape[0], 1)

    def test_multimodal_preprocessor_basics(self):
        input_data = {
            "prompts": ["image <start_of_image> audio <start_of_audio>"],
            "responses": ["test"],
            "images": [[np.ones((8, 8, 3))]],
            "audios": [[np.ones((16000,))]],
        }
        preprocessor = Gemma3nCausalLMPreprocessor(
            **self.init_multimodal_kwargs
        )
        output = preprocessor(input_data)
        # Check that we have all the right keys.
        self.assertIn("token_ids", output[0])
        self.assertIn("vision_indices", output[0])
        self.assertIn("audio_indices", output[0])
        self.assertIn("vision_mask", output[0])
        self.assertIn("audio_mask", output[0])
        self.assertIn("padding_mask", output[0])
        self.assertIn("images", output[0])
        self.assertIn("audios", output[0])
        self.assertIn("input_features", output[0])
        self.assertIn("input_features_mask", output[0])
        # Check shapes for images.
        self.assertAllEqual(output[0]["images"].shape, [1, 2, 4, 4, 3])
        # Check shapes for audios.
        self.assertAllEqual(output[0]["audios"].shape[0:2], [1, 2])
        self.assertEqual(output[0]["input_features"].shape[0], 1)
        self.assertEqual(output[0]["input_features_mask"].shape[0], 1)
        # Check that both vision and audio masks have some True values.
        vision_mask_sum = np.sum(np.array(output[0]["vision_mask"]))
        audio_mask_sum = np.sum(np.array(output[0]["audio_mask"]))
        self.assertGreater(vision_mask_sum, 0)
        self.assertGreater(audio_mask_sum, 0)

    def test_text_no_start_end_token(self):
        input_data = {
            "prompts": ["the quick brown fox"] * 4,
            "responses": ["round"] * 4,
        }
        preprocessor = Gemma3nCausalLMPreprocessor(
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
        preprocessor = Gemma3nCausalLMPreprocessor(**self.init_text_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(x["token_ids"], [1, 9, 14, 10, 12, 0, 0, 0])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 0, 0, 0])

    def test_vision_generate_preprocess(self):
        input_data = {
            "prompts": "the quick brown fox <start_of_image>",
            "images": np.ones((8, 8, 3)),
        }
        preprocessor = Gemma3nCausalLMPreprocessor(**self.init_vision_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(
            x["token_ids"],
            [1, 9, 14, 10, 12, 16, 4] + [8] * 5 + [5, 16] + [0] * 6,
        )
        self.assertAllEqual(x["padding_mask"], [1] * 14 + [0] * 6)
        self.assertAllEqual(x["vision_indices"], list(range(7, 12)) + [0] * 5)
        self.assertAllEqual(x["vision_mask"], [0] * 7 + [1] * 5 + [0] * 8)
        self.assertAllEqual(x["images"].shape, [2, 4, 4, 3])

    def test_audio_generate_preprocess(self):
        input_data = {
            "prompts": "the quick <start_of_audio>",
            "audios": np.ones((16000,)),
        }
        preprocessor = Gemma3nCausalLMPreprocessor(**self.init_audio_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        # Check that we have the right keys.
        self.assertIn("token_ids", x)
        self.assertIn("audio_indices", x)
        self.assertIn("audio_mask", x)
        self.assertIn("audios", x)
        self.assertIn("input_features", x)
        self.assertIn("input_features_mask", x)

    def test_text_generate_postprocess(self):
        input_data = {
            "token_ids": [1, 9, 14, 10, 12, 0, 0, 0],
            "padding_mask": [1, 1, 1, 1, 1, 0, 0, 0],
        }
        preprocessor = Gemma3nCausalLMPreprocessor(**self.init_text_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, "the quick brown fox")

    def test_vision_generate_postprocess(self):
        input_data = {
            "token_ids": [1, 9, 14, 10, 12, 16, 4]
            + [8] * 5
            + [5, 16]
            + [0] * 6,
            "padding_mask": [1] * 14 + [0] * 6,
        }
        preprocessor = Gemma3nCausalLMPreprocessor(**self.init_text_kwargs)
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

    def test_text_only_with_images_raises_error(self):
        with self.assertRaises(ValueError):
            input_data = {
                "prompts": ["hello"],
                "responses": ["world"],
                "images": [np.ones((8, 8, 3))],
            }
            self.text_preprocessor(input_data)

    def test_text_only_with_audios_raises_error(self):
        with self.assertRaises(ValueError):
            input_data = {
                "prompts": ["hello"],
                "responses": ["world"],
                "audios": [np.ones((16000,))],
            }
            self.text_preprocessor(input_data)
