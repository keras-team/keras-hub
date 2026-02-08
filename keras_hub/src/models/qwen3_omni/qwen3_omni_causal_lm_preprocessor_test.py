import numpy as np
import pytest

from keras_hub.src.models.qwen3_omni.qwen3_omni_audio_converter import (
    Qwen3OmniAudioConverter,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_causal_lm_preprocessor import (
    Qwen3OmniCausalLMPreprocessor,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_image_converter import (
    Qwen3OmniImageConverter,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_tokenizer import (
    Qwen3OmniTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3OmniCausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.vocab = ["!", "air", "Ġair", "plane", "Ġat", "port"]
        self.vocab += ["<|im_end|>", "<|endoftext|>"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.tokenizer = Qwen3OmniTokenizer(
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
            cls=Qwen3OmniCausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[1, 3, 4, 2, 5, 6, 7, 7]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0]],
                },
                [[3, 4, 2, 5, 6, 7, 7, 7]],
                [[1, 1, 1, 1, 1, 0, 0, 0]],
            ),
        )

    def test_with_start_end_token(self):
        input_data = ["airplane at airport"] * 4
        preprocessor = Qwen3OmniCausalLMPreprocessor(
            **self.init_kwargs,
            add_start_token=True,
            add_end_token=True,
        )
        x, y, sw = preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[1, 3, 4, 2, 5, 6, 7, 7]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 1, 0, 0]] * 4)
        self.assertAllEqual(y, [[3, 4, 2, 5, 6, 7, 7, 7]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 1, 1, 0, 0, 0]] * 4)

    def test_generate_preprocess(self):
        input_data = "airplane at airport"
        preprocessor = Qwen3OmniCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(x["token_ids"], [1, 3, 4, 2, 5, 7, 7, 7])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 0, 0, 0])

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": [1, 3, 4, 2, 5, 7, 7, 7],
            "padding_mask": [1, 1, 1, 1, 1, 0, 0, 0],
        }
        preprocessor = Qwen3OmniCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, "airplane at airport")

    def test_with_audio_converter(self):
        # Test preprocessing with audio converter
        try:
            import librosa  # noqa: F401

            audio_converter = Qwen3OmniAudioConverter()
            preprocessor = Qwen3OmniCausalLMPreprocessor(
                tokenizer=self.tokenizer,
                audio_converter=audio_converter,
                sequence_length=8,
            )
            # Create dummy audio (1 second at 16kHz)
            audio_data = np.random.randn(16000).astype(np.float32)
            input_data = {
                "prompts": "airplane at airport",
                "responses": "airplane",
                "audio": audio_data,
            }
            x, y, sw = preprocessor(input_data)
            # Check that audio_features key is present
            self.assertIn("audio_features", x)
            self.assertIn("token_ids", x)
            self.assertIn("padding_mask", x)
        except ImportError:
            self.skipTest("librosa not installed")

    def test_with_image_converter(self):
        # Test preprocessing with image converter
        image_converter = Qwen3OmniImageConverter()
        preprocessor = Qwen3OmniCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=image_converter,
            sequence_length=8,
        )
        # Create dummy image
        image_data = np.ones((224, 224, 3), dtype=np.uint8) * 128
        input_data = {
            "prompts": "airplane at airport",
            "responses": "airplane",
            "images": image_data,
        }
        x, y, sw = preprocessor(input_data)
        # Check that pixel_values key is present
        self.assertIn("pixel_values", x)
        self.assertIn("token_ids", x)
        self.assertIn("padding_mask", x)

    def test_multimodal_generate_preprocess(self):
        # Test generate preprocess with audio and images
        try:
            import librosa  # noqa: F401

            audio_converter = Qwen3OmniAudioConverter()
            image_converter = Qwen3OmniImageConverter()
            preprocessor = Qwen3OmniCausalLMPreprocessor(
                tokenizer=self.tokenizer,
                audio_converter=audio_converter,
                image_converter=image_converter,
                sequence_length=8,
            )
            audio_data = np.random.randn(16000).astype(np.float32)
            image_data = np.ones((224, 224, 3), dtype=np.uint8) * 128
            input_data = {
                "prompts": "airplane",
                "audio": audio_data,
                "images": image_data,
            }
            x = preprocessor.generate_preprocess(input_data)
            # Check all keys are present
            self.assertIn("token_ids", x)
            self.assertIn("padding_mask", x)
            self.assertIn("audio_features", x)
            self.assertIn("pixel_values", x)
        except ImportError:
            self.skipTest("librosa not installed")

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Qwen3OmniCausalLMPreprocessor.presets:
            self.run_preset_test(
                cls=Qwen3OmniCausalLMPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )
