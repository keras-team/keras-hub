import numpy as np
import pytest

from keras_hub.src.models.qwen2_vl.qwen2_vl_causal_lm_preprocessor import (
    Qwen2VLCausalLMPreprocessor,
)
from keras_hub.src.models.qwen2_vl.qwen2_vl_image_converter import (
    Qwen2VLImageConverter,
)
from keras_hub.src.models.qwen2_vl.qwen2_vl_tokenizer import Qwen2VLTokenizer
from keras_hub.src.tests.test_case import TestCase


class Qwen2VLCausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.vocab = ["!", "air", "Ġair", "plane", "Ġat", "port"]
        self.vocab += ["<|endoftext|>"]
        self.vocab += ["<|eot_id|>"]
        self.vocab += ["<|image_pad|>"]
        self.vocab += ["<|vision_start|>"]
        self.vocab += ["<|vision_end|>"]
        self.vocab = dict([(token, i) for i, token in enumerate(self.vocab)])
        self.merges = ["Ġ a", "Ġ t", "Ġ i", "Ġ b", "a i", "p l", "n e"]
        self.merges += ["Ġa t", "p o", "r t", "Ġt h", "ai r", "pl a", "po rt"]
        self.merges += ["Ġai r", "Ġa i", "pla ne"]
        self.tokenizer = Qwen2VLTokenizer(
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
            cls=Qwen2VLCausalLMPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[1, 3, 4, 2, 5, 6, 0, 0]],
                    "padding_mask": [[1, 1, 1, 1, 1, 1, 0, 0]],
                },
                [[3, 4, 2, 5, 6, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 0, 0, 0]],
            ),
        )

    def test_with_end_token(self):
        input_data = ["airplane at airport"] * 4

        preprocessor = Qwen2VLCausalLMPreprocessor(
            **self.init_kwargs,
            add_end_token=True,
        )
        x, y, sw = preprocessor(input_data)
        self.assertAllEqual(x["token_ids"], [[1, 3, 4, 2, 5, 6, 0, 0]] * 4)
        self.assertAllEqual(x["padding_mask"], [[1, 1, 1, 1, 1, 1, 0, 0]] * 4)
        self.assertAllEqual(y, [[3, 4, 2, 5, 6, 0, 0, 0]] * 4)
        self.assertAllEqual(sw, [[1, 1, 1, 1, 1, 0, 0, 0]] * 4)

    def test_generate_preprocess(self):
        input_data = "airplane at airport"
        preprocessor = Qwen2VLCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_preprocess(input_data)
        self.assertAllEqual(x["token_ids"], [1, 3, 4, 2, 5, 0, 0, 0])
        self.assertAllEqual(x["padding_mask"], [1, 1, 1, 1, 1, 0, 0, 0])

    def test_generate_postprocess(self):
        input_data = {
            "token_ids": [1, 3, 4, 2, 5, 6, 0, 0],
            "padding_mask": [1, 1, 1, 1, 1, 1, 0, 0],
        }
        preprocessor = Qwen2VLCausalLMPreprocessor(**self.init_kwargs)
        x = preprocessor.generate_postprocess(input_data)
        self.assertAllEqual(x, "airplane at airport")

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Qwen2VLCausalLMPreprocessor.presets:
            self.run_preset_test(
                cls=Qwen2VLCausalLMPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )

    def test_generate_preprocess_with_image(self):
        """Test that generate_preprocess handles image inputs."""
        image_converter = Qwen2VLImageConverter()
        preprocessor = Qwen2VLCausalLMPreprocessor(
            tokenizer=self.tokenizer,
            image_converter=image_converter,
            sequence_length=64,
            spatial_merge_size=2,
        )
        # Create a small dummy image (56x56 is the minimum after smart_resize).
        dummy_image = np.random.randint(0, 255, (56, 56, 3), dtype=np.uint8)
        x = {"text": "Describe this image", "images": dummy_image}
        result = preprocessor.generate_preprocess(x)

        self.assertIn("token_ids", result)
        self.assertIn("padding_mask", result)
        self.assertIn("patch_values", result)
        self.assertIn("image_grid_thw", result)
        self.assertEqual(len(result["token_ids"]), 64)
        self.assertEqual(len(result["padding_mask"]), 64)
        self.assertIsNotNone(result["patch_values"])
        self.assertIsNotNone(result["image_grid_thw"])
