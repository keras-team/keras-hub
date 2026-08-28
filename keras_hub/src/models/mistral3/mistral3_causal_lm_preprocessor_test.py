import numpy as np

from keras_hub.src.models.mistral3.mistral3_causal_lm_preprocessor import (
    Mistral3CausalLMPreprocessor,
)
from keras_hub.src.models.mistral3.mistral3_image_converter import (
    Mistral3ImageConverter,
)
from keras_hub.src.models.mistral3.mistral3_tokenizer import Mistral3Tokenizer
from keras_hub.src.models.mistral3.mistral3_tokenizer_test import (
    _tekken_vision_init_kwargs,
)
from keras_hub.src.tests.test_case import TestCase


class Mistral3CausalLMPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = Mistral3Tokenizer(**_tekken_vision_init_kwargs())
        self.image_converter = Mistral3ImageConverter(
            longest_edge=16, patch_size=4, spatial_merge_size=1
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "image_converter": self.image_converter,
            "sequence_length": 32,
            "spatial_merge_size": 1,
        }

    def test_generate_preprocess_with_images(self):
        preprocessor = Mistral3CausalLMPreprocessor(**self.init_kwargs)
        image = np.zeros((8, 8, 3), dtype="float32")
        x = preprocessor.generate_preprocess(
            {"prompts": "the [IMG] quick", "images": [image]}
        )
        for key in (
            "token_ids",
            "padding_mask",
            "pixel_values",
            "image_sizes",
            "placeholder_indices",
        ):
            self.assertIn(key, x)
        # An 8x8 image with `patch_size=4`, `spatial_merge_size=1` expands
        # to a 2x2 grid of placeholder tokens.
        token_ids = np.array(x["token_ids"])
        num_placeholders = int(
            np.sum(
                token_ids == preprocessor.tokenizer.image_placeholder_token_id
            )
        )
        self.assertEqual(num_placeholders, 4)
