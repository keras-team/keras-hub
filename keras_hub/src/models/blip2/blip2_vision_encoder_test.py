"""Tests for BLIP-2 vision encoder."""

import numpy as np

from keras_hub.src.models.blip2.blip2_vision_encoder import Blip2VisionEncoder
from keras_hub.src.tests.test_case import TestCase


class Blip2VisionEncoderTest(TestCase):
    def setUp(self):
        self.image_size = 32
        self.patch_size = 8
        self.init_kwargs = {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            "use_patch_bias": True,
            "use_class_token": True,
            "use_mha_bias": True,
            "use_mlp_bias": True,
            "dropout_rate": 0.0,
            "layer_norm_epsilon": 1e-6,
        }
        self.input_data = np.ones(
            (2, self.image_size, self.image_size, 3), dtype="float32"
        )

    def test_vision_encoder_basics(self):
        encoder = Blip2VisionEncoder(**self.init_kwargs)
        self.run_serialization_test(encoder)
        output = encoder(self.input_data)
        num_patches = (self.image_size // self.patch_size) ** 2
        self.assertEqual(output.shape, (2, num_patches + 1, 8))

    def test_no_class_token(self):
        kwargs = {**self.init_kwargs, "use_class_token": False}
        encoder = Blip2VisionEncoder(**kwargs)
        output = encoder(self.input_data)
        num_patches = (self.image_size // self.patch_size) ** 2
        self.assertEqual(output.shape, (2, num_patches, 8))

    def test_patch_size_controls_sequence_length(self):
        patch_size = 4
        kwargs = {**self.init_kwargs, "patch_size": patch_size}
        encoder = Blip2VisionEncoder(**kwargs)
        output = encoder(self.input_data)
        num_patches = (self.image_size // patch_size) ** 2
        self.assertEqual(output.shape, (2, num_patches + 1, 8))

    def test_invalid_image_size_raises(self):
        with self.assertRaises(ValueError):
            Blip2VisionEncoder(**{**self.init_kwargs, "image_size": 33})
