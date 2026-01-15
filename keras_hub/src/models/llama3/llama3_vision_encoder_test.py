import numpy as np

from keras_hub.src.models.llama3.llama3_vision_encoder import (
    Llama3VisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class Llama3VisionEncoderTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "hidden_dim": 32,
            "num_layers": 2,
            "num_heads": 2,
            "intermediate_dim": 64,
            "patch_size": 14,
            "image_size": 28,
            "global_layers": 1,
        }
        self.input_data = np.random.uniform(size=(2, 28, 28, 3)).astype(
            "float32"
        )

    def test_encoder_basics(self):
        encoder = Llama3VisionEncoder(**self.init_kwargs)
        outputs = encoder(self.input_data)
        # Output shape: (batch, num_patches + 1 for CLS, hidden_dim)
        # num_patches = (28/14)^2 = 4, so total = 5
        self.assertEqual(outputs.shape, (2, 5, 32))

    def test_encoder_with_global_layers(self):
        """Test encoder with specified global layers."""
        init_kwargs = dict(self.init_kwargs)
        init_kwargs["num_layers"] = 3
        init_kwargs["global_layers"] = 2

        encoder = Llama3VisionEncoder(**init_kwargs)
        outputs = encoder(self.input_data)

        self.assertEqual(outputs.shape, (2, 5, 32))
        self.assertEqual(len(encoder.transformer_layers), 3)
        self.assertEqual(len(encoder.global_transformer_layers), 2)

    def test_serialization(self):
        """Test config serialization."""
        encoder = Llama3VisionEncoder(**self.init_kwargs)
        config = encoder.get_config()

        self.assertEqual(config["hidden_dim"], 32)
        self.assertEqual(config["num_layers"], 2)
        self.assertEqual(config["global_layers"], 1)

        new_encoder = Llama3VisionEncoder.from_config(config)
        self.assertEqual(len(new_encoder.transformer_layers), 2)

    def test_freeze_local_encoder(self):
        """Test freezing local encoder layers."""
        encoder = Llama3VisionEncoder(**self.init_kwargs)
        encoder.freeze_local_encoder()

        self.assertFalse(encoder.patch_embedding.trainable)
        for layer in encoder.transformer_layers:
            self.assertFalse(layer.trainable)

    def test_freeze_global_encoder(self):
        """Test freezing global encoder layers."""
        encoder = Llama3VisionEncoder(**self.init_kwargs)
        encoder.freeze_global_encoder()

        self.assertFalse(encoder.layernorm_pre.trainable)
        self.assertFalse(encoder.layernorm_post.trainable)
        for layer in encoder.global_transformer_layers:
            self.assertFalse(layer.trainable)

    def test_freeze_all(self):
        """Test freezing entire encoder."""
        encoder = Llama3VisionEncoder(**self.init_kwargs)
        self.assertTrue(encoder.trainable)
        encoder.freeze_all()
        self.assertFalse(encoder.trainable)

    def test_unfreeze_all(self):
        """Test unfreezing all components."""
        encoder = Llama3VisionEncoder(**self.init_kwargs)
        encoder.freeze_all()
        encoder.unfreeze_all()

        self.assertTrue(encoder.trainable)
        self.assertTrue(encoder.patch_embedding.trainable)
