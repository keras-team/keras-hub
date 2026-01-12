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
        }
        self.input_data = np.random.uniform(size=(2, 28, 28, 3)).astype(
            "float32"
        )

    def test_encoder_basics(self):
        self.run_layer_test(
            cls=Llama3VisionEncoder,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 4, 32),
            expected_num_trainable_weights=37,
            run_precision_checks=False,
        )

    def test_encoder_two_stage(self):
        """Test two-stage encoder architecture."""
        init_kwargs = dict(self.init_kwargs)
        init_kwargs["num_layers"] = 4
        init_kwargs["local_layers"] = 3
        init_kwargs["global_layers"] = 1

        encoder = Llama3VisionEncoder(**init_kwargs)
        outputs = encoder(self.input_data)

        self.assertEqual(outputs.shape, (2, 4, 32))
        self.assertTrue(encoder.is_two_stage)
        self.assertEqual(len(encoder.local_transformer_layers), 3)
        self.assertEqual(len(encoder.global_transformer_layers), 1)

    def test_serialization(self):
        """Test config serialization."""
        init_kwargs = dict(self.init_kwargs)
        init_kwargs["num_layers"] = 4
        init_kwargs["local_layers"] = 3
        init_kwargs["global_layers"] = 1

        encoder = Llama3VisionEncoder(**init_kwargs)
        config = encoder.get_config()

        self.assertEqual(config["hidden_dim"], 32)
        self.assertEqual(config["local_layers"], 3)
        self.assertEqual(config["global_layers"], 1)

        new_encoder = Llama3VisionEncoder(**config)
        self.assertTrue(new_encoder.is_two_stage)

    def test_freeze_local_encoder(self):
        """Test freezing local encoder in two-stage mode."""
        init_kwargs = dict(self.init_kwargs)
        init_kwargs["num_layers"] = 4
        init_kwargs["local_layers"] = 3
        init_kwargs["global_layers"] = 1

        encoder = Llama3VisionEncoder(**init_kwargs)
        encoder.freeze_local_encoder()

        self.assertFalse(encoder.patch_embedding.trainable)
        self.assertFalse(encoder.position_embedding.trainable)
        for layer in encoder.local_transformer_layers:
            self.assertFalse(layer.trainable)

    def test_freeze_local_encoder_single_stage_raises(self):
        """Test freeze_local_encoder raises error in single-stage mode."""
        encoder = Llama3VisionEncoder(**self.init_kwargs)
        with self.assertRaisesRegex(ValueError, "requires two-stage mode"):
            encoder.freeze_local_encoder()

    def test_freeze_global_encoder(self):
        """Test freezing global encoder in two-stage mode."""
        init_kwargs = dict(self.init_kwargs)
        init_kwargs["num_layers"] = 4
        init_kwargs["local_layers"] = 3
        init_kwargs["global_layers"] = 1

        encoder = Llama3VisionEncoder(**init_kwargs)
        encoder.freeze_global_encoder()

        self.assertFalse(encoder.layer_norm.trainable)
        for layer in encoder.global_transformer_layers:
            self.assertFalse(layer.trainable)

    def test_freeze_global_encoder_single_stage_raises(self):
        """Test freeze_global_encoder raises error in single-stage mode."""
        encoder = Llama3VisionEncoder(**self.init_kwargs)
        with self.assertRaisesRegex(ValueError, "requires two-stage mode"):
            encoder.freeze_global_encoder()

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
        self.assertTrue(encoder.position_embedding.trainable)
        for layer in encoder.transformer_layers:
            self.assertTrue(layer.trainable)
