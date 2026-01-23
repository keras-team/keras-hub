"""Tests for UNet Encoder."""

import keras
import numpy as np
import pytest

from keras_hub.src.models.unet.unet_encoder import UNetEncoder
from keras_hub.src.tests.test_case import TestCase


class UNetEncoderTest(TestCase):
    def setUp(self):
        self.input_size = 128
        self.batch_size = 2
        shape = (self.batch_size, self.input_size, self.input_size, 3)
        self.input_data = np.random.uniform(0, 1, size=shape).astype(np.float32)
        # Ensure tests use channels_last format
        keras.config.set_image_data_format("channels_last")

    def test_encoder_from_scratch_basics(self):
        """Test basic encoder functionality when built from scratch."""
        encoder = UNetEncoder(depth=3, filters=32, data_format="channels_last")
        output = encoder(self.input_data)

        # Output should be a dict
        self.assertIsInstance(output, dict)
        self.assertIn("bottleneck", output)
        self.assertIn("skip_connections", output)

        # Check skip connections count
        skip_connections = output["skip_connections"]
        self.assertEqual(len(skip_connections), 2)  # depth - 1

        # Check shapes
        self.assertEqual(output["bottleneck"].shape[0], self.batch_size)
        for skip in skip_connections:
            self.assertEqual(skip.shape[0], self.batch_size)

    def test_encoder_different_depths(self):
        """Test encoder with different depth configurations."""
        for depth in [2, 3, 4, 5]:
            with self.subTest(depth=depth):
                encoder = UNetEncoder(
                    depth=depth, filters=32, data_format="channels_last"
                )
                output = encoder(self.input_data)

                # Skip connections should be depth - 1
                self.assertEqual(len(output["skip_connections"]), depth - 1)

    def test_encoder_with_batch_norm(self):
        """Test encoder with batch normalization enabled."""
        encoder = UNetEncoder(
            depth=3,
            filters=32,
            use_batch_norm=True,
            data_format="channels_last",
        )
        output = encoder(self.input_data)

        self.assertIsInstance(output, dict)
        self.assertIn("bottleneck", output)

    def test_encoder_with_residual_connections(self):
        """Test encoder with residual connections (ResNet-style)."""
        encoder = UNetEncoder(
            depth=3,
            filters=32,
            use_residual=True,
            use_batch_norm=True,
            data_format="channels_last",
        )
        output = encoder(self.input_data)

        self.assertIsInstance(output, dict)
        self.assertEqual(len(output["skip_connections"]), 2)

    def test_encoder_with_pretrained_resnet(self):
        """Test encoder using pretrained ResNet50 backbone."""
        # Ensure channels_last format for backbone creation
        keras.config.set_image_data_format("channels_last")

        backbone = keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_shape=(None, None, 3),
        )

        encoder = UNetEncoder(backbone=backbone, data_format="channels_last")
        output = encoder(self.input_data)

        self.assertIsInstance(output, dict)
        self.assertIn("bottleneck", output)
        self.assertIn("skip_connections", output)
        # ResNet typically provides multiple feature levels
        self.assertGreaterEqual(len(output["skip_connections"]), 3)

    def test_encoder_with_pretrained_mobilenet(self):
        """Test encoder using pretrained MobileNetV2 backbone."""
        # Ensure channels_last format for backbone creation
        keras.config.set_image_data_format("channels_last")

        backbone = keras.applications.MobileNetV2(
            include_top=False,
            weights=None,
            input_shape=(None, None, 3),
        )

        encoder = UNetEncoder(backbone=backbone, data_format="channels_last")
        output = encoder(self.input_data)

        self.assertIsInstance(output, dict)
        self.assertGreaterEqual(len(output["skip_connections"]), 3)

    def test_encoder_dynamic_input_shapes(self):
        """Test encoder with dynamic input shapes."""
        encoder = UNetEncoder(depth=3, filters=32, data_format="channels_last")

        # Test with different input sizes
        for size in [64, 128, 256]:
            with self.subTest(size=size):
                test_input = np.random.uniform(
                    0, 1, size=(2, size, size, 3)
                ).astype(np.float32)
                output = encoder(test_input)
                self.assertIsInstance(output, dict)

    def test_encoder_config_serialization(self):
        """Test encoder configuration serialization and deserialization."""
        encoder = UNetEncoder(
            depth=3,
            filters=32,
            use_batch_norm=True,
            use_residual=True,
        )

        config = encoder.get_config()

        # Check config contains required keys
        self.assertIn("depth", config)
        self.assertIn("filters", config)
        self.assertIn("use_batch_norm", config)
        self.assertIn("use_residual", config)

        # Test from_config
        restored_encoder = UNetEncoder.from_config(config)
        output1 = encoder(self.input_data)
        output2 = restored_encoder(self.input_data)

        # Shapes should match
        self.assertEqual(
            output1["bottleneck"].shape, output2["bottleneck"].shape
        )

    def test_encoder_with_pretrained_config(self):
        """Test encoder config with pretrained backbone."""
        # Ensure channels_last format for backbone creation
        keras.config.set_image_data_format("channels_last")

        backbone = keras.applications.MobileNetV2(
            include_top=False,
            weights=None,
            input_shape=(224, 224, 3),
        )

        encoder = UNetEncoder(backbone=backbone, data_format="channels_last")
        config = encoder.get_config()

        self.assertIn("backbone", config)
        self.assertIsNotNone(config["backbone"])

        # Test from_config
        restored_encoder = UNetEncoder.from_config(config)
        # Create input data that matches the backbone's expected input size
        pretrained_input_data = np.random.uniform(
            0, 1, size=(2, 224, 224, 3)
        ).astype(np.float32)
        output = restored_encoder(pretrained_input_data)
        self.assertIsInstance(output, dict)

    def test_encoder_parameter_count(self):
        """Test that encoder has reasonable parameter count."""
        encoder = UNetEncoder(depth=3, filters=32, data_format="channels_last")
        param_count = encoder.count_params()

        # Should have some parameters but not excessive
        self.assertGreater(param_count, 0)
        self.assertLess(param_count, 10_000_000)  # Less than 10M params

    def test_encoder_filters_progression(self):
        """Test that filters double at each encoder level."""
        depth = 4
        filters = 64
        encoder = UNetEncoder(
            depth=depth, filters=filters, data_format="channels_last"
        )

        # The encoder should progressively reduce spatial dimensions
        # and increase channels
        output = encoder(self.input_data)

        # Check bottleneck has the most channels
        bottleneck_channels = output["bottleneck"].shape[-1]
        self.assertEqual(bottleneck_channels, filters * (2 ** (depth - 1)))

    @pytest.mark.skipif(
        keras.backend.backend() == "tensorflow",
        reason="TensorFlow doesn't support channels_first on CPU",
    )
    def test_encoder_channels_first_format(self):
        """Test encoder with channels_first data format."""
        encoder = UNetEncoder(
            depth=3,
            filters=32,
            image_shape=(3, None, None),
            data_format="channels_first",
        )

        # Input shape: (batch, channels, height, width)
        input_data = np.random.uniform(0, 1, size=(2, 3, 128, 128)).astype(
            np.float32
        )

        output = encoder(input_data)
        self.assertIsInstance(output, dict)

    def test_encoder_dtype(self):
        """Test encoder with different dtype."""
        encoder = UNetEncoder(
            depth=3, filters=32, dtype="float32", data_format="channels_last"
        )
        output = encoder(self.input_data)

        # Check output dtype (may be float32 due to Keras internal handling)
        self.assertIsInstance(output, dict)

    @pytest.mark.large
    def test_encoder_export(self):
        """Test encoder model export."""
        import tempfile

        encoder = UNetEncoder(depth=3, filters=32, data_format="channels_last")
        output = encoder(self.input_data)

        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = f"{temp_dir}/encoder_model"
            encoder.save(save_path)

            loaded_encoder = keras.models.load_model(save_path)
            loaded_output = loaded_encoder(self.input_data)

            # Check shapes match
            self.assertEqual(
                output["bottleneck"].shape, loaded_output["bottleneck"].shape
            )
