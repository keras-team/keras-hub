"""Tests for UNet Decoder."""

import keras
import numpy as np
import pytest

from keras_hub.src.models.unet.unet_decoder import UNetDecoder
from keras_hub.src.models.unet.unet_encoder import UNetEncoder
from keras_hub.src.tests.test_case import TestCase


class UNetDecoderTest(TestCase):
    def setUp(self):
        self.input_size = 128
        self.batch_size = 2
        self.input_data = np.random.uniform(
            0, 1, size=(self.batch_size, self.input_size, self.input_size, 3)
        ).astype(np.float32)

        # Create encoder to generate features for decoder
        self.encoder = UNetEncoder(depth=3, filters=32)
        self.encoder_output = self.encoder(self.input_data)

    def test_decoder_basics(self):
        """Test basic decoder functionality."""
        decoder = UNetDecoder(filters=32)
        output = decoder(self.encoder_output)

        # Output should be a tensor
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(
            len(output.shape), 4
        )  # (batch, height, width, channels)

    def test_decoder_with_transpose_upsampling(self):
        """Test decoder with transpose convolution upsampling."""
        decoder = UNetDecoder(
            filters=32,
            upsampling_strategy="transpose",
        )
        output = decoder(self.encoder_output)
        self.assertEqual(output.shape[0], self.batch_size)

    def test_decoder_with_interpolation_upsampling(self):
        """Test decoder with bilinear interpolation upsampling."""
        decoder = UNetDecoder(
            filters=32,
            upsampling_strategy="interpolation",
        )
        output = decoder(self.encoder_output)
        self.assertEqual(output.shape[0], self.batch_size)

    def test_decoder_with_batch_norm(self):
        """Test decoder with batch normalization."""
        decoder = UNetDecoder(
            filters=32,
            use_batch_norm=True,
        )
        output = decoder(self.encoder_output)
        self.assertEqual(output.shape[0], self.batch_size)

    def test_decoder_with_dropout(self):
        """Test decoder with dropout enabled."""
        decoder = UNetDecoder(
            filters=32,
            use_dropout=True,
            dropout_rate=0.5,
        )
        output = decoder(self.encoder_output, training=True)
        self.assertEqual(output.shape[0], self.batch_size)

    def test_decoder_with_attention_gates(self):
        """Test decoder with attention gates on skip connections."""
        decoder = UNetDecoder(
            filters=32,
            use_attention=True,
        )
        output = decoder(self.encoder_output)
        self.assertEqual(output.shape[0], self.batch_size)

    def test_decoder_with_all_features(self):
        """Test decoder with all advanced features enabled."""
        decoder = UNetDecoder(
            filters=32,
            use_batch_norm=True,
            use_dropout=True,
            dropout_rate=0.3,
            use_attention=True,
            upsampling_strategy="interpolation",
        )
        output = decoder(self.encoder_output)
        self.assertEqual(output.shape[0], self.batch_size)

    def test_decoder_with_pretrained_encoder(self):
        """Test decoder with features from pretrained encoder."""
        # Create encoder with pretrained backbone
        backbone = keras.applications.MobileNetV2(
            include_top=False,
            weights=None,
            input_shape=(None, None, 3),
        )
        encoder = UNetEncoder(backbone=backbone)
        encoder_output = encoder(self.input_data)

        # Create decoder (filters=None to infer from encoder)
        decoder = UNetDecoder(
            filters=None,
            use_attention=True,
        )
        output = decoder(encoder_output)

        self.assertEqual(output.shape[0], self.batch_size)

    def test_decoder_config_serialization(self):
        """Test decoder configuration serialization."""
        decoder = UNetDecoder(
            filters=32,
            use_batch_norm=True,
            use_dropout=True,
            dropout_rate=0.3,
            upsampling_strategy="interpolation",
            use_attention=True,
        )

        # Build the decoder first
        _ = decoder(self.encoder_output)

        config = decoder.get_config()

        # Check config contains required keys
        self.assertIn("filters", config)
        self.assertIn("use_batch_norm", config)
        self.assertIn("use_dropout", config)
        self.assertIn("dropout_rate", config)
        self.assertIn("upsampling_strategy", config)
        self.assertIn("use_attention", config)

    def test_decoder_invalid_upsampling_strategy(self):
        """Test that invalid upsampling strategy raises error."""
        with self.assertRaises(ValueError):
            UNetDecoder(filters=32, upsampling_strategy="invalid")

    def test_decoder_parameter_count(self):
        """Test that decoder has reasonable parameter count."""
        decoder = UNetDecoder(filters=32)
        _ = decoder(self.encoder_output)

        param_count = decoder.count_params()
        self.assertGreater(param_count, 0)

    def test_decoder_with_different_encoder_depths(self):
        """Test decoder with encoders of different depths."""
        for depth in [2, 3, 4]:
            with self.subTest(depth=depth):
                encoder = UNetEncoder(depth=depth, filters=32)
                encoder_output = encoder(self.input_data)

                decoder = UNetDecoder(filters=32)
                output = decoder(encoder_output)

                self.assertEqual(output.shape[0], self.batch_size)

    def test_decoder_training_mode(self):
        """Test decoder in training vs inference mode (for dropout)."""
        decoder = UNetDecoder(
            filters=32,
            use_dropout=True,
            dropout_rate=0.5,
        )

        # Training mode
        output_train = decoder(self.encoder_output, training=True)

        # Inference mode
        output_infer = decoder(self.encoder_output, training=False)

        # Shapes should match
        self.assertEqual(output_train.shape, output_infer.shape)

    @pytest.mark.skipif(
        keras.backend.backend() == "tensorflow",
        reason="TensorFlow doesn't support channels_first on CPU",
    )
    def test_decoder_channels_first(self):
        """Test decoder with channels_first data format."""
        # Create encoder with channels_first
        encoder = UNetEncoder(
            depth=3,
            filters=32,
            image_shape=(3, None, None),
            data_format="channels_first",
        )

        input_data_cf = np.random.uniform(0, 1, size=(2, 3, 128, 128)).astype(
            np.float32
        )
        encoder_output = encoder(input_data_cf)

        # Create decoder with channels_first
        decoder = UNetDecoder(
            filters=32,
            data_format="channels_first",
        )
        output = decoder(encoder_output)

        # Output should be (batch, channels, height, width)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(output.shape[1], 32)  # channels

    def test_encoder_decoder_pipeline(self):
        """Test full encoder-decoder pipeline."""
        # Create encoder
        encoder = UNetEncoder(
            depth=4,
            filters=64,
            use_batch_norm=True,
            use_residual=True,
        )

        # Create matching decoder
        decoder = UNetDecoder(
            filters=64,
            use_batch_norm=True,
            use_attention=True,
            upsampling_strategy="interpolation",
        )

        # Forward pass
        encoder_output = encoder(self.input_data)
        decoder_output = decoder(encoder_output)

        # Check output shape has same spatial dimensions as input
        self.assertEqual(decoder_output.shape[1], self.input_size)
        self.assertEqual(decoder_output.shape[2], self.input_size)
