import keras

from keras_hub.src.models.qwen3_asr.qwen3_asr_encoder import Qwen3ASREncoder
from keras_hub.src.models.qwen3_asr.qwen3_asr_encoder import (
    Qwen3ASREncoderLayer,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3ASREncoderLayerTest(TestCase):
    def test_output_shape(self):
        self.run_layer_test(
            cls=Qwen3ASREncoderLayer,
            init_kwargs={
                "d_model": 64,
                "num_heads": 4,
                "ffn_dim": 128,
            },
            input_data=keras.random.uniform((2, 10, 64), dtype="float32"),
            expected_output_shape=(2, 10, 64),
            # q, k, v, out projections: 4 kernels + 4 biases = 8
            # fc1, fc2: 2 kernels + 2 biases = 4
            # 2 layer norms: 2 gamma + 2 beta = 4
            expected_num_trainable_weights=16,
            expected_num_non_trainable_weights=0,
            run_precision_checks=False,
        )

    def test_with_dropout(self):
        self.run_layer_test(
            cls=Qwen3ASREncoderLayer,
            init_kwargs={
                "d_model": 64,
                "num_heads": 4,
                "ffn_dim": 128,
                "dropout": 0.1,
            },
            input_data=keras.random.uniform((2, 10, 64), dtype="float32"),
            expected_output_shape=(2, 10, 64),
            expected_num_trainable_weights=16,
            expected_num_non_trainable_weights=0,
            # Dropout layer creates a seed generator state variable.
            expected_num_non_trainable_variables=1,
            run_precision_checks=False,
        )


class Qwen3ASREncoderTest(TestCase):
    def test_output_shape(self):
        encoder = Qwen3ASREncoder(
            num_mel_bins=32,
            d_model=64,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=128,
            downsample_hidden_size=16,
        )
        x = keras.random.uniform((2, 80, 32), dtype="float32")
        output = encoder(x)
        # time: 80 -> 40 -> 20 -> 10 after three stride-2 convolutions.
        self.assertEqual(output.shape, (2, 10, 64))

    def test_output_shape_with_projection(self):
        encoder = Qwen3ASREncoder(
            num_mel_bins=32,
            d_model=64,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=128,
            downsample_hidden_size=16,
            output_dim=96,
        )
        x = keras.random.uniform((2, 80, 32), dtype="float32")
        output = encoder(x)
        self.assertEqual(output.shape, (2, 10, 96))

    def test_compute_output_shape(self):
        encoder = Qwen3ASREncoder(
            num_mel_bins=128,
            d_model=64,
            encoder_layers=1,
            encoder_attention_heads=4,
            encoder_ffn_dim=128,
            downsample_hidden_size=16,
        )
        # 800 -> 400 -> 200 -> 100
        shape = encoder.compute_output_shape((2, 800, 128))
        self.assertEqual(shape, (2, 100, 64))

    def test_compute_output_shape_with_projection(self):
        encoder = Qwen3ASREncoder(
            num_mel_bins=128,
            d_model=64,
            encoder_layers=1,
            encoder_attention_heads=4,
            encoder_ffn_dim=128,
            downsample_hidden_size=16,
            output_dim=96,
        )
        shape = encoder.compute_output_shape((2, 800, 128))
        self.assertEqual(shape, (2, 100, 96))

    def test_serialization(self):
        encoder = Qwen3ASREncoder(
            num_mel_bins=32,
            d_model=64,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=128,
            downsample_hidden_size=16,
            output_dim=96,
            dropout=0.1,
        )
        config = encoder.get_config()
        restored = Qwen3ASREncoder.from_config(config)
        self.assertEqual(restored.get_config(), config)

    def test_1_7b_config_output_shape(self):
        """Verify output shape with the 1.7B model config values."""
        encoder = Qwen3ASREncoder(
            num_mel_bins=128,
            d_model=1024,
            encoder_layers=1,
            encoder_attention_heads=16,
            encoder_ffn_dim=4096,
            downsample_hidden_size=480,
            output_dim=2048,
        )
        # Use a single-layer encoder to keep the test fast.
        x = keras.random.uniform((1, 800, 128), dtype="float32")
        output = encoder(x)
        # 800 -> 400 -> 200 -> 100
        self.assertEqual(output.shape, (1, 100, 2048))
