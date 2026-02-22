import numpy as np

from keras_hub.src.models.qwen3_omni.qwen3_omni_audio_encoder import (
    Qwen3OmniAudioEncoder,
)
from keras_hub.src.models.qwen3_omni.qwen3_omni_audio_encoder import (
    Qwen3OmniAudioEncoderLayer,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3OmniAudioEncoderTest(TestCase):
    def test_encoder_layer_output_shape(self):
        layer = Qwen3OmniAudioEncoderLayer(
            embed_dim=32,
            num_heads=4,
            ffn_dim=64,
            dtype="float32",
        )
        hidden_states = np.random.rand(1, 10, 32).astype("float32")
        output = layer(hidden_states)
        self.assertEqual(output.shape, (1, 10, 32))

    def test_encoder_output_shape(self):
        encoder = Qwen3OmniAudioEncoder(
            num_mel_bins=80,
            d_model=32,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            output_dim=16,
            max_source_positions=100,
            scale_embedding=False,
            dtype="float32",
        )
        input_features = np.random.rand(1, 160, 80).astype("float32")
        output = encoder({"input_features": input_features})
        self.assertEqual(output.shape[-1], 16)

    def test_encoder_config_roundtrip(self):
        encoder = Qwen3OmniAudioEncoder(
            num_mel_bins=80,
            d_model=32,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            output_dim=16,
            max_source_positions=100,
            scale_embedding=False,
            dtype="float32",
        )
        config = encoder.get_config()
        restored = Qwen3OmniAudioEncoder.from_config(config)
        self.assertEqual(restored.d_model, 32)
        self.assertEqual(restored.output_dim, 16)
        self.assertEqual(len(restored.encoder_layers), 2)
