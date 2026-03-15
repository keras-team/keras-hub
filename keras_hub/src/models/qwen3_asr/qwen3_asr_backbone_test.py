import numpy as np

from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone
from keras_hub.src.tests.test_case import TestCase


class Qwen3ASRBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 256,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "hidden_dim": 32,
            "intermediate_dim": 64,
            "num_mel_bins": 32,
            "encoder_d_model": 32,
            "encoder_num_layers": 1,
            "encoder_attention_heads": 4,
            "encoder_ffn_dim": 64,
            "downsample_hidden_size": 16,
            "layer_norm_epsilon": 1e-6,
            "dtype": "float32",
        }
        # 80 time steps -> 10 after 3x stride-2 conv downsampling.
        self.input_data = {
            "audio_features": np.random.uniform(size=(2, 80, 32)).astype(
                "float32"
            ),
            "token_ids": np.array(
                [[1, 2, 3, 4, 5, 0, 0], [1, 2, 3, 4, 5, 6, 7]],
                dtype="int32",
            ),
            "padding_mask": np.array(
                [
                    [1] * 10 + [1, 1, 1, 1, 1, 0, 0],
                    [1] * 10 + [1, 1, 1, 1, 1, 1, 1],
                ],
                dtype="int32",
            ),
        }

    def test_output_shape(self):
        model = Qwen3ASRBackbone(**self.init_kwargs)
        output = model(self.input_data)
        # 10 audio tokens + 7 text tokens = 17.
        self.assertEqual(output.shape, (2, 17, 32))

    def test_serialization(self):
        model = Qwen3ASRBackbone(**self.init_kwargs)
        config = model.get_config()
        restored = Qwen3ASRBackbone.from_config(config)
        self.assertEqual(restored.get_config(), config)

    def test_1_7b_config_output_shape(self):
        """Smoke test with 1.7B config values (single layer for speed)."""
        model = Qwen3ASRBackbone(
            vocabulary_size=151936,
            num_layers=1,
            num_query_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            hidden_dim=2048,
            intermediate_dim=6144,
            num_mel_bins=128,
            encoder_d_model=1024,
            encoder_num_layers=1,
            encoder_attention_heads=16,
            encoder_ffn_dim=4096,
            downsample_hidden_size=480,
            dtype="float32",
        )
        input_data = {
            "audio_features": np.random.uniform(size=(1, 800, 128)).astype(
                "float32"
            ),
            "token_ids": np.array([[1, 2, 3]], dtype="int32"),
            "padding_mask": np.array([[1] * 100 + [1, 1, 1]], dtype="int32"),
        }
        output = model(input_data)
        # 800 -> 400 -> 200 -> 100 audio tokens + 3 text tokens = 103.
        self.assertEqual(output.shape, (1, 103, 2048))

    def test_variable_audio_length(self):
        """Verify the backbone handles different audio lengths."""
        model = Qwen3ASRBackbone(**self.init_kwargs)
        # 160 time steps -> 20 after downsampling.
        input_data = {
            "audio_features": np.random.uniform(size=(1, 160, 32)).astype(
                "float32"
            ),
            "token_ids": np.array([[1, 2, 3]], dtype="int32"),
            "padding_mask": np.array([[1] * 20 + [1, 1, 1]], dtype="int32"),
        }
        output = model(input_data)
        self.assertEqual(output.shape, (1, 23, 32))
