import numpy as np

from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone
from keras_hub.src.tests.test_case import TestCase


class Qwen3ASRBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 512,
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
            "audio_token_id": 200,
            "n_window": 40,
            "max_source_positions": 100,
            "layer_norm_epsilon": 1e-6,
            "dtype": "float32",
        }
        # 80 time steps -> 10 after 3x stride-2 conv downsampling.
        # token_ids: 10 audio placeholders (ID=200) + 7 text tokens.
        self.input_data = {
            "audio_features": np.random.uniform(size=(2, 80, 32)).astype(
                "float32"
            ),
            "token_ids": np.array(
                [
                    [200] * 10 + [1, 2, 3, 4, 5, 0, 0],
                    [200] * 10 + [1, 2, 3, 4, 5, 6, 7],
                ],
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
        # 10 audio placeholders + 7 text tokens = 17 total.
        self.assertEqual(output.shape, (2, 17, 32))

    def test_serialization(self):
        model = Qwen3ASRBackbone(**self.init_kwargs)
        config = model.get_config()
        restored = Qwen3ASRBackbone.from_config(config)
        self.assertEqual(restored.get_config(), config)

    def test_variable_audio_length(self):
        """Verify the backbone handles different audio lengths."""
        model = Qwen3ASRBackbone(**self.init_kwargs)
        # 160 time steps -> 20 after downsampling.
        # 20 audio placeholders + 3 text tokens.
        input_data = {
            "audio_features": np.random.uniform(size=(1, 160, 32)).astype(
                "float32"
            ),
            "token_ids": np.array([[200] * 20 + [1, 2, 3]], dtype="int32"),
            "padding_mask": np.array([[1] * 20 + [1, 1, 1]], dtype="int32"),
        }
        output = model(input_data)
        self.assertEqual(output.shape, (1, 23, 32))
