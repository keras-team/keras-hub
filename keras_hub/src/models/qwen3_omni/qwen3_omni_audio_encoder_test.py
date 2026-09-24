import numpy as np
from keras import ops

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
        self.assertEqual(restored.encoder_layers, 2)

    def test_window_aftercnn_uses_hf_defaults(self):
        # HF defaults: chunk_length=n_window*2=200 -> ceil(200/8)=25 after
        # a single-pass (non-chunked) Keras CNN of three stride-2
        # padding=same convs. ratio = n_window_infer // chunk_length =
        # 400 // 200 = 2. So window_aftercnn = 25 * 2 = 50.
        encoder = Qwen3OmniAudioEncoder(
            num_mel_bins=80,
            d_model=32,
            encoder_layers=1,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            output_dim=16,
            max_source_positions=100,
            scale_embedding=False,
            dtype="float32",
        )
        self.assertEqual(encoder.n_window, 100)
        self.assertEqual(encoder.n_window_infer, 400)
        self.assertEqual(encoder.conv_chunksize, 500)
        self.assertEqual(encoder.window_aftercnn, 50)

    def test_window_mask_block_diagonal(self):
        # Small toy window so we can assert the block structure without
        # running a 200-frame mel through the full encoder.
        encoder = Qwen3OmniAudioEncoder(
            num_mel_bins=80,
            d_model=32,
            encoder_layers=1,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            output_dim=16,
            max_source_positions=100,
            scale_embedding=False,
            n_window=2,
            n_window_infer=4,
            conv_chunksize=500,
            dtype="float32",
        )
        self.assertEqual(encoder.window_aftercnn, 1)
        mask = ops.convert_to_numpy(encoder._build_window_attention_mask(5))
        # Expect the identity matrix (each position only attends to self).
        expected = np.eye(5, dtype=bool)[None, ...]
        np.testing.assert_array_equal(mask, expected)

    def test_window_mask_multi_block(self):
        encoder = Qwen3OmniAudioEncoder(
            num_mel_bins=80,
            d_model=32,
            encoder_layers=1,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            output_dim=16,
            max_source_positions=100,
            scale_embedding=False,
            n_window=2,
            n_window_infer=8,
            conv_chunksize=500,
            dtype="float32",
        )
        # chunk_length=4 -> after CNN = 1, ratio = 8//4 = 2,
        # window_aftercnn = 1 * 2 = 2.
        self.assertEqual(encoder.window_aftercnn, 2)
        mask = ops.convert_to_numpy(encoder._build_window_attention_mask(5))
        # Blocks: [0,1] [2,3] [4]. Build expected manually.
        expected = np.zeros((5, 5), dtype=bool)
        for i in range(5):
            for j in range(5):
                if i // 2 == j // 2:
                    expected[i, j] = True
        np.testing.assert_array_equal(mask[0], expected)

    def test_short_audio_mask_is_full(self):
        encoder = Qwen3OmniAudioEncoder(
            num_mel_bins=80,
            d_model=32,
            encoder_layers=1,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            output_dim=16,
            max_source_positions=100,
            scale_embedding=False,
            dtype="float32",
        )
        mask = ops.convert_to_numpy(encoder._build_window_attention_mask(20))
        self.assertTrue(bool(mask.all()))
