import numpy as np

from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_encoder import (
    Qwen3ASRAudioEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3ASRAudioEncoderTest(TestCase):
    """Tests for Qwen3ASRAudioEncoder."""

    def setUp(self):
        self.init_kwargs = {
            "d_model": 16,
            "encoder_layers": 2,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 32,
            "downsample_hidden_size": 8,
            "num_mel_bins": 8,
            "n_window": 5,  # chunk_len = 10
            "max_position_embeddings": 100,
            "output_dim": 24,
        }
        self.batch_size = 2
        self.chunk_len = 10
        self.num_chunks = 3
        self.seq_len = self.num_chunks * self.chunk_len  # 30
        self.num_mel_bins = 8

        # Input data: (B, T, F)
        self.audio_mel = np.ones(
            (self.batch_size, self.seq_len, self.num_mel_bins), dtype="float32"
        )
        self.audio_mel_mask = np.ones(
            (self.batch_size, self.seq_len), dtype="int32"
        )

    def test_encoder_basics(self):
        """Encoder initialises and produces the correct output shape."""
        encoder = Qwen3ASRAudioEncoder(**self.init_kwargs)
        out = encoder(self.audio_mel)

        # Output Time steps = num_chunks * W_out
        # W_out for chunk_len=10:
        # 10 -> 5 -> 3 -> 2
        # So W_out = 2
        # Total output steps = 3 * 2 = 6
        expected_time_steps = 6
        self.assertEqual(out.shape, (self.batch_size, expected_time_steps, 24))

    def test_encoder_with_mask(self):
        """Encoder works with padding mask."""
        encoder = Qwen3ASRAudioEncoder(**self.init_kwargs)

        # Make second sample shorter
        mask = np.ones_like(self.audio_mel_mask)
        mask[1, 15:] = (
            0  # Only first 15 steps valid in 2nd sample (belongs to chunk 0 and 1)
        )

        out = encoder(self.audio_mel, audio_mel_mask=mask)

        expected_time_steps = 6
        self.assertEqual(out.shape, (self.batch_size, expected_time_steps, 24))

    def test_get_config_round_trip(self):
        """get_config / from_config should reproduce identical parameters."""
        encoder = Qwen3ASRAudioEncoder(**self.init_kwargs)
        config = encoder.get_config()
        restored = Qwen3ASRAudioEncoder.from_config(config)
        for key, val in self.init_kwargs.items():
            self.assertEqual(getattr(restored, key), val)
