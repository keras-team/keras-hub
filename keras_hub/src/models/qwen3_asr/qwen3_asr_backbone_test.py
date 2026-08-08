import numpy as np
from keras import ops

from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_encoder import (
    Qwen3ASRAudioEncoder,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone
from keras_hub.src.tests.test_case import TestCase


class Qwen3ASRBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 2,
            "hidden_dim": 16,
            "intermediate_dim": 8,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

        self.audio_encoder = Qwen3ASRAudioEncoder(
            d_model=8,
            encoder_layers=1,
            encoder_attention_heads=1,
            encoder_ffn_dim=16,
            downsample_hidden_size=4,
            num_mel_bins=8,
            n_window=5,
            max_position_embeddings=100,
            output_dim=16,
        )

    def test_backbone_basics_text_only(self):
        self.run_backbone_test(
            cls=Qwen3ASRBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 16),
            run_quantization_check=False,
        )

    def test_backbone_multimodal(self):
        kwargs = {**self.init_kwargs, "audio_encoder": self.audio_encoder}
        model = Qwen3ASRBackbone(**kwargs)

        seq_len = 10
        chunk_len = 10
        num_chunks = 3
        total_mels = num_chunks * chunk_len

        audio_mel = np.ones((2, total_mels, 8), dtype="float32")
        audio_mel_mask = np.ones((2, total_mels), dtype="int32")

        audio_indices = np.array(
            [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dtype="int32"
        )

        input_data = {
            "token_ids": np.ones((2, seq_len), dtype="int32"),
            "padding_mask": np.ones((2, seq_len), dtype="int32"),
            "audio_mel": audio_mel,
            "audio_mel_mask": audio_mel_mask,
            "audio_indices": audio_indices,
        }

        output = model(input_data)
        self.assertEqual(ops.shape(output), (2, seq_len, 16))

    def test_get_config_round_trip(self):
        encoder = self.audio_encoder
        kwargs = {**self.init_kwargs, "audio_encoder": encoder}
        model = Qwen3ASRBackbone(**kwargs)
        config = model.get_config()

        restored = Qwen3ASRBackbone.from_config(config)

        self.assertEqual(restored.vocabulary_size, model.vocabulary_size)
        self.assertEqual(restored.hidden_dim, model.hidden_dim)
        self.assertIsNotNone(restored.audio_encoder)
        self.assertEqual(restored.audio_encoder.output_dim, encoder.output_dim)
