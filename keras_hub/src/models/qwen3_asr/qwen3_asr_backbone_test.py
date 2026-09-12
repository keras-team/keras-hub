import numpy as np
import pytest

from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone
from keras_hub.src.tests.test_case import TestCase


class Qwen3ASRBackboneTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 20
        self.num_mel_bins = 128
        self.audio_seq_len = 100  # 1 chunk (13 tokens)
        self.audio_token_id = 10

        self.init_kwargs = {
            "vocabulary_size": 100,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "hidden_dim": 32,
            "intermediate_dim": 64,
            "audio_num_mel_bins": self.num_mel_bins,
            "audio_num_layers": 2,
            "audio_num_attention_heads": 4,
            "audio_intermediate_dim": 64,
            "audio_d_model": 16,
            "audio_n_window": 50,
            "audio_downsample_hidden_size": 8,
            "audio_max_position_embeddings": 13,
            "audio_token_id": self.audio_token_id,
        }

        token_ids = np.ones((self.batch_size, self.seq_len), dtype="int32")
        # Placeholders at indices 2 to 14 (13 tokens)
        token_ids[:, 2:15] = self.audio_token_id

        padding_mask = np.ones((self.batch_size, self.seq_len), dtype="int32")
        audio_mel = np.random.uniform(
            size=(self.batch_size, self.audio_seq_len, self.num_mel_bins)
        ).astype("float32")
        audio_mask = np.ones(
            (self.batch_size, self.audio_seq_len), dtype="int32"
        )

        self.input_data = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "audio_mel": audio_mel,
            "audio_mask": audio_mask,
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Qwen3ASRBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(self.batch_size, self.seq_len, 32),
            variable_length_data=[self.input_data],
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen3ASRBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_variable_audio_lengths(self):
        # Test with 2 chunks (26 tokens)
        seq_len = 40
        audio_seq_len = 200

        token_ids = np.ones((self.batch_size, seq_len), dtype="int32")
        token_ids[0, 2:15] = self.audio_token_id  # Sample 0: 1 chunk
        token_ids[1, 2:28] = self.audio_token_id  # Sample 1: 2 chunks

        padding_mask = np.ones((self.batch_size, seq_len), dtype="int32")
        audio_mel = np.random.uniform(
            size=(self.batch_size, audio_seq_len, self.num_mel_bins)
        ).astype("float32")
        audio_mask = np.ones((self.batch_size, audio_seq_len), dtype="int32")
        audio_mask[0, 100:] = 0  # Sample 0 has only 100 valid frames

        inputs = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "audio_mel": audio_mel,
            "audio_mask": audio_mask,
        }

        backbone = Qwen3ASRBackbone(**self.init_kwargs)
        output = backbone(inputs)
        self.assertEqual(output.shape, (self.batch_size, seq_len, 32))
