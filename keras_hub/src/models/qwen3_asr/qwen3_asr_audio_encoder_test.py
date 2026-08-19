import numpy as np

from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_encoder import (
    Qwen3ASRAudioEncoder,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_encoder import (
    Qwen3ASRMultiModalProjector,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3ASRAudioEncoderTest(TestCase):
    def test_encoder_call(self):
        batch_size = 2
        num_mel_bins = 128
        T = 200  # 2 chunks

        self.run_layer_test(
            cls=Qwen3ASRAudioEncoder,
            init_kwargs={
                "num_mel_bins": num_mel_bins,
                "num_layers": 2,
                "num_attention_heads": 4,
                "intermediate_dim": 64,
                "d_model": 32,
                "n_window": 50,
                "downsample_hidden_size": 16,
                "max_position_embeddings": 13,
            },
            input_data={
                "input_features": np.random.uniform(
                    size=(batch_size, T, num_mel_bins)
                ).astype("float32"),
                "input_features_mask": np.ones((batch_size, T), dtype="int32"),
            },
            expected_output_shape=(batch_size, 26, 32),
            expected_num_trainable_weights=41,
            expected_num_non_trainable_weights=1,
            expected_num_non_trainable_variables=1,
        )

    def test_projector_call(self):
        batch_size = 2
        seq_len = 10
        d_model = 32
        output_dim = 64

        self.run_layer_test(
            cls=Qwen3ASRMultiModalProjector,
            init_kwargs={
                "output_dim": output_dim,
                "activation": "gelu",
            },
            input_data=np.random.uniform(
                size=(batch_size, seq_len, d_model)
            ).astype("float32"),
            expected_output_shape=(batch_size, seq_len, output_dim),
            expected_num_trainable_weights=4,
            expected_num_non_trainable_weights=0,
        )
