import numpy as np

from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_encoder import (
    Qwen3ASRAudioEncoder,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_encoder import (
    Qwen3ASRMultiModalProjector,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_encoder import (
    Qwen3AudioEncoderTransformerLayer,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3ASRAudioEncoderTest(TestCase):
    def test_transformer_layer_dropout(self):
        layer = Qwen3AudioEncoderTransformerLayer(
            hidden_dim=8,
            intermediate_dim=16,
            num_heads=2,
            dropout=1.0,
        )
        inputs = np.random.uniform(size=(2, 4, 8)).astype("float32")

        outputs = layer(inputs, training=True)

        self.assertAllClose(outputs, inputs)

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
            expected_num_non_trainable_weights=0,
            expected_num_non_trainable_variables=0,
        )

    def test_projector_call(self):
        batch_size = 2
        seq_len = 10
        d_model = 32
        output_dim = 64

        self.run_layer_test(
            cls=Qwen3ASRMultiModalProjector,
            init_kwargs={
                "d_model": d_model,
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

    def test_projector_layers_exist_before_build_and_config_round_trip(self):
        d_model = 32
        output_dim = 64
        projector = Qwen3ASRMultiModalProjector(
            d_model=d_model,
            output_dim=output_dim,
        )

        self.assertIsNotNone(projector.linear_1)
        self.assertIsNotNone(projector.linear_2)
        self.assertEqual(projector.linear_1.units, d_model)

        restored = Qwen3ASRMultiModalProjector.from_config(
            projector.get_config()
        )
        self.assertEqual(restored.d_model, d_model)
        self.assertEqual(restored.linear_1.units, d_model)
        self.assertEqual(restored.linear_2.units, output_dim)
