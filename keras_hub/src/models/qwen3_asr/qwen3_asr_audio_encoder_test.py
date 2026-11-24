import json

import keras
import numpy as np
from keras import ops

from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_encoder import (
    Qwen3ASRAudioEncoder,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_encoder import (
    Qwen3ASRMultiModalProjector,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_encoder import (
    Qwen3AudioEncoderTransformerLayer,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_encoder import (
    compute_sinusoidal_positional_embedding,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3ASRAudioEncoderTest(TestCase):
    def test_transformer_layer_basics(self):
        self.run_layer_test(
            cls=Qwen3AudioEncoderTransformerLayer,
            init_kwargs={
                "hidden_dim": 8,
                "intermediate_dim": 16,
                "num_heads": 2,
                "dropout": 0.0,
            },
            input_data=np.random.uniform(size=(2, 4, 8)).astype("float32"),
            expected_output_shape=(2, 4, 8),
            expected_num_trainable_weights=16,
            expected_num_non_trainable_weights=0,
        )

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

    def test_transformer_layer_large_float16(self):
        layer = Qwen3AudioEncoderTransformerLayer(
            hidden_dim=8,
            intermediate_dim=16,
            num_heads=2,
            dtype="float16",
        )
        inputs = np.full((1, 4, 8), 65000.0, dtype="float16")
        outputs = ops.convert_to_numpy(layer(inputs))
        self.assertTrue(np.all(np.isfinite(outputs)))

    def test_sinusoidal_positional_embedding(self):
        emb = compute_sinusoidal_positional_embedding(13, 2)
        self.assertEqual(emb.shape, (13, 2))
        self.assertTrue(np.all(np.isfinite(emb)))
        with self.assertRaises(ValueError):
            compute_sinusoidal_positional_embedding(13, 0)
        with self.assertRaises(ValueError):
            compute_sinusoidal_positional_embedding(13, 3)

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

    def test_encoder_multi_window_and_output_spec(self):
        encoder = Qwen3ASRAudioEncoder(
            num_mel_bins=16,
            num_layers=1,
            num_attention_heads=2,
            intermediate_dim=32,
            d_model=16,
            n_window=25,
            n_window_infer=100,
            downsample_hidden_size=8,
            max_position_embeddings=13,
        )
        spec = encoder.compute_output_spec(
            keras.KerasTensor((1, 50, 16)),
            keras.KerasTensor((1, 50), dtype="int32"),
        )
        self.assertEqual(spec.shape, (1, 7, 16))

        feats = np.random.uniform(size=(2, 250, 16)).astype("float32")
        mask = np.ones((2, 250), dtype="int32")
        out = ops.convert_to_numpy(encoder(feats, mask))
        self.assertEqual(out.shape, (2, 35, 16))

    def test_encoder_masked_frame_invariance(self):
        encoder = Qwen3ASRAudioEncoder(
            num_mel_bins=16,
            num_layers=1,
            num_attention_heads=2,
            intermediate_dim=32,
            d_model=16,
            n_window=50,
            n_window_infer=200,
            downsample_hidden_size=8,
            max_position_embeddings=13,
        )
        rng = np.random.default_rng(11)
        feats_a = rng.normal(size=(1, 100, 16)).astype("float32")
        feats_b = feats_a.copy()
        feats_b[:, 50:, :] = 123.0
        mask = np.ones((1, 100), dtype="int32")
        mask[:, 50:] = 0

        out_a = ops.convert_to_numpy(encoder(feats_a, mask))
        out_b = ops.convert_to_numpy(encoder(feats_b, mask))
        self.assertAllClose(out_a, out_b, atol=0.0, rtol=0.0)

    def test_encoder_invalid_configs(self):
        with self.assertRaises(ValueError):
            Qwen3ASRAudioEncoder(n_window=0)
        with self.assertRaises(ValueError):
            Qwen3ASRAudioEncoder(n_window=50, n_window_infer=50)
        with self.assertRaises(ValueError):
            Qwen3ASRAudioEncoder(n_window=50, max_position_embeddings=10)

    def test_float16_masked_no_nan(self):
        rng = np.random.RandomState(0)
        encoder = Qwen3ASRAudioEncoder(
            num_mel_bins=16,
            num_layers=2,
            num_attention_heads=2,
            intermediate_dim=32,
            d_model=16,
            n_window=50,
            n_window_infer=200,
            downsample_hidden_size=8,
            max_position_embeddings=13,
            dtype="float16",
        )
        feats = rng.randn(2, 100, 16).astype("float16")
        mask = np.ones((2, 100), dtype="int32")
        mask[0, 50:] = 0
        enc_out = ops.convert_to_numpy(encoder(feats, mask))
        self.assertTrue(np.all(np.isfinite(enc_out)))

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
            activation=ops.silu,
        )

        self.assertIsNotNone(projector.linear_1)
        self.assertIsNotNone(projector.linear_2)
        self.assertEqual(projector.linear_1.units, d_model)

        config = projector.get_config()
        json.dumps(config)
        restored = Qwen3ASRMultiModalProjector.from_config(config)
        self.assertEqual(restored.d_model, d_model)
        self.assertEqual(restored.linear_1.units, d_model)
        self.assertEqual(restored.linear_2.units, output_dim)
