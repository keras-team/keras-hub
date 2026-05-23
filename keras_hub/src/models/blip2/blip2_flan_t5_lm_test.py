"""Tests for BLIP-2 Flan-T5 language model."""

import numpy as np
import pytest

from keras_hub.src.models.blip2.blip2_flan_t5_lm import BLIP2FlanT5
from keras_hub.src.tests.test_case import TestCase

_TINY = {
    "vocabulary_size": 128,
    "num_layers": 2,
    "num_heads": 2,
    "hidden_dim": 16,
    "intermediate_dim": 32,
    "key_value_dim": 8,
    "num_query_tokens": 4,
    "qformer_hidden_dim": 16,
    "dropout": 0.0,
    "layer_norm_epsilon": 1e-6,
}

_XL = {
    "vocabulary_size": 32128,
    "num_layers": 24,
    "num_heads": 32,
    "hidden_dim": 2048,
    "intermediate_dim": 5120,
    "key_value_dim": 64,
    "num_query_tokens": 32,
    "qformer_hidden_dim": 768,
    "dropout": 0.1,
    "layer_norm_epsilon": 1e-6,
}

BATCH = 2
ENC_LEN = 6
DEC_LEN = 3


def _make_inputs(
    batch=BATCH,
    enc_len=ENC_LEN,
    dec_len=DEC_LEN,
    num_query_tokens=_TINY["num_query_tokens"],
    qformer_hidden_dim=_TINY["qformer_hidden_dim"],
    seed=0,
):
    rng = np.random.default_rng(seed)
    return {
        "token_ids": np.ones((batch, enc_len), dtype="int32"),
        "padding_mask": np.ones((batch, enc_len), dtype="int32"),
        "decoder_token_ids": np.ones((batch, dec_len), dtype="int32"),
        "decoder_padding_mask": np.ones((batch, dec_len), dtype="int32"),
        "qformer_features": rng.uniform(
            size=(batch, num_query_tokens, qformer_hidden_dim)
        ).astype("float32"),
    }


def _build(cfg=None, **overrides):
    cfg = {**(_TINY if cfg is None else cfg), **overrides}
    model = BLIP2FlanT5(**cfg)
    data = _make_inputs(
        num_query_tokens=cfg["num_query_tokens"],
        qformer_hidden_dim=cfg["qformer_hidden_dim"],
    )
    model(data)
    if not model.lm_head.built:
        model.lm_head.build((1, 1, cfg["hidden_dim"]))
    return model


class BLIP2FlanT5Test(TestCase):
    def setUp(self):
        self.model = _build()
        self.data = _make_inputs()

    def test_decoder_hidden_state_shape(self):
        out = self.model(self.data)
        self.assertEqual(out.shape, (BATCH, DEC_LEN, _TINY["hidden_dim"]))

    def test_lm_head_shape(self):
        out = self.model.lm_head(self.model(self.data))
        self.assertEqual(out.shape, (BATCH, DEC_LEN, _TINY["vocabulary_size"]))

    def test_lm_head_not_tied_to_token_embedding(self):
        from keras import ops

        emb = self.model.t5.token_embedding.embeddings
        kernel = self.model.lm_head.kernel

        self.assertEqual(
            kernel.shape, (_TINY["hidden_dim"], _TINY["vocabulary_size"])
        )
        self.assertEqual(
            emb.shape, (_TINY["vocabulary_size"], _TINY["hidden_dim"])
        )

        original = ops.convert_to_numpy(emb).copy()
        kernel.assign(np.zeros_like(ops.convert_to_numpy(kernel)))
        self.assertAllClose(emb, original)

    def test_language_projection_shape(self):
        k = self.model.language_projection.kernel
        self.assertEqual(
            k.shape, (_TINY["qformer_hidden_dim"], _TINY["hidden_dim"])
        )

    def test_encoder_context_reaches_decoder(self):
        data_b = {
            **self.data,
            "token_ids": np.full((BATCH, ENC_LEN), 5, dtype="int32"),
        }
        self.assertNotAllClose(self.model(self.data), self.model(data_b))

    def test_visual_prefix_changes_decoder_output(self):
        data_b = {
            **self.data,
            "qformer_features": np.zeros(
                (BATCH, _TINY["num_query_tokens"], _TINY["qformer_hidden_dim"]),
                dtype="float32",
            ),
        }
        self.assertNotAllClose(self.model(self.data), self.model(data_b))

    def test_decoder_tokens_affect_output(self):
        data_b = {
            **self.data,
            "decoder_token_ids": np.full((BATCH, DEC_LEN), 7, dtype="int32"),
        }
        self.assertNotAllClose(self.model(self.data), self.model(data_b))

    def test_causal_decoder_mask(self):
        out_base = self.model(self.data)

        corrupted = self.data["decoder_token_ids"].copy()
        corrupted[:, 1:] = 99
        data_b = {**self.data, "decoder_token_ids": corrupted}
        out_b = self.model(data_b)

        self.assertAllClose(out_base[:, 0, :], out_b[:, 0, :])
        self.assertNotAllClose(out_base[:, 1:, :], out_b[:, 1:, :])

    def test_batch_elements_independent(self):
        data = _make_inputs(
            batch=2,
            enc_len=ENC_LEN,
            dec_len=DEC_LEN,
            num_query_tokens=_TINY["num_query_tokens"],
            qformer_hidden_dim=_TINY["qformer_hidden_dim"],
        )
        data["token_ids"] = np.array(
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype="int32"
        )
        out = self.model(data)
        self.assertNotAllClose(out[0], out[1])

    def test_config_round_trip(self):
        cfg = self.model.get_config()
        self.assertIn("language_projection", cfg)
        self.assertIn("lm_head", cfg)
        self.assertEqual(BLIP2FlanT5.from_config(cfg).get_config(), cfg)

    def test_weights_round_trip(self):
        import os
        import tempfile

        clone = _build()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "w.weights.h5")
            self.model.save_weights(path)
            clone.load_weights(path)

        self.assertAllClose(self.model(self.data), clone(self.data))
        self.assertAllClose(
            self.model.lm_head(self.model(self.data)),
            clone.lm_head(clone(self.data)),
        )

    def test_inference_is_deterministic(self):
        model = _build(dropout=0.5)
        self.assertAllClose(
            model(self.data, training=False),
            model(self.data, training=False),
        )

    def test_training_is_stochastic(self):
        model = _build(dropout=0.5)
        self.assertNotAllClose(
            model(self.data, training=True),
            model(self.data, training=True),
        )

    @pytest.mark.large
    def test_xl_config_instantiates(self):
        model = BLIP2FlanT5(**_XL)
        self.assertIsNotNone(model)
        self.assertEqual(model.hidden_dim, 2048)
        self.assertEqual(model.num_layers, 24)

    @pytest.mark.large
    def test_saved_model(self):
        import keras
        from keras import ops

        path = self.get_temp_dir() + "/flan_t5_lm.keras"
        self.model.save(path)
        loaded = keras.models.load_model(path)
        self.assertAllClose(
            ops.convert_to_numpy(self.model(self.data)),
            ops.convert_to_numpy(loaded(self.data)),
            atol=1e-5,
        )
