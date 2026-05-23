"""Tests for BLIP-2 Flan-T5 language model."""

import numpy as np
import pytest

from keras_hub.src.models.blip2.blip2_flan_t5_lm import BLIP2FlanT5
from keras_hub.src.tests.test_case import TestCase


class BLIP2FlanT5Test(TestCase):
    def setUp(self):
        self.vocab_size = 128
        self.hidden_dim = 16
        self.num_query_tokens = 4
        self.qformer_hidden_dim = 16

        self.init_kwargs = {
            "vocabulary_size": self.vocab_size,
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": 32,
            "num_query_tokens": self.num_query_tokens,
            "qformer_hidden_dim": self.qformer_hidden_dim,
            "key_value_dim": 8,
            "dropout": 0.0,
            "layer_norm_epsilon": 1e-6,
        }
        self.batch = 2
        self.enc_len = 6
        self.dec_len = 3
        self.input_data = {
            "token_ids": np.ones((self.batch, self.enc_len), dtype="int32"),
            "padding_mask": np.ones((self.batch, self.enc_len), dtype="int32"),
            "decoder_token_ids": np.ones(
                (self.batch, self.dec_len), dtype="int32"
            ),
            "decoder_padding_mask": np.ones(
                (self.batch, self.dec_len), dtype="int32"
            ),
            "qformer_features": np.random.uniform(
                size=(
                    self.batch,
                    self.num_query_tokens,
                    self.qformer_hidden_dim,
                )
            ).astype("float32"),
        }

    def _build(self, model=None, **kw):
        if model is None:
            model = BLIP2FlanT5(**{**self.init_kwargs, **kw})
        model(self.input_data)
        if not model.lm_head.built:
            model.lm_head.build((1, 1, self.hidden_dim))
        return model

    # ── Shape contracts ───────────────────────────────────────────────────────

    def test_call_returns_hidden_states_not_logits(self):
        """call() must return decoder hidden states (hidden_dim), not logits
        (vocab_size). This verifies no lm_head is buried inside call()."""
        model = self._build()
        out = model(self.input_data)
        self.assertEqual(out.shape, (self.batch, self.dec_len, self.hidden_dim))

    def test_lm_head_output_shape(self):
        """lm_head maps hidden states → (batch, seq, vocab_size)."""
        model = self._build()
        hidden = model(self.input_data)
        logits = model.lm_head(hidden)
        self.assertEqual(
            logits.shape, (self.batch, self.dec_len, self.vocab_size)
        )

    # ── Separate lm_head (the architectural invariant we fixed) ───────────────

    def test_lm_head_weights_not_tied_to_token_embedding(self):
        """lm_head.kernel must be a distinct weight from token_embedding.
        Flan-T5 uses tie_word_embeddings=False: assigning one must not
        change the other, and their shapes differ by a transpose."""
        from keras import ops

        model = self._build()
        emb = model.t5.token_embedding.embeddings  # (vocab, hidden)
        kernel = model.lm_head.kernel  # (hidden, vocab)

        # Kernel shape is transposed relative to embedding.
        self.assertEqual(kernel.shape, (self.hidden_dim, self.vocab_size))
        self.assertEqual(emb.shape, (self.vocab_size, self.hidden_dim))

        # Zero out lm_head kernel; embedding must be unaffected.
        original_emb = ops.convert_to_numpy(emb).copy()
        kernel.assign(np.zeros_like(ops.convert_to_numpy(kernel)))
        self.assertAllClose(
            emb,
            original_emb,
            msg="token_embedding changed when lm_head.kernel was reassigned",
        )

        # Zero out embedding; lm_head kernel must stay zeroed (not restored).
        emb.assign(np.zeros_like(ops.convert_to_numpy(emb)))
        self.assertAllClose(kernel, np.zeros_like(ops.convert_to_numpy(kernel)))

    def test_lm_head_and_token_embedding_give_different_logits(self):
        """lm_head(x) ≠ token_embedding(x, reverse=True) because their weight
        matrices are distinct (tie_word_embeddings=False)."""
        model = self._build()

        # Give both distinct, non-trivial values.
        emb_val = np.random.uniform(
            size=model.t5.token_embedding.embeddings.shape
        ).astype("float32")
        lm_val = np.random.uniform(size=model.lm_head.kernel.shape).astype(
            "float32"
        )
        model.t5.token_embedding.embeddings.assign(emb_val)
        model.lm_head.kernel.assign(lm_val)

        hidden = model(self.input_data)
        lm_logits = model.lm_head(hidden)
        rev_logits = model.token_embedding(hidden, reverse=True)

        self.assertNotAllClose(lm_logits, rev_logits)

    # ── Encoder context reaches the decoder ───────────────────────────────────

    def test_encoder_input_changes_decoder_output(self):
        """Cross-attention must propagate encoder context to the decoder:
        different encoder token_ids → different decoder hidden states."""
        model = self._build()
        data_a = dict(self.input_data)
        data_b = {
            **data_a,
            "token_ids": np.full((self.batch, self.enc_len), 5, dtype="int32"),
        }
        out_a = model(data_a)
        out_b = model(data_b)
        self.assertNotAllClose(out_a, out_b)

    def test_visual_prefix_changes_decoder_output(self):
        """Different qformer_features must produce different decoder hidden
        states, proving the visual prefix is wired into the encoder path."""
        model = self._build()
        out_a = model(self.input_data)
        data_b = {
            **self.input_data,
            "qformer_features": np.zeros(
                (self.batch, self.num_query_tokens, self.qformer_hidden_dim),
                dtype="float32",
            ),
        }
        out_b = model(data_b)
        self.assertNotAllClose(out_a, out_b)

    def test_decoder_token_ids_affect_output(self):
        """The decoder is causal: different decoder_token_ids must produce
        different hidden states (proving the decoder path is wired up)."""
        model = self._build()
        data_a = dict(self.input_data)
        data_b = {
            **data_a,
            "decoder_token_ids": np.full(
                (self.batch, self.dec_len), 7, dtype="int32"
            ),
        }
        out_a = model(data_a)
        out_b = model(data_b)
        self.assertNotAllClose(out_a, out_b)

    def test_decoder_defaults_to_encoder_input(self):
        """Passing encoder token_ids as decoder_token_ids is valid and covers
        the degenerate case used during backbone forward tracing."""
        model = self._build()
        data = {
            **self.input_data,
            "decoder_token_ids": self.input_data["token_ids"],
            "decoder_padding_mask": self.input_data["padding_mask"],
        }
        out = model(data)
        # decoder receives enc_len tokens, so output seq dim = enc_len
        self.assertEqual(out.shape, (self.batch, self.enc_len, self.hidden_dim))

    # ── Serialization ─────────────────────────────────────────────────────────

    def test_serialization_round_trip(self):
        """get_config()/from_config() must reconstruct an identical model,
        including language_projection and lm_head."""
        model = self._build()
        cfg = model.get_config()

        self.assertIn("language_projection", cfg)
        self.assertIn("lm_head", cfg)

        restored = BLIP2FlanT5.from_config(cfg)
        self.assertEqual(restored.get_config(), cfg)

    def test_weights_survive_serialization(self):
        """Weights transferred via save_weights/load_weights must give
        numerically identical outputs — this exercises lm_head too."""
        import os
        import tempfile

        model = self._build()
        clone = BLIP2FlanT5.from_config(model.get_config())
        clone(self.input_data)
        if not clone.lm_head.built:
            clone.lm_head.build((1, 1, self.hidden_dim))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "weights.weights.h5")
            model.save_weights(path)
            clone.load_weights(path)

        hidden = model(self.input_data)
        logits = model.lm_head(hidden)
        clone_hidden = clone(self.input_data)
        clone_logits = clone.lm_head(clone_hidden)

        self.assertAllClose(hidden, clone_hidden)
        self.assertAllClose(logits, clone_logits)

    # ── Training / inference behaviour ────────────────────────────────────────

    def test_dropout_deterministic_at_inference(self):
        """With dropout > 0, training=False must give identical outputs
        on repeated calls (no stochasticity at inference)."""
        model = self._build(dropout=0.5)
        out_a = model(self.input_data, training=False)
        out_b = model(self.input_data, training=False)
        self.assertAllClose(out_a, out_b)

    def test_dropout_stochastic_during_training(self):
        """With dropout > 0, training=True must produce different outputs
        on different calls (stochastic)."""
        model = self._build(dropout=0.5)
        out_a = model(self.input_data, training=True)
        out_b = model(self.input_data, training=True)
        self.assertNotAllClose(out_a, out_b)

    # ── Batch independence ────────────────────────────────────────────────────

    def test_batch_elements_are_independent(self):
        """Different batch elements with different inputs must produce
        different outputs (no cross-batch leakage in attention)."""
        model = self._build()
        data = {
            "token_ids": np.array(
                [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype="int32"
            ),
            "padding_mask": np.ones((2, 6), dtype="int32"),
            "decoder_token_ids": np.array(
                [[1, 2, 3], [4, 5, 6]], dtype="int32"
            ),
            "decoder_padding_mask": np.ones((2, 3), dtype="int32"),
            "qformer_features": np.random.uniform(
                size=(2, self.num_query_tokens, self.qformer_hidden_dim)
            ).astype("float32"),
        }
        out = model(data)
        self.assertNotAllClose(out[0], out[1])

    @pytest.mark.large
    def test_saved_model(self):
        """Full Keras model save/load must preserve numerical outputs."""
        import keras
        from keras import ops

        model = self._build()
        path = self.get_temp_dir() + "/flan_t5_lm.keras"
        model.save(path)
        loaded = keras.models.load_model(path)

        np.testing.assert_allclose(
            ops.convert_to_numpy(model(self.input_data)),
            ops.convert_to_numpy(loaded(self.input_data)),
            atol=1e-5,
        )
