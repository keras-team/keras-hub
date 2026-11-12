import json

import keras
import numpy as np
from absl.testing import parameterized
from keras import ops

from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.models.mobile_bert.mobile_bert_layers import (
    MobileBertAttention,
)
from keras_hub.src.models.mobile_bert.mobile_bert_layers import (
    MobileBertBottleneck,
)
from keras_hub.src.models.mobile_bert.mobile_bert_layers import (
    MobileBertEncoderLayer,
)
from keras_hub.src.models.mobile_bert.mobile_bert_layers import (
    MobileBertFeedForward,
)
from keras_hub.src.models.mobile_bert.mobile_bert_layers import (
    MobileBertMaskedLMHead,
)
from keras_hub.src.models.mobile_bert.mobile_bert_layers import (
    MobileBertTrigramProjection,
)
from keras_hub.src.models.mobile_bert.mobile_bert_layers import NoNorm
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import running_on_gpu

NORM_CASES = (("no_norm", "no_norm"), ("layer_norm", "layer_norm"))


class MobileBertLayersTest(TestCase):
    def setUp(self):
        self.compute_dtype = keras.config.dtype_policy().compute_dtype

    def assert_norm_layer(self, norm, normalization_type):
        if normalization_type == "layer_norm":
            self.assertIsInstance(norm, keras.layers.LayerNormalization)
            self.assertEqual(norm.epsilon, 1e-5)
        else:
            self.assertIsInstance(norm, NoNorm)

    def test_no_norm(self):
        layer = NoNorm()
        layer.build((2, 4, 16))
        layer.gamma.assign(np.ones((16,)) * 2.0)
        layer.beta.assign(np.ones((16,)) * 3.0)
        input_data = ops.ones((2, 4, 16), dtype=self.compute_dtype) * 2.0
        expected = ops.ones((2, 4, 16), dtype=self.compute_dtype) * 7.0
        self.assertAllClose(layer(input_data), expected)

        self.run_layer_test(
            cls=NoNorm,
            init_kwargs={},
            input_data=input_data,
            expected_output_shape=(2, 4, 16),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_non_trainable_variables=0,
        )

    def test_no_norm_output_dtype_matches_spec(self):
        # With autocast off, an uncast float32 input must still come out in
        # the layer's compute dtype, matching `compute_output_spec`.
        layer = NoNorm(dtype="float16")
        layer.autocast = False
        layer.build((2, 4))
        layer.gamma.assign(np.full((4,), 2.0))
        layer.beta.assign(np.full((4,), 3.0))
        spec = layer.compute_output_spec(
            keras.KerasTensor((2, 4), dtype="float32")
        )
        output = layer(ops.ones((2, 4), dtype="float32"))
        self.assertEqual(spec.dtype, "float16")
        self.assertEqual(
            keras.backend.standardize_dtype(output.dtype), "float16"
        )
        self.assertAllClose(output, np.full((2, 4), 5.0))

    @parameterized.named_parameters(
        ("bottleneck", MobileBertBottleneck, {"intra_bottleneck_dim": 8}),
        (
            "attention",
            MobileBertAttention,
            {"num_heads": 2, "intra_bottleneck_dim": 16},
        ),
        (
            "feed_forward",
            MobileBertFeedForward,
            {"intermediate_dim": 32, "output_dim": 16},
        ),
        (
            "encoder_layer",
            MobileBertEncoderLayer,
            {
                "hidden_dim": 32,
                "intra_bottleneck_dim": 16,
                "intermediate_dim": 32,
                "num_heads": 2,
            },
        ),
    )
    def test_invalid_normalization_type(self, cls, kwargs):
        with self.assertRaisesRegex(ValueError, "normalization_type"):
            cls(normalization_type="layernorm", **kwargs)

    def test_attention_rejects_indivisible_heads(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            MobileBertAttention(num_heads=6, intra_bottleneck_dim=128)

    def test_attention_residual_shape(self):
        attn = MobileBertAttention(
            num_heads=2,
            intra_bottleneck_dim=8,
            normalization_type="layer_norm",
        )
        attn.build((2, 4, 8), (2, 4, 16), (2, 4, 16))
        self.assertEqual(tuple(attn.output_layer_norm.gamma.shape), (16,))
        self.assertEqual(
            attn.compute_output_shape((2, 4, 8), (2, 4, 16), (2, 4, 16)),
            (2, 4, 16),
        )

    def test_invalid_num_ffn(self):
        with self.assertRaisesRegex(ValueError, "num_feedforward_networks"):
            MobileBertEncoderLayer(
                hidden_dim=32,
                intra_bottleneck_dim=16,
                intermediate_dim=32,
                num_heads=2,
                num_feedforward_networks=0,
            )

    def test_encoder_layer_activation_attr(self):
        enc = MobileBertEncoderLayer(
            hidden_dim=16,
            intra_bottleneck_dim=8,
            intermediate_dim=32,
            num_heads=2,
            activation="relu",
        )
        ffn = MobileBertFeedForward(
            intermediate_dim=32, output_dim=8, activation="relu"
        )
        self.assertEqual(enc.activation, "relu")
        self.assertEqual(ffn.activation, "relu")
        self.assertEqual(enc.get_config()["activation"], "relu")
        self.assertEqual(ffn.get_config()["activation"], "relu")

    def test_dense_kernel_initializers(self):
        encoder = MobileBertEncoderLayer(
            hidden_dim=32,
            intra_bottleneck_dim=16,
            intermediate_dim=32,
            num_heads=2,
            num_feedforward_networks=2,
        )
        head = MobileBertMaskedLMHead(
            ReversibleEmbedding(100, 16), hidden_dim=32
        )
        roots = [encoder, MobileBertTrigramProjection(8), head]
        dense_layers = [
            layer
            for root in roots
            for layer in root._flatten_layers(include_self=False)
            if isinstance(layer, keras.layers.Dense)
        ]
        # 2 bottlenecks + 4 attention + 2 x 2 ffn + output bottleneck,
        # trigram projection, head transform.
        self.assertEqual(len(dense_layers), 13)
        for dense in dense_layers:
            self.assertIsInstance(
                dense.kernel_initializer, keras.initializers.TruncatedNormal
            )
            self.assertEqual(dense.kernel_initializer.stddev, 0.02)
        head.build((2, 4, 32))
        extra_kernel = ops.convert_to_numpy(head.extra_kernel)
        self.assertGreater(np.abs(extra_kernel).max(), 0.0)
        self.assertBetween(np.std(extra_kernel), 0.01, 0.03)

    def test_trigram_projection(self):
        # We must build a layer with hidden_dim=3*embedding, set dense.kernel to
        # identity and bias to zero, feed a known sequence, and assert the
        # output equals concat([next, current, previous]) with zero padding.
        layer = MobileBertTrigramProjection(hidden_dim=3)
        input_data = ops.convert_to_tensor(
            [[[1.0], [2.0], [3.0]]], dtype=self.compute_dtype
        )
        layer.build((1, 3, 1))
        layer.set_weights([np.eye(3), np.zeros(3)])
        output = layer(input_data)

        expected = ops.convert_to_tensor(
            [[[2.0, 1.0, 0.0], [3.0, 2.0, 1.0], [0.0, 3.0, 2.0]]],
            dtype=self.compute_dtype,
        )
        self.assertAllClose(output, expected)

        self.run_layer_test(
            cls=MobileBertTrigramProjection,
            init_kwargs={"hidden_dim": 32},
            input_data=ops.ones((2, 4, 16), dtype=self.compute_dtype),
            expected_output_shape=(2, 4, 32),
            expected_num_trainable_weights=2,
            expected_num_non_trainable_weights=0,
            expected_num_non_trainable_variables=0,
        )

    @parameterized.named_parameters(*NORM_CASES)
    def test_bottleneck(self, normalization_type):
        init_kwargs = {
            "intra_bottleneck_dim": 8,
            "normalization_type": normalization_type,
            "layer_norm_epsilon": 1e-5,
        }
        self.assert_norm_layer(
            MobileBertBottleneck(**init_kwargs).layer_norm, normalization_type
        )
        self.run_layer_test(
            cls=MobileBertBottleneck,
            init_kwargs=init_kwargs,
            input_data=ops.ones((2, 4, 16), dtype=self.compute_dtype),
            expected_output_shape=(2, 4, 8),
            expected_num_trainable_weights=4,
            expected_num_non_trainable_weights=0,
            expected_num_non_trainable_variables=0,
        )

    @parameterized.named_parameters(*NORM_CASES)
    def test_attention(self, normalization_type):
        init_kwargs = {
            "num_heads": 2,
            "intra_bottleneck_dim": 16,
            "normalization_type": normalization_type,
            "layer_norm_epsilon": 1e-5,
        }
        self.assert_norm_layer(
            MobileBertAttention(**init_kwargs).output_layer_norm,
            normalization_type,
        )
        self.run_layer_test(
            cls=MobileBertAttention,
            init_kwargs=init_kwargs,
            input_data={
                "query": ops.ones((2, 4, 16), dtype=self.compute_dtype),
                "value": ops.ones((2, 4, 32), dtype=self.compute_dtype),
                "residual": ops.ones((2, 4, 16), dtype=self.compute_dtype),
            },
            expected_output_shape=(2, 4, 16),
            expected_num_trainable_weights=10,
            expected_num_non_trainable_weights=0,
            expected_num_non_trainable_variables=0,
        )

    @parameterized.named_parameters(*NORM_CASES)
    def test_feed_forward(self, normalization_type):
        init_kwargs = {
            "intermediate_dim": 32,
            "output_dim": 16,
            "normalization_type": normalization_type,
            "layer_norm_epsilon": 1e-5,
        }
        self.assert_norm_layer(
            MobileBertFeedForward(**init_kwargs).output_layer_norm,
            normalization_type,
        )
        self.run_layer_test(
            cls=MobileBertFeedForward,
            init_kwargs=init_kwargs,
            input_data=ops.ones((2, 4, 16), dtype=self.compute_dtype),
            expected_output_shape=(2, 4, 16),
            expected_num_trainable_weights=6,
            expected_num_non_trainable_weights=0,
            expected_num_non_trainable_variables=0,
        )

    @parameterized.named_parameters(*NORM_CASES)
    def test_encoder_layer(self, normalization_type):
        init_kwargs = {
            "hidden_dim": 32,
            "intra_bottleneck_dim": 16,
            "intermediate_dim": 32,
            "num_heads": 2,
            "num_feedforward_networks": 2,
            "normalization_type": normalization_type,
            "layer_norm_epsilon": 1e-5,
        }
        encoder = MobileBertEncoderLayer(**init_kwargs)
        self.assert_norm_layer(
            encoder.output_bottleneck_layer_norm, normalization_type
        )
        self.assert_norm_layer(
            encoder.ffn_layers[0].output_layer_norm, normalization_type
        )
        self.run_layer_test(
            cls=MobileBertEncoderLayer,
            init_kwargs=init_kwargs,
            input_data=ops.ones((2, 4, 32), dtype=self.compute_dtype),
            expected_output_shape=(2, 4, 32),
            expected_num_trainable_weights=4 + 4 + 10 + (6 * 2) + 4,
            expected_num_non_trainable_weights=0,
            # Dropout seed_generator_state
            expected_num_non_trainable_variables=1,
        )

    def test_encoder_layer_padding_independence(self):
        keras.utils.set_random_seed(42)
        rng = np.random.default_rng(42)
        layer = MobileBertEncoderLayer(
            hidden_dim=32,
            intra_bottleneck_dim=16,
            intermediate_dim=32,
            num_heads=2,
            num_feedforward_networks=2,
            normalization_type="no_norm",
        )
        inputs_unmasked_np = rng.uniform(size=(2, 4, 32))
        inputs_unmasked = ops.convert_to_tensor(
            inputs_unmasked_np, dtype=self.compute_dtype
        )
        # mask a trailing token
        padding_mask = ops.convert_to_tensor([[1, 1, 1, 0], [1, 1, 1, 0]])
        output_masked = layer(inputs_unmasked, padding_mask=padding_mask)

        # 3-token prefix alone
        inputs_prefix = ops.convert_to_tensor(
            inputs_unmasked_np[:, :3, :], dtype=self.compute_dtype
        )
        output_prefix = layer(inputs_prefix)

        # compare the first tokens' outputs with and without it
        self.assertAllClose(output_masked[:, :3, :], output_prefix, atol=1e-5)

    def test_masked_lm_head(self):
        keras.utils.set_random_seed(42)
        rng = np.random.default_rng(42)
        vocab_size = 100
        embedding_dim = 16
        hidden_dim = 32

        # Create random weights
        token_emb_w = rng.uniform(size=(vocab_size, embedding_dim))
        extra_kernel = rng.uniform(size=(embedding_dim, vocab_size))
        bias = rng.uniform(size=(vocab_size,))

        token_embedding = ReversibleEmbedding(vocab_size, embedding_dim)
        token_embedding.build((2, 4))
        token_embedding.set_weights([token_emb_w])

        # `layer_norm_epsilon` is an optional keyword after `vocabulary_size`
        # (docstring order), so positional construction with just the tied
        # embedding works and the epsilon reaches the transform layer norm.
        self.assertEqual(
            MobileBertMaskedLMHead(
                token_embedding, hidden_dim=hidden_dim
            ).transform_layer_norm.epsilon,
            1e-12,
        )
        self.assertEqual(
            MobileBertMaskedLMHead(
                token_embedding, hidden_dim=hidden_dim, layer_norm_epsilon=1e-5
            ).transform_layer_norm.epsilon,
            1e-5,
        )

        layer = MobileBertMaskedLMHead(
            vocabulary_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            layer_norm_epsilon=1e-12,
            token_embedding=token_embedding,
        )
        inputs_np = rng.uniform(size=(2, 4, hidden_dim))
        inputs = ops.convert_to_tensor(inputs_np, dtype=self.compute_dtype)
        layer.build((2, 4, hidden_dim))

        # Test: verify logits match numpy computation.
        # Set transform_dense_w to identity and bias to zero
        transform_dense_w = np.eye(hidden_dim)
        transform_dense_b = np.zeros(hidden_dim)
        transform_ln_gamma = np.ones(hidden_dim)
        transform_ln_beta = np.zeros(hidden_dim)

        layer.transform_dense.set_weights(
            [transform_dense_w, transform_dense_b]
        )
        layer.transform_layer_norm.set_weights(
            [transform_ln_gamma, transform_ln_beta]
        )
        layer.extra_kernel.assign(extra_kernel)
        layer.bias.assign(bias)

        mask_positions = ops.convert_to_tensor([[1, 2], [0, 1]])
        output = layer(inputs, mask_positions=mask_positions)

        self.assertEqual(output.shape, (2, 2, vocab_size))

        # Compute t = layernorm(relu(x)) in numpy
        # First relu
        t = np.maximum(inputs_np, 0.0)
        # LayerNorm (gamma=1, beta=0, eps=1e-12)
        mean = np.mean(t, axis=-1, keepdims=True)
        var = np.var(t, axis=-1, keepdims=True)
        t = (t - mean) / np.sqrt(var + 1e-12)

        # Gather mask_positions
        # mask_positions is [[1, 2], [0, 1]]
        t_gathered = np.zeros((2, 2, hidden_dim))
        t_gathered[0, 0] = t[0, 1]
        t_gathered[0, 1] = t[0, 2]
        t_gathered[1, 0] = t[1, 0]
        t_gathered[1, 1] = t[1, 1]

        logits_emb = t_gathered[..., :embedding_dim] @ token_emb_w.T
        logits_extra = t_gathered[..., embedding_dim:] @ extra_kernel
        expected = logits_emb + logits_extra + bias

        # CPU fp32 vs this fp64 reference measured 5.99e-6; A100 TF32 matmuls
        # measured 3.09e-3. Keep CPU strict.
        tol = 1e-2 if running_on_gpu() else 1e-5
        self.assertAllClose(
            output,
            ops.convert_to_tensor(expected, dtype=self.compute_dtype),
            atol=tol,
            rtol=tol,
        )

        self.run_layer_test(
            cls=MobileBertMaskedLMHead,
            init_kwargs={
                "vocabulary_size": 100,
                "embedding_dim": 16,
                "hidden_dim": 32,
                "layer_norm_epsilon": 1e-12,
                "token_embedding": ReversibleEmbedding(100, 16),
            },
            input_data={
                "inputs": ops.ones((2, 4, 32), dtype=self.compute_dtype),
                "mask_positions": ops.array([[1, 2], [0, 3]], dtype="int32"),
            },
            expected_output_shape=(2, 2, 100),
            # embeddings, transform kernel/bias, layer norm gamma/beta,
            # extra_kernel, bias.
            expected_num_trainable_weights=7,
            expected_num_non_trainable_weights=0,
            expected_num_non_trainable_variables=0,
        )

    def test_masked_lm_head_builds_token_embedding(self):
        head = MobileBertMaskedLMHead(
            ReversibleEmbedding(100, 16), hidden_dim=32
        )
        head.build((2, 4, 32))
        self.assertTrue(head.token_embedding.built)
        self.assertEqual(len(head.trainable_weights), 7)
        head(ops.ones((2, 4, 32), dtype=self.compute_dtype))
        self.assertEqual(len(head.trainable_weights), 7)

    def test_masked_lm_head_from_config_idempotent(self):
        head = MobileBertMaskedLMHead(ReversibleEmbedding(50, 8), hidden_dim=16)
        cfg = head.get_config()
        restored_1 = MobileBertMaskedLMHead.from_config(cfg)
        restored_2 = MobileBertMaskedLMHead.from_config(cfg)
        self.assertIsInstance(cfg["token_embedding"], dict)
        self.assertTrue(json.dumps(cfg))
        self.assertEqual(restored_1.vocabulary_size, 50)
        self.assertEqual(restored_2.vocabulary_size, 50)

    def test_masked_lm_head_validation(self):
        with self.assertRaisesRegex(TypeError, "token_embedding"):
            MobileBertMaskedLMHead(vocabulary_size=100)
        with self.assertRaisesRegex(ValueError, "token_embedding"):
            MobileBertMaskedLMHead(None, hidden_dim=32)
        with self.assertRaisesRegex(ValueError, "vocabulary_size"):
            MobileBertMaskedLMHead(
                ReversibleEmbedding(100, 16), vocabulary_size=99
            )
        with self.assertRaisesRegex(ValueError, "embedding_dim"):
            MobileBertMaskedLMHead(
                ReversibleEmbedding(100, 16), embedding_dim=32, hidden_dim=64
            )
        with self.assertRaisesRegex(ValueError, "hidden_dim"):
            MobileBertMaskedLMHead(ReversibleEmbedding(100, 16), hidden_dim=8)
        # Both sizes are inferred from the embedding when omitted.
        head = MobileBertMaskedLMHead(
            ReversibleEmbedding(100, 16), hidden_dim=32
        )
        self.assertEqual(head.vocabulary_size, 100)
        self.assertEqual(head.embedding_dim, 16)
        output = head(
            ops.ones((2, 4, 32), dtype=self.compute_dtype),
            mask_positions=ops.array([[1, 2], [0, 3]]),
        )
        self.assertEqual(output.shape, (2, 2, 100))
        self.assertTrue(np.isfinite(ops.convert_to_numpy(output)).all())

    def test_masked_lm_head_hidden_equals_embedding(self):
        # `hidden_dim == embedding_dim` leaves a zero-row `extra_kernel`; the
        # head then reduces to the tied embedding projection plus bias.
        keras.utils.set_random_seed(0)
        rng = np.random.default_rng(0)
        token_embedding = ReversibleEmbedding(100, 16)
        head = MobileBertMaskedLMHead(
            token_embedding, hidden_dim=16, layer_norm_epsilon=1e-12
        )
        head.build((2, 4, 16))
        self.assertEqual(tuple(head.extra_kernel.shape), (0, 100))
        emb_w = rng.uniform(size=(100, 16))
        bias = rng.uniform(size=(100,))
        token_embedding.set_weights([emb_w])
        head.transform_dense.set_weights([np.eye(16), np.zeros(16)])
        head.transform_layer_norm.set_weights([np.ones(16), np.zeros(16)])
        head.bias.assign(bias)
        x_np = rng.uniform(size=(2, 4, 16))
        t = np.maximum(x_np, 0.0)
        t = (t - t.mean(-1, keepdims=True)) / np.sqrt(
            t.var(-1, keepdims=True) + 1e-12
        )
        expected = t @ emb_w.T + bias
        output = head(ops.convert_to_tensor(x_np, dtype=self.compute_dtype))
        # A100 TF32 matmul drift across stacked MobileBERT ops + vocab
        # projection reaches ~3.3e-3; keep CPU strict at 1e-5.
        tol = 1e-2 if running_on_gpu() else 1e-5
        self.assertAllClose(
            output,
            ops.convert_to_tensor(expected, dtype=self.compute_dtype),
            atol=tol,
            rtol=tol,
        )
        self.assertTrue(np.isfinite(ops.convert_to_numpy(output)).all())

    def test_masked_lm_head_mixed_precision(self):
        layer = MobileBertMaskedLMHead(
            vocabulary_size=100,
            embedding_dim=16,
            hidden_dim=32,
            token_embedding=ReversibleEmbedding(100, 16, dtype="mixed_float16"),
            layer_norm_epsilon=1e-12,
            dtype="mixed_float16",
        )
        layer.build((2, 4, 32))

        input_data = ops.ones((2, 4, 32), dtype="float16")
        output = layer(input_data, mask_positions=ops.array([[1], [2]]))

        # Verify output is finiteness and properly casted back to compute_dtype
        self.assertTrue(np.isfinite(ops.convert_to_numpy(output)).all())
        self.assertEqual(
            keras.backend.standardize_dtype(output.dtype), "float16"
        )

        # Force `transform_layer_norm` to `mixed_float16` (`x` in `float16` at
        # +1000/sqrt(3) and -3000/sqrt(3)) while `ReversibleEmbedding` with
        # `reverse_dtype="float32"` computes `logits_emb = 3.2e6 / sqrt(3)`
        # (~1.85e6 > 65504) in `float32`, pinning `ops.cast(logits_emb,
        # self.compute_dtype)` without overflowing `float16`.
        head = MobileBertMaskedLMHead(
            vocabulary_size=100,
            embedding_dim=16,
            hidden_dim=32,
            token_embedding=ReversibleEmbedding(
                100, 16, reverse_dtype="float32", dtype="float32"
            ),
            layer_norm_epsilon=1e-12,
            dtype="float32",
        )
        head.build((2, 4, 32))
        head.transform_layer_norm.dtype_policy = keras.dtype_policies.get(
            "mixed_float16"
        )
        head.transform_dense.set_weights(
            [np.eye(32, dtype="float32"), np.zeros(32, dtype="float32")]
        )
        head.transform_layer_norm.set_weights(
            [
                np.full(32, 1000.0, dtype="float32"),
                np.zeros(32, dtype="float32"),
            ]
        )
        head.extra_kernel.assign(np.zeros((16, 100), dtype="float32"))
        emb_w = np.full((100, 16), 100.0, dtype="float32")
        emb_w[:, 8:] = -100.0
        head.token_embedding.embeddings.assign(emb_w)

        inp = np.ones((2, 4, 32), dtype="float32")
        inp[:, :, 8:16] = -1.0
        out32 = head(inp, mask_positions=ops.array([[1], [2]]))
        self.assertEqual(
            keras.backend.standardize_dtype(out32.dtype), "float32"
        )
        self.assertTrue(np.isfinite(ops.convert_to_numpy(out32)).all())
        self.assertAllClose(out32[0, 0, 0], 3.2e6 / np.sqrt(3), rtol=1e-4)
