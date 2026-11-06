import keras
import numpy as np
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


class MobileBertLayersTest(TestCase):
    def setUp(self):
        self.compute_dtype = keras.config.dtype_policy().compute_dtype

    def test_no_norm(self):
        layer = NoNorm()
        layer.build((2, 4, 16))
        layer.gamma.assign(np.ones((16,)) * 2.0)
        layer.beta.assign(np.ones((16,)) * 3.0)
        input_data = ops.ones((2, 4, 16), dtype=self.compute_dtype)
        expected = ops.ones((2, 4, 16), dtype=self.compute_dtype) * 5.0
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

    def test_bottleneck(self):
        self.run_layer_test(
            cls=MobileBertBottleneck,
            init_kwargs={
                "intra_bottleneck_dim": 8,
                "normalization_type": "no_norm",
            },
            input_data=ops.ones((2, 4, 16), dtype=self.compute_dtype),
            expected_output_shape=(2, 4, 8),
            expected_num_trainable_weights=4,
            expected_num_non_trainable_weights=0,
            expected_num_non_trainable_variables=0,
        )

    def test_attention(self):
        self.run_layer_test(
            cls=MobileBertAttention,
            init_kwargs={
                "num_heads": 2,
                "intra_bottleneck_dim": 16,
                "normalization_type": "no_norm",
            },
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

    def test_feed_forward(self):
        self.run_layer_test(
            cls=MobileBertFeedForward,
            init_kwargs={
                "intermediate_dim": 32,
                "output_dim": 16,
                "normalization_type": "no_norm",
            },
            input_data=ops.ones((2, 4, 16), dtype=self.compute_dtype),
            expected_output_shape=(2, 4, 16),
            expected_num_trainable_weights=6,
            expected_num_non_trainable_weights=0,
            expected_num_non_trainable_variables=0,
        )

    def test_encoder_layer(self):
        self.run_layer_test(
            cls=MobileBertEncoderLayer,
            init_kwargs={
                "hidden_dim": 32,
                "intra_bottleneck_dim": 16,
                "intermediate_dim": 32,
                "num_heads": 2,
                "num_feedforward_networks": 2,
                "normalization_type": "no_norm",
            },
            input_data=ops.ones((2, 4, 32), dtype=self.compute_dtype),
            expected_output_shape=(2, 4, 32),
            expected_num_trainable_weights=4 + 4 + 10 + (6 * 2) + 4,
            expected_num_non_trainable_weights=0,
            expected_num_non_trainable_variables=1,
        )

    def test_encoder_layer_padding_independence(self):
        np.random.seed(42)
        layer = MobileBertEncoderLayer(
            hidden_dim=32,
            intra_bottleneck_dim=16,
            intermediate_dim=32,
            num_heads=2,
            num_feedforward_networks=2,
            normalization_type="no_norm",
        )
        inputs_unmasked_np = np.random.uniform(size=(2, 4, 32))
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
        np.random.seed(42)
        vocab_size = 100
        embedding_dim = 16
        hidden_dim = 32

        # Create random weights
        token_emb_w = np.random.uniform(size=(vocab_size, embedding_dim))
        extra_kernel = np.random.uniform(size=(embedding_dim, vocab_size))
        bias = np.random.uniform(size=(vocab_size,))

        token_embedding = ReversibleEmbedding(vocab_size, embedding_dim)
        token_embedding.build((2, 4))
        token_embedding.set_weights([token_emb_w])

        layer = MobileBertMaskedLMHead(
            vocabulary_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            token_embedding=token_embedding,
        )
        inputs_np = np.random.uniform(size=(2, 4, hidden_dim))
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

        # TF32 on GPU (jax/tf) gives ~3e-3 abs drift on these matmuls; tolerance
        # chosen so the layer-logic mutations below still fail by >0.3
        self.assertAllClose(
            output,
            ops.convert_to_tensor(expected, dtype=self.compute_dtype),
            atol=1e-2,
            rtol=1e-2,
        )

        self.run_layer_test(
            cls=MobileBertMaskedLMHead,
            init_kwargs={
                "vocabulary_size": 100,
                "embedding_dim": 16,
                "hidden_dim": 32,
                "token_embedding": ReversibleEmbedding(100, 16),
            },
            input_data={
                "inputs": ops.ones((2, 4, 32), dtype=self.compute_dtype),
            },
            expected_output_shape=(2, 4, 100),
            expected_num_trainable_weights=6,
            expected_num_non_trainable_weights=0,
            expected_num_non_trainable_variables=0,
            run_training_check=False,
        )
