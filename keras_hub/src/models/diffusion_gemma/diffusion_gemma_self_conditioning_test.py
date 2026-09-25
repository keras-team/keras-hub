from unittest.mock import Mock
from unittest.mock import patch

import keras
import numpy as np
from keras import ops

from keras_hub.src.models.diffusion_gemma.diffusion_gemma_self_conditioning import (  # noqa: E501
    DiffusionGemmaSelfConditioning,
)
from keras_hub.src.tests.test_case import TestCase


class DiffusionGemmaSelfConditioningTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.canvas_length = 6
        self.hidden_dim = 8
        self.intermediate_dim = 16
        self.vocabulary_size = 64

        self.layer = DiffusionGemmaSelfConditioning(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
        )

        self.canvas_embeds = np.random.randn(
            self.batch_size, self.canvas_length, self.hidden_dim
        ).astype("float32")

        self.prev_logits = np.random.randn(
            self.batch_size, self.canvas_length, self.vocabulary_size
        ).astype("float32")

    def _attach_embedding(self):
        """Wire a small embedding layer so the full forward pass works."""
        embedding_layer = keras.layers.Embedding(
            self.vocabulary_size, self.hidden_dim
        )
        embedding_layer.build((None,))
        object.__setattr__(
            self.layer, "_token_embedding_layer", embedding_layer
        )

    def test_output_shape_no_prev_logits(self):
        self.layer.build((self.batch_size, self.canvas_length, self.hidden_dim))
        out = self.layer(
            ops.convert_to_tensor(self.canvas_embeds), prev_logits=None
        )
        self.assertEqual(
            out.shape,
            (self.batch_size, self.canvas_length, self.hidden_dim),
        )

    def test_output_shape_with_prev_logits(self):
        self.layer.build((self.batch_size, self.canvas_length, self.hidden_dim))
        self._attach_embedding()
        out = self.layer(
            ops.convert_to_tensor(self.canvas_embeds),
            prev_logits=ops.convert_to_tensor(self.prev_logits),
        )
        self.assertEqual(
            out.shape,
            (self.batch_size, self.canvas_length, self.hidden_dim),
        )

    def test_first_step_is_post_norm_of_embeds(self):
        self.layer.build((self.batch_size, self.canvas_length, self.hidden_dim))
        embeds_t = ops.convert_to_tensor(self.canvas_embeds)
        out = self.layer(embeds_t, prev_logits=None)
        # post_norm is L2 normalization applied to canvas embeddings.
        expected = self.layer.post_norm(embeds_t)
        self.assertAllClose(
            ops.convert_to_numpy(out),
            ops.convert_to_numpy(expected),
        )

    def test_conditioning_changes_output(self):
        self.layer.build((self.batch_size, self.canvas_length, self.hidden_dim))
        self._attach_embedding()
        embeds_t = ops.convert_to_tensor(self.canvas_embeds)
        out_no_cond = self.layer(embeds_t, prev_logits=None)
        out_with_cond = self.layer(
            embeds_t,
            prev_logits=ops.convert_to_tensor(self.prev_logits),
        )
        self.assertNotAllClose(
            ops.convert_to_numpy(out_no_cond),
            ops.convert_to_numpy(out_with_cond),
        )

    def test_serialization(self):
        self.run_serialization_test(self.layer)

    def test_self_conditioning_matmul_uses_embedding_dtype(self):
        layer = DiffusionGemmaSelfConditioning(
            hidden_dim=4,
            intermediate_dim=8,
            dtype="float16",
        )
        canvas_embeds = ops.ones((1, 2, 4), dtype="float16")
        prev_logits = ops.array(
            [
                [
                    [0.10001, 0.20002, 0.30003, 0.40004, 0.50005, 0.60006],
                    [0.70007, 0.80008, 0.90009, 1.0001, 1.1001, 1.2001],
                ]
            ],
            dtype="float32",
        )
        embedding_weights = ops.ones((6, 4), dtype="float16")

        layer._token_embedding_layer = Mock(embeddings=embedding_weights)

        operand_dtypes = []
        softmax_inputs = []
        original_matmul = ops.matmul
        original_softmax = ops.softmax

        def record_matmul(x, y):
            operand_dtypes.append(
                (
                    keras.backend.standardize_dtype(x.dtype),
                    keras.backend.standardize_dtype(y.dtype),
                )
            )
            return original_matmul(x, y)

        def record_softmax(x, axis=-1):
            softmax_inputs.append(ops.convert_to_numpy(x))
            return original_softmax(x, axis=axis)

        with (
            patch(
                "keras_hub.src.models.diffusion_gemma."
                "diffusion_gemma_self_conditioning.ops.matmul",
                side_effect=record_matmul,
            ),
            patch(
                "keras_hub.src.models.diffusion_gemma."
                "diffusion_gemma_self_conditioning.ops.softmax",
                side_effect=record_softmax,
            ),
        ):
            layer(canvas_embeds, prev_logits)

        self.assertEqual(operand_dtypes, [("float16", "float16")])
        expected_logits = ops.cast(ops.cast(prev_logits, "float16"), "float32")
        self.assertAllEqual(softmax_inputs[0], expected_logits)
