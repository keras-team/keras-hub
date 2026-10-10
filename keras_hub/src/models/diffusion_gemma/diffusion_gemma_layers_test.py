from unittest.mock import Mock
from unittest.mock import patch

import keras
import numpy as np
from keras import ops

from keras_hub.src.models.diffusion_gemma.diffusion_gemma_layers import (
    DiffusionGemmaInterleaveEmbeddings,
)
from keras_hub.src.models.diffusion_gemma.diffusion_gemma_layers import (
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
                "diffusion_gemma_layers.ops.matmul",
                side_effect=record_matmul,
            ),
            patch(
                "keras_hub.src.models.diffusion_gemma."
                "diffusion_gemma_layers.ops.softmax",
                side_effect=record_softmax,
            ),
        ):
            layer(canvas_embeds, prev_logits)

        self.assertEqual(operand_dtypes, [("float16", "float16")])
        expected_logits = ops.cast(ops.cast(prev_logits, "float16"), "float32")
        self.assertAllEqual(softmax_inputs[0], expected_logits)


class DiffusionGemmaInterleaveEmbeddingsTest(TestCase):
    def setUp(self):
        self.init_kwargs = {"num_vision_tokens_per_image": 4, "pool_size": 1}
        # Image 0 has 2 real bins, image 1 has 4. Bin `b` of image `k` holds
        # the value `100 * k + b`.
        real = [[0, 0], [1, 0], [0, 1], [1, 1]]
        # Use backend tensors: `compute_output_spec` reads shapes only from
        # tensor arguments.
        self.input_data = {
            "image_embeddings": ops.array(
                [[[[0], [1], [2], [3]], [[100], [101], [102], [103]]]],
                dtype="float32",
            ),
            "text_embeddings": -ops.ones((1, 10, 1), dtype="float32"),
            # Image 0 fills positions 1-2, image 1 fills positions 5-8.
            "vision_indices": ops.array(
                [[1, 2, 5, 6, 7, 8, 0, 0]], dtype="int32"
            ),
            "pixel_position_ids": ops.array(
                [[real[:2] + [[-1, -1]] * 2, real]], dtype="int32"
            ),
        }
        # Each image's real bins fill its own placeholders.
        self.expected_output_data = np.reshape(
            np.array(
                [-1, 0, 1, -1, -1, 100, 101, 102, 103, -1], dtype="float32"
            ),
            (1, 10, 1),
        )

    def test_layer_basics(self):
        self.run_layer_test(
            cls=DiffusionGemmaInterleaveEmbeddings,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(1, 10, 1),
            expected_output_data=self.expected_output_data,
            # The layer has no weights to train.
            run_training_check=False,
        )
