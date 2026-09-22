import numpy as np
from keras import ops

from keras_hub.src.models.muse_glimmer.muse_glimmer_layers import (
    MuseGlimmerCenteredRMSNorm,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_layers import (
    MuseGlimmerContextProjection,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_layers import (
    MuseGlimmerInterleaveEmbeddings,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_layers import (
    MuseGlimmerRMSNorm,
)
from keras_hub.src.tests.test_case import TestCase


class MuseGlimmerLayersTest(TestCase):
    def test_scaleless_rms_norm_basics(self):
        self.run_layer_test(
            cls=MuseGlimmerRMSNorm,
            init_kwargs={"with_scale": False},
            input_data=ops.convert_to_tensor(
                np.random.randn(2, 3, 8).astype("float32")
            ),
            expected_output_shape=(2, 3, 8),
            expected_num_trainable_weights=0,
            run_precision_checks=False,
        )

    def test_scaled_rms_norm_basics(self):
        self.run_layer_test(
            cls=MuseGlimmerRMSNorm,
            init_kwargs={"with_scale": True},
            input_data=ops.convert_to_tensor(
                np.random.randn(2, 3, 8).astype("float32")
            ),
            expected_output_shape=(2, 3, 8),
            expected_num_trainable_weights=1,
            run_precision_checks=False,
        )

    def test_centered_rms_norm_basics(self):
        # Weight initialized to zero -> `(1 + 0) * norm(x)` == plain RMSNorm.
        input_data = ops.convert_to_tensor(
            np.random.randn(2, 3, 8).astype("float32")
        )
        self.run_layer_test(
            cls=MuseGlimmerCenteredRMSNorm,
            init_kwargs={"eps": 1e-6},
            input_data=input_data,
            expected_output_shape=(2, 3, 8),
            expected_output_data=input_data
            * ops.rsqrt(
                ops.mean(ops.square(input_data), axis=-1, keepdims=True) + 1e-6
            ),
            expected_num_trainable_weights=1,
            run_precision_checks=False,
        )

    def test_context_projection(self):
        self.run_layer_test(
            cls=MuseGlimmerContextProjection,
            init_kwargs={"hidden_dim": 8, "eps": 1e-5},
            input_data=ops.convert_to_tensor(
                np.random.randn(2, 4, 6).astype("float32")
            ),
            expected_output_shape=(2, 4, 8),
            expected_num_trainable_weights=2,
        )

    def test_interleave_embeddings(self):
        layer = MuseGlimmerInterleaveEmbeddings(hidden_dim=4)
        self.run_serialization_test(layer)
        text_emb = ops.zeros((1, 6, 4))
        vision_emb = np.ones((2, 4), dtype="float32")
        indices = ops.convert_to_tensor([1, 3], dtype="int32")
        result = layer(
            image_embeddings=vision_emb,
            text_embeddings=text_emb,
            vision_indices=indices,
        )
        result_np = ops.convert_to_numpy(result)
        np.testing.assert_allclose(result_np[0, 0, :], 0.0)
        np.testing.assert_allclose(result_np[0, 1, :], 1.0)
        np.testing.assert_allclose(result_np[0, 2, :], 0.0)
        np.testing.assert_allclose(result_np[0, 3, :], 1.0)

    def test_interleave_embeddings_2d_indices_with_batch_offset(self):
        layer = MuseGlimmerInterleaveEmbeddings(hidden_dim=4)
        text_emb = ops.zeros((2, 6, 4))
        vision_emb = np.stack(
            [np.full((4,), i, dtype="float32") for i in range(4)]
        )
        # Local per-example positions: example 0 uses cols 1, 3; example 1
        # uses cols 0, 2. Without a batch offset these collide in the
        # flattened buffer.
        indices = ops.convert_to_tensor([[1, 3], [0, 2]], dtype="int32")
        result = layer(
            image_embeddings=vision_emb,
            text_embeddings=text_emb,
            vision_indices=indices,
        )
        result_np = ops.convert_to_numpy(result)
        np.testing.assert_allclose(result_np[0, 1, :], 0.0)
        np.testing.assert_allclose(result_np[0, 3, :], 1.0)
        np.testing.assert_allclose(result_np[1, 0, :], 2.0)
        np.testing.assert_allclose(result_np[1, 2, :], 3.0)
