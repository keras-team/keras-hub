import numpy as np
from keras import ops

from keras_hub.src.models.muse_glimmer.muse_glimmer_layers import (
    MuseGlimmerCenteredRMSNorm,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_layers import (
    MuseGlimmerInterleaveEmbeddings,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_layers import (
    MuseGlimmerRMSNorm,
)
from keras_hub.src.tests.test_case import TestCase


class MuseGlimmerLayersTest(TestCase):
    def test_scaleless_rms_norm(self):
        self.run_serialization_test(MuseGlimmerRMSNorm(with_scale=False))
        layer = MuseGlimmerRMSNorm(with_scale=False)
        x = ops.convert_to_tensor(np.random.randn(2, 3, 8).astype("float32"))
        output = layer(x)
        self.assertEqual(ops.shape(output), (2, 3, 8))
        self.assertEqual(len(layer.weights), 0)

    def test_scaled_rms_norm_has_weight(self):
        self.run_serialization_test(MuseGlimmerRMSNorm(with_scale=True))
        layer = MuseGlimmerRMSNorm(with_scale=True)
        x = ops.convert_to_tensor(np.random.randn(2, 3, 8).astype("float32"))
        layer(x)
        self.assertEqual(len(layer.weights), 1)

    def test_centered_rms_norm_identity_at_init(self):
        self.run_serialization_test(MuseGlimmerCenteredRMSNorm(eps=1e-6))
        # Weight initialized to zero -> `(1 + 0) * norm(x)` == plain RMSNorm.
        layer = MuseGlimmerCenteredRMSNorm(eps=1e-6)
        x = ops.convert_to_tensor(np.random.randn(2, 3, 8).astype("float32"))
        output = layer(x)
        expected = x * ops.rsqrt(
            ops.mean(ops.square(x), axis=-1, keepdims=True) + 1e-6
        )
        self.assertAllClose(output, expected, atol=1e-5)

    def test_interleave_embeddings(self):
        self.run_serialization_test(
            MuseGlimmerInterleaveEmbeddings(hidden_dim=4)
        )
        layer = MuseGlimmerInterleaveEmbeddings(hidden_dim=4)
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
