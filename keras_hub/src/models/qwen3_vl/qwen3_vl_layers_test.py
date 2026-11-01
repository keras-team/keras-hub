import numpy as np
from keras import ops

from keras_hub.src.models.qwen3_vl.qwen3_vl_layers import (
    Qwen3_VLDeepStackInject,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_layers import (
    Qwen3_VLInterleaveEmbeddings,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_layers import Qwen3_VLLayerNorm
from keras_hub.src.tests.test_case import TestCase


class Qwen3VLLayersTest(TestCase):
    def test_interleave_embeddings(self):
        text = np.zeros((2, 4, 3), dtype="float32")
        vision = np.arange(9, dtype="float32").reshape(1, 3, 3) + 1
        indices = np.array([[1, 2, 6]], dtype="int32")
        output = Qwen3_VLInterleaveEmbeddings(hidden_dim=3)(
            vision, text, indices
        )
        output = ops.convert_to_numpy(output).reshape(8, 3)
        self.assertAllClose(output[[1, 2, 6]], vision[0])
        self.assertAllClose(output[[0, 3, 4, 5, 7]], np.zeros((5, 3)))

    def test_deepstack_inject(self):
        hidden = np.ones((2, 4, 3), dtype="float32")
        feature = np.arange(9, dtype="float32").reshape(3, 3)
        indices = np.array([1, 2, 6], dtype="int32")
        output = Qwen3_VLDeepStackInject(hidden_dim=3)(hidden, feature, indices)
        expected = np.ones((8, 3), dtype="float32")
        expected[[1, 2, 6]] += feature
        self.assertAllClose(
            ops.convert_to_numpy(output).reshape(8, 3), expected
        )
        self.assertEqual(tuple(output.shape), (2, 4, 3))

    def test_deepstack_batched_features(self):
        hidden = np.zeros((2, 4, 3), dtype="float32")
        feature = np.ones((2, 2, 3), dtype="float32")
        indices = np.array([[0, 1], [4, 7]], dtype="int32")
        output = Qwen3_VLDeepStackInject(hidden_dim=3)(hidden, feature, indices)
        output = ops.convert_to_numpy(output)
        self.assertAllClose(output.sum(axis=-1), [[3, 3, 0, 0], [3, 0, 0, 3]])

    def test_layer_norm_matches_numpy(self):
        x = np.random.RandomState(0).normal(size=(2, 5, 8)).astype("float32")
        layer = Qwen3_VLLayerNorm(epsilon=1e-6)
        layer.build(x.shape)
        scale = np.linspace(0.5, 1.5, 8).astype("float32")
        layer.scale.assign(scale)
        expected = x / np.sqrt((x**2).mean(-1, keepdims=True) + 1e-6) * scale
        self.assertAllClose(layer(x), expected, atol=1e-5)

    def test_config_round_trip(self):
        for layer in (
            Qwen3_VLInterleaveEmbeddings(hidden_dim=7),
            Qwen3_VLDeepStackInject(hidden_dim=7),
            Qwen3_VLLayerNorm(head_dim=6, epsilon=1e-5),
        ):
            self.run_serialization_test(layer)

    def test_layer_norm_basics(self):
        self.run_layer_test(
            cls=Qwen3_VLLayerNorm,
            init_kwargs={"head_dim": 8, "epsilon": 1e-6},
            input_data=np.random.RandomState(0).normal(size=(2, 5, 8)),
            expected_output_shape=(2, 5, 8),
            expected_num_trainable_weights=1,
        )
