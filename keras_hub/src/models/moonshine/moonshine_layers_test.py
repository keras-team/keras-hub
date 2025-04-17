import keras

from keras_hub.src.models.moonshine.moonshine_layers import MoonshineMLP
from keras_hub.src.models.moonshine.moonshine_layers import (
    MoonshineRotaryEmbedding,
)
from keras_hub.src.tests.test_case import TestCase


class MoonshineLayersTest(TestCase):
    def test_moonshine_rotary_embedding(self):
        layer = MoonshineRotaryEmbedding(
            head_dim=64,
            max_position_embeddings=2048,
            base_value=10000,
            rope_scaling=None,
            partial_rotary_factor=0.62,
            dtype="float32",
        )
        input_data = keras.ops.arange(10, dtype="float32")
        output_data = layer(input_data)
        expected_output_shape = (1, 10, 38)
        self.assertEqual(keras.ops.shape(output_data), expected_output_shape)
        self.assertEqual(len(layer.trainable_weights), 0)
        self.assertEqual(len(layer.non_trainable_weights), 1)
        self.assertEqual(len(layer.non_trainable_variables), 1)

    def test_moonshine_rotary_embedding_dynamic(self):
        layer = MoonshineRotaryEmbedding(
            head_dim=64,
            max_position_embeddings=10,
            base_value=10000,
            rope_scaling={"rope_type": "dynamic"},
            partial_rotary_factor=1.0,
        )
        # Compute original inverse frequencies.
        rotary_dim = 32  # Derived from head_dim = 64, partial_rotary_factor = 1
        arange = keras.ops.arange(0, rotary_dim, dtype="float32")
        original_inv_freq = 1.0 / (10000 ** (arange / rotary_dim))

        # seq_len = 5 < 10.
        position_ids = keras.ops.arange(5, dtype="int32")[None, :]  # [1, 5]
        cos1, sin1 = layer(None, position_ids=position_ids)  # [1, 5, 32]
        expected_cos1 = keras.ops.cos(original_inv_freq)
        expected_sin1 = keras.ops.sin(original_inv_freq)
        self.assertAllClose(cos1[0, 1, :], expected_cos1, rtol=1e-5)
        self.assertAllClose(sin1[0, 1, :], expected_sin1, rtol=1e-5)

        # seq_len = 15 > 10.
        position_ids = keras.ops.arange(15, dtype="int32")[None, :]  # [1, 15]
        cos2, sin2 = layer(None, position_ids=position_ids)  # [1, 15, 32]
        scaling = 10 / 15  # 2 / 3
        expected_cos2 = keras.ops.cos(original_inv_freq * scaling)
        expected_sin2 = keras.ops.sin(original_inv_freq * scaling)
        self.assertAllClose(cos2[0, 1, :], expected_cos2, rtol=1e-5)
        self.assertAllClose(sin2[0, 1, :], expected_sin2, rtol=1e-5)

        # seq_len = 8 < 10, should reset.
        position_ids = keras.ops.arange(8, dtype="int32")[None, :]  # [1, 8]
        cos3, sin3 = layer(None, position_ids=position_ids)  # [1, 8, 32]
        self.assertAllClose(cos3[0, 1, :], expected_cos1, rtol=1e-5)
        self.assertAllClose(sin3[0, 1, :], expected_sin1, rtol=1e-5)

    def test_moonshine_mlp_swiglu(self):
        self.run_layer_test(
            cls=MoonshineMLP,
            init_kwargs={
                "hidden_dim": 64,
                "feedforward_expansion_factor": 4,
                "use_swiglu_activation": True,
            },
            input_data=keras.random.uniform((2, 10, 64), dtype="float32"),
            expected_output_shape=(2, 10, 64),
            expected_num_trainable_weights=4,
            expected_num_non_trainable_weights=0,
            run_precision_checks=False,
        )

    def test_moonshine_mlp_linear_gelu(self):
        self.run_layer_test(
            cls=MoonshineMLP,
            init_kwargs={
                "hidden_dim": 64,
                "feedforward_expansion_factor": 4,
                "use_swiglu_activation": False,
            },
            input_data=keras.random.uniform((2, 10, 64), dtype="float32"),
            expected_output_shape=(2, 10, 64),
            expected_num_trainable_weights=4,
            expected_num_non_trainable_weights=0,
            run_precision_checks=False,
        )
