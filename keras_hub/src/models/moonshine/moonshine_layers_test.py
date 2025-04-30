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
            partial_rotary_factor=0.62,
            dtype="float32",
        )
        input_data = keras.ops.arange(10, dtype="float32")
        output_data = layer(input_data)
        expected_output_shape = (10, 38)
        self.assertEqual(keras.ops.shape(output_data), expected_output_shape)
        self.assertEqual(len(layer.trainable_weights), 0)
        self.assertEqual(len(layer.non_trainable_weights), 1)
        self.assertEqual(len(layer.non_trainable_variables), 1)

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
