import keras

from keras_hub.src.models.moonshine.moonshine_layers import MoonshineArange
from keras_hub.src.models.moonshine.moonshine_layers import (
    MoonshineInvFreqInitializer,
)
from keras_hub.src.models.moonshine.moonshine_layers import MoonshineLinearGeLU
from keras_hub.src.models.moonshine.moonshine_layers import (
    MoonshineReversibleEmbedding,
)
from keras_hub.src.models.moonshine.moonshine_layers import (
    MoonshineRotaryEmbedding,
)
from keras_hub.src.models.moonshine.moonshine_layers import MoonshineSwiGLU
from keras_hub.src.tests.test_case import TestCase


class MoonshineLayersTest(TestCase):
    def test_moonshine_inv_freq_initializer(self):
        initializer = MoonshineInvFreqInitializer(
            inv_freq_dim=32,
            max_position_embeddings=2048,
            base_value=10000,
            scaling_factor=1.0,
        )
        inv_freq = initializer(shape=(32,), dtype="float32")
        self.assertEqual(inv_freq.shape, (32,))
        self.assertAlmostEqual(inv_freq[0], 1.0, places=5)
        expected_freq = 1.0 / (10000 ** (1.0 / 32))
        self.assertAlmostEqual(float(inv_freq[1]), expected_freq, places=5)

    def test_moonshine_rotary_embedding(self):
        self.run_layer_test(
            cls=MoonshineRotaryEmbedding,
            init_kwargs={
                "head_dim": 64,
                "max_position_embeddings": 2048,
                "base_value": 10000,
                "scaling_factor": 1.0,
                "partial_rotary_factor": 0.62,
            },
            input_data=keras.ops.arange(10, dtype="float32"),
            expected_output_shape=(10, 38),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=1,
            expected_num_non_trainable_variables=1,
            run_precision_checks=False,
        )

    def test_moonshine_arange(self):
        layer = MoonshineArange()
        input_data = keras.ops.array([10])
        output = layer(input_data)
        self.assertEqual(output.shape, (10,))
        self.assertAllEqual(output, keras.ops.arange(10))
        keras_tensor_input = keras.KerasTensor(shape=(), dtype="int32")
        keras_tensor_output = layer.compute_output_spec(keras_tensor_input)
        self.assertEqual(keras_tensor_output.shape, (None,))

    def test_moonshine_swiglu(self):
        self.run_layer_test(
            cls=MoonshineSwiGLU,
            init_kwargs={"hidden_dim": 64, "feedforward_expansion_factor": 4},
            input_data=keras.random.uniform((2, 10, 64), dtype="float32"),
            expected_output_shape=(2, 10, 64),
            expected_num_trainable_weights=4,
            expected_num_non_trainable_weights=0,
            run_precision_checks=False,
        )

    def test_moonshine_linear_gelu(self):
        self.run_layer_test(
            cls=MoonshineLinearGeLU,
            init_kwargs={"hidden_dim": 64, "feedforward_expansion_factor": 4},
            input_data=keras.random.uniform((2, 10, 64), dtype="float32"),
            expected_output_shape=(2, 10, 64),
            expected_num_trainable_weights=4,
            expected_num_non_trainable_weights=0,
            run_precision_checks=False,
        )

    def test_moonshine_reversible_embedding(self):
        # Forward mode.
        self.run_layer_test(
            cls=MoonshineReversibleEmbedding,
            init_kwargs={
                "vocabulary_size": 10000,
                "hidden_dim": 64,
            },
            input_data=keras.random.randint((2, 10), 0, 10000),
            expected_output_shape=(2, 10, 64),
            expected_num_trainable_weights=1,
            expected_num_non_trainable_weights=0,
            run_training_check=True,
        )

        # Reverse mode.
        layer = MoonshineReversibleEmbedding(
            vocabulary_size=10000, hidden_dim=64
        )
        hidden_states = keras.random.uniform((2, 10, 64))
        logits = layer(hidden_states, reverse=True)
        self.assertEqual(logits.shape, (2, 10, 10000))
