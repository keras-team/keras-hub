import keras
from absl.testing import parameterized
from keras import ops
from keras import random
from keras.src.backend import get_keras_mask
from keras.src.backend import set_keras_mask

from keras_hub.src.layers.modeling.transformer_encoder import TransformerEncoder
from keras_hub.src.tests.test_case import TestCase


class TransformerEncoderTest(TestCase):
    @parameterized.named_parameters(
        ("without_norm_first", False),
        ("with_norm_first", True),
    )
    def test_layer_behaviors(self, normalize_first):
        self.run_layer_test(
            cls=TransformerEncoder,
            init_kwargs={
                "intermediate_dim": 4,
                "num_heads": 2,
                "normalize_first": normalize_first,
                "activation": "relu",
                "layer_norm_epsilon": 1e-05,
                "kernel_initializer": "HeNormal",
                "bias_initializer": "Zeros",
                "dropout": 0.1,
            },
            input_data=random.uniform(shape=(2, 4, 6)),
            expected_output_shape=(2, 4, 6),
            expected_num_trainable_weights=16,
            expected_num_non_trainable_variables=3,  # dropout rng seeds
        )

    @parameterized.named_parameters(
        ("without_norm_first", False),
        ("with_norm_first", True),
    )
    def test_valid_call(self, normalize_first):
        encoder = TransformerEncoder(
            intermediate_dim=4,
            num_heads=2,
            normalize_first=normalize_first,
        )
        model = keras.Sequential(
            [
                keras.Input(shape=(4, 6)),
                encoder,
            ]
        )
        input = random.uniform(shape=[2, 4, 6])
        model(input)

    def test_valid_call_with_mask(self):
        encoder = TransformerEncoder(
            intermediate_dim=4,
            num_heads=2,
        )
        encoder.build([2, 4, 6])
        input = random.uniform(shape=[2, 4, 6])
        mask = input[:, :, 0] < 0.5
        encoder(input, mask)

    def test_value_error_when_invalid_kernel_inititalizer(self):
        with self.assertRaises(ValueError):
            TransformerEncoder(
                intermediate_dim=4,
                num_heads=2,
                dropout=0.5,
                kernel_initializer="Invalid",
            )

    def test_training_propagation(self):
        encoder = TransformerEncoder(
            intermediate_dim=4,
            num_heads=2,
            dropout=0.99999,  # Zeros out the outputs after the dropout layer
        )
        inputs = random.uniform(shape=[1, 4, 6])
        outputs = encoder(inputs, training=True)

        # Custom computation with dropout rates set to about 1.0
        x = inputs
        x = encoder._self_attention_layer_norm(x)
        x = encoder._feedforward_layer_norm(x)

        self.assertAllClose(outputs, x, atol=1e-5)

    def test_mask_propagation(self):
        encoder = TransformerEncoder(
            intermediate_dim=4,
            num_heads=2,
        )
        inputs = random.uniform(shape=[1, 4, 6])
        mask = ops.array([[True, True, False, False]])
        set_keras_mask(inputs, mask)
        outputs = encoder(inputs)
        self.assertAllEqual(get_keras_mask(outputs), mask)

    def test_attention_scores(self):
        encoder = TransformerEncoder(intermediate_dim=4, num_heads=2)
        inputs = random.uniform(shape=[1, 4, 6])
        outputs, attention_scores = encoder(
            inputs, return_attention_scores=True
        )
        self.assertAllEqual(outputs.shape, inputs.shape)

        # attention scores shape
        # (batch_size, num_of_attn_heads, seq_length, seq_length)
        self.assertAllEqual(attention_scores.shape, [1, 2, 4, 4])
