from keras import ops
from keras import random

from keras_hub.src.layers.modeling.f_net_encoder import FNetEncoder
from keras_hub.src.tests.test_case import TestCase


class FNetEncoderTest(TestCase):
    def test_layer_behaviors(self):
        self.run_layer_test(
            cls=FNetEncoder,
            init_kwargs={
                "intermediate_dim": 4,
                "activation": "relu",
                "layer_norm_epsilon": 1e-5,
                "kernel_initializer": "HeNormal",
                "bias_initializer": "Zeros",
                "dropout": 0.1,
            },
            input_data=random.uniform(shape=(2, 4, 6)),
            expected_output_shape=(2, 4, 6),
            expected_num_trainable_weights=8,
            expected_num_non_trainable_variables=1,
        )

    def test_value_error_when_invalid_kernel_initializer(self):
        with self.assertRaises(ValueError):
            FNetEncoder(
                intermediate_dim=4,
                dropout=0.5,
                kernel_initializer="Invalid",
            )

    def test_training_propagation(self):
        x = random.uniform(shape=(2, 4, 6))
        layer = FNetEncoder(
            intermediate_dim=4,
            dropout=0.99999,  # Zeros out the outputs after the dropout layer
        )
        outputs = layer(x, training=True)

        # Custom computation with dropout rate sets to about 1.0
        def fourier_transform(input):
            # Apply FFT on the input and take the real part.
            input_dtype = input.dtype
            # FFT transforms do not support float16.
            input = ops.cast(input, "float32")
            real_in, imaginary_in = (input, ops.zeros_like(input))
            real_out, _ = ops.fft2((real_in, imaginary_in))
            return ops.cast(real_out, input_dtype)

        def add_and_norm(input1, input2, norm_layer):
            return norm_layer(input1 + input2)

        mixing_output = fourier_transform(x)
        mixing_output = add_and_norm(x, mixing_output, layer._mixing_layer_norm)
        x = add_and_norm(
            mixing_output,
            ops.zeros_like(mixing_output),
            layer._output_layer_norm,
        )

        self.assertAllClose(outputs, x, atol=1e-5)
