# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import keras
from keras import ops

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.utils.keras_utils import clone_initializer


@keras_nlp_export("keras_nlp.layers.FNetEncoder")
class FNetEncoder(keras.layers.Layer):
    """FNet encoder.

    This class follows the architecture of FNet encoder layer in the
    [FNet paper](https://arxiv.org/abs/2105.03824). Users can instantiate
    multiple instances of this class to stack up the encoder.

    Note on masking: In the official FNet code, padding tokens are added to the
    the input. However, the padding masks are deleted, i.e., mixing of
    all tokens is done. This is because certain frequencies will be zeroed
    out if we apply padding masks in every encoder layer. Hence, we don't
    take padding mask as input in the call() function.

    Args:
        intermediate_dim: int. The hidden size of feedforward network.
        dropout: float. The dropout value, applied in the
            feedforward network. Defaults to `0.`.
        activation: string or `keras.activations`. The
            activation function of feedforward network.
            Defaults to `"relu"`.
        layer_norm_epsilon: float. The epsilon value in layer
            normalization components. Defaults to `1e-5`.
        kernel_initializer: `str` or `keras.initializers` initializer.
            The kernel initializer for the dense layers.
            Defaults to `"glorot_uniform"`.
        bias_initializer: "string" or `keras.initializers` initializer.
            The bias initializer for the dense layers.
            Defaults to `"zeros"`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `trainable`, `dtype` etc.

    Example:

    ```python
    # Create a single FNet encoder layer.
    encoder = keras_nlp.layers.FNetEncoder(
        intermediate_dim=64)

    # Create a simple model containing the encoder.
    input = keras.Input(shape=(10, 64))
    output = encoder(input)
    model = keras.Model(inputs=input, outputs=output)

    # Call encoder on the inputs.
    input_data = np.random.uniform(size=(1, 10, 64))
    output = model(input_data)
    ```

    References:
     - [Lee-Thorp et al., 2021](https://arxiv.org/abs/2105.03824)
    """

    def __init__(
        self,
        intermediate_dim,
        dropout=0,
        activation="relu",
        layer_norm_epsilon=1e-5,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

    def build(self, inputs_shape):
        # Create layers based on input shape.
        feature_size = inputs_shape[-1]

        # Layer Norm layers.
        self._mixing_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="mixing_layer_norm",
        )
        self._mixing_layer_norm.build(inputs_shape)
        self._output_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="output_layer_norm",
        )
        self._output_layer_norm.build(inputs_shape)

        # Feedforward layers.
        self._intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="intermediate_dense",
        )
        self._intermediate_dense.build(inputs_shape)
        self._output_dense = keras.layers.Dense(
            feature_size,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="output_dense",
        )
        self._output_dense.build(
            self._intermediate_dense.compute_output_shape(inputs_shape)
        )
        self._output_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="output_dropout",
        )
        self.built = True

    def call(self, inputs, training=None):
        """Forward pass of the FNetEncoder.

        Args:
            inputs: a Tensor. The input data to TransformerEncoder, should be
                of shape [batch_size, sequence_length, feature_dim].
            training: a boolean indicating whether the layer should behave in
                training mode or in inference mode.

        Returns:
            A Tensor of the same shape as the `inputs`.
        """

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

        def feed_forward(input):
            x = self._intermediate_dense(input)
            x = self._output_dense(x)
            return self._output_dropout(x, training=training)

        mixing_output = fourier_transform(inputs)

        mixing_output = add_and_norm(
            inputs, mixing_output, self._mixing_layer_norm
        )

        feed_forward_output = feed_forward(mixing_output)

        x = add_and_norm(
            mixing_output, feed_forward_output, self._output_layer_norm
        )
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "dropout": self.dropout,
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        return inputs_shape
