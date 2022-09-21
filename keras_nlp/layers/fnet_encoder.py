# Copyright 2022 The KerasNLP Authors
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

"""FNet encoder block implementation based on `keras.layers.Layer`."""

import tensorflow as tf
from tensorflow import keras

from keras_nlp.utils.keras_utils import clone_initializer


@keras.utils.register_keras_serializable(package="keras_nlp")
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
        dropout: float, defaults to 0. The dropout value, applied in the
            feedforward network.
        activation: string or `keras.activations`, defaults to "relu". The
            activation function of feedforward network.
        layer_norm_epsilon: float, defaults to 1e-5. The epsilon value in layer
            normalization components.
        kernel_initializer: "string" or `keras.initializers` initializer,
            defaults to "glorot_uniform". The kernel initializer for the dense
            layers.
        bias_initializer: "string" or `keras.initializers` initializer,
            defaults to "zeros". The bias initializer for the dense layers.
        name: string, defaults to None. The name of the layer.
        **kwargs: other keyword arguments.

    Examples:

    ```python
    # Create a single FNet encoder layer.
    encoder = keras_nlp.layers.FNetEncoder(
        intermediate_dim=64)

    # Create a simple model containing the encoder.
    input = keras.Input(shape=[10, 64])
    output = encoder(input)
    model = keras.Model(inputs=input, outputs=output)

    # Call encoder on the inputs.
    input_data = tf.random.uniform(shape=[1, 10, 64])
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
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

    def build(self, input_shape):
        # Create layers based on input shape.
        feature_size = input_shape[-1]

        # Layer Norm layers.
        self._mixing_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )
        self._output_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )

        # Feedforward layers.
        self._intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._output_dense = keras.layers.Dense(
            feature_size,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._output_dropout = keras.layers.Dropout(rate=self.dropout)

    def call(self, inputs):
        """Forward pass of the FNetEncoder.

        Args:
            inputs: a Tensor. The input data to TransformerEncoder, should be
                of shape [batch_size, sequence_length, feature_dim].

        Returns:
            A Tensor of the same shape as the `inputs`.
        """

        def fourier_transform(input):
            # Apply FFT on the input and take the real part.
            # Before we apply fourier transform, let's convert the dtype of the
            # input tensor to complex64.
            input = tf.cast(input, tf.complex64)
            mixing_output = tf.math.real(tf.signal.fft2d(input))
            return mixing_output

        def add_and_norm(input1, input2, norm_layer):
            return norm_layer(input1 + input2)

        def feed_forward(input):
            x = self._intermediate_dense(input)
            x = self._output_dense(x)
            return self._output_dropout(x)

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
