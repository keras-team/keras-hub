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


class FNetEncoder(keras.layers.Layer):
    """FNet encoder.

    This class follows the architecture of FNet encoder layer in paper
    "FNet: Mixing Tokens with Fourier Transforms"
    (https://arxiv.org/abs/2105.03824). Users can instantiate multiple instances
    of this class to stack up the encoder.

    Args:
        intermediate_dim: int, defaults to 3072. The hidden size of feedforward
            network.
        dropout: float, defaults to 0.1. The dropout value, applied in the
            feedforward network.
        activation: string or `tf.keras.activations`, defaults to "gelu". The
            activation function of feedforward network.
        layer_norm_epsilon: float, defaults to 1e-12. The epsilon value in layer
            normalization components.
        name: string, defaults to None. The name of the layer.
        **kwargs: other keyword arguments.

    Examples:

    ```python
    # Create a single FNet encoder layer.
    encoder = keras_nlp.layers.FNetEncoder(
        intermediate_dim=64)

    # Create a simple model containing the encoder.
    input = tf.keras.Input(shape=[4, 6])
    output = encoder(input)
    model = tf.keras.Model(inputs=input, outputs=output)

    # Call encoder on the inputs.
    input_data = tf.random.uniform(shape=[1, 10, 64])
    output = model(input_data)

    ```

    References:
        [Lee-Thorp et al., 2021](https://arxiv.org/abs/2105.03824)
    """

    def __init__(
        self,
        intermediate_dim=3072,
        dropout=0.1,
        activation="gelu",
        layer_norm_epsilon=1e-12,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self._built = False

    def _build(self, input_shape):
        # Create layers based on input shape.
        self._built = True
        feature_size = input_shape[-1]

        # Layer Norm layers.
        self._mixing_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )
        self._output_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon
        )

        # Feedforward layer.
        self._intermediate_dense = keras.layers.Dense(
            self.intermediate_dim, activation=self.activation
        )
        self._output_dense = keras.layers.Dense(feature_size)
        self._output_dropout = keras.layers.Dropout(rate=self.dropout)

    def _fourier_transform(self, input):
        # Apply FFT on the input and take the real part.
        # Before we apply fourier transform, let's convert the dtype of the
        # input tensor to complex64.
        input = tf.cast(input, tf.complex64)
        mixing_output = tf.math.real(tf.signal.fft2d(input))
        return mixing_output

    def _add_and_norm(self, input1, input2, norm_layer):
        return norm_layer(input1 + input2)

    def _feed_forward(self, input):
        x = self._intermediate_dense(input)
        x = self._output_dense(x)
        return self._output_dropout(x)

    def call(self, inputs):
        """Forward pass of the FNetEncoder.

        Args:
            inputs: a Tensor. The input data to TransformerEncoder, should be
                of shape [batch_size, sequence_length, feature_dim].

        Returns:
            A Tensor of the same shape as the `inputs`.
        """

        if not self._built:
            self._build(inputs.shape)

        # Apply fourier transform on the input. Note: We don't have padding
        # tokens in the official FNet code.
        # https://github.com/google-research/google-research/blob/master/f_net/layers.py#L137
        mixing_output = self._fourier_transform(inputs)

        # LayerNorm layer.
        mixing_output = self._add_and_norm(
            inputs, mixing_output, self._mixing_layer_norm
        )

        # Feedforward layer.
        feed_forward_output = self._feed_forward(mixing_output)

        # LayerNorm layer.
        x = self._add_and_norm(
            mixing_output, feed_forward_output, self._output_layer_norm
        )
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "dropout": self.dropout,
                "activation": self.activation,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
