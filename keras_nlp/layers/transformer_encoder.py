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

"""Transformer encoder block implementation based on `keras.layers.Layer`."""

from tensorflow import keras

from keras_nlp.layers.transformer_layer_utils import (  # isort:skip
    merge_padding_and_attention_mask,
)


class TransformerEncoder(keras.layers.Layer):
    """Transformer encoder.

    This class follows the architecture of transformer encoder layer in paper
    "Attention is All You Need" (https://arxiv.org/abs/1706.03762). Users can
    instantiate multiple instances of this class to stack up the encoder.

    Args:
        intermediate_dim: int, the hidden size of feedforward network.
        num_heads: int, the number of heads in MultiHeadAttention.
        dropout: float, defaults to 0. the dropout value, shared by
            MultiHeadAttention and feedforward network.
        activation: string or `tf.keras.activations`, defaults to "relu". the
            activation function of feedforward network.
        layer_norm_epsilon: float, defaults to 1e-5. The epsilon value in layer
            normalization components.
        kernel_initializer: string or tf.keras.initializers initializer,
            defaults to "glorot_uniform". The kernel initializer for
            the dense and multiheaded attention layers.
        bias_initializer: string or tf.keras.initializers initializer,
            defaults to "zeros". The bias initializer for
            the dense and multiheaded attention layers.
        name: string, defaults to None. The name of the layer.
        **kwargs: other keyword arguments.

    Examples:

    ```python
    # Create a single transformer encoder layer.
    encoder = keras_nlp.layers.TransformerEncoder(
        intermediate_dim=64, num_heads=8)

    # Create a simple model containing the encoder.
    input = tf.keras.Input(shape=[4, 6])
    output = encoder(input)
    model = tf.keras.Model(inputs=input, outputs=output)

    # Call encoder on the inputs.
    input_data = tf.random.uniform(shape=[1, 10, 64])
    output = model(input_data)

    ```

    References:
        [Vaswani et al., 20XX](https://arxiv.org/abs/1706.03762)
    """

    def __init__(
        self,
        intermediate_dim,
        num_heads,
        dropout=0,
        activation="relu",
        layer_norm_epsilon=1e-05,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self._built = False

    def _build(self, input_shape):
        # Create layers based on input shape.
        self._built = True
        feature_size = input_shape[-1]
        self._attention_head_size = int(feature_size // self.num_heads)
        self._multi_head_attention_layer = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self._attention_head_size,
            value_dim=self._attention_head_size,
            dropout=self.dropout,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )

        self._attention_layernorm = keras.layers.LayerNormalization()
        self._feedforward_layernorm = keras.layers.LayerNormalization()

        self._attention_dropout = keras.layers.Dropout(rate=self.dropout)

        self._intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        self._output_dense = keras.layers.Dense(
            feature_size,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        self._output_dropout = keras.layers.Dropout(rate=self.dropout)

    def _add_and_norm(self, input1, input2, norm_layer):
        return norm_layer(input1 + input2)

    def _feed_forward(self, input):
        x = self._intermediate_dense(input)
        x = self._output_dense(x)
        return self._output_dropout(x)

    def call(self, inputs, padding_mask=None, attention_mask=None):
        """Forward pass of the TransformerEncoder.

        Args:
            inputs: a Tensor. The input data to TransformerEncoder, should be
                of shape [batch_size, sequence_length, feature_dim].
            padding_mask: a boolean Tensor. It indicates if the token should be
                masked because the token is introduced due to padding.
                `padding_mask` should have shape [batch_size, sequence_length].
                False means the certain certain is masked out.
            attention_mask: a boolean Tensor. Customized mask used to mask out
                certain tokens. `attention_mask` should have shape
                [batch_size, sequence_length, sequence_length].

        Returns:
            A Tensor of the same shape as the `inputs`.
        """

        if not self._built:
            self._build(inputs.shape)

        mask = merge_padding_and_attention_mask(
            inputs,
            padding_mask,
            attention_mask,
        )

        # Self attention.
        attended = self._multi_head_attention_layer(
            inputs, inputs, inputs, attention_mask=mask
        )
        attended = self._attention_dropout(attended)
        attended = self._add_and_norm(
            inputs,
            attended,
            self._attention_layernorm,
        )
        # Feedforward.
        feed_forward_output = self._feed_forward(attended)
        return self._add_and_norm(
            attended, feed_forward_output, self._feedforward_layernorm
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
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
