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

"""Transformer decoder block implementation based on `keras.layers.Layer`."""

import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers.transformer_layer_utils import (  # isort:skip
    compute_causal_mask,
    merge_padding_and_attention_mask,
)


class TransformerDecoder(keras.layers.Layer):
    """Transformer decoder.

    This class follows the architecture of transformer decoder layer in paper
    "Attention is All You Need" (https://arxiv.org/abs/1706.03762). Users can
    instantiate multiple instances of this class to stack up the decoder.

    Args:
        intermediate_dim: int, the hidden size of feedforward network.
        num_heads: int, the number of heads in MultiHeadAttention.
        dropout: float, defaults to 0. the dropout value, shared by
            MultiHeadAttention and feedforward network.
        activation: string or tf.keras.activations, defaults to "relu". the
            activation function of feedforward network.
        layer_norm_epsilon: float, defaults to 1e-5. The eps value in layer
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
    # Create a single transformer decoder layer.
    decoder = keras_nlp.layers.TransformerDecoder(
        intermediate_dim=64, num_heads=8)

    # Create a simple model containing the decoder.
    decoder_input = tf.keras.Input(shape=[4, 6])
    encoder_input = tf.keras.Input(shape=[4, 6])
    output = decoder(decoder_input, encoder_input)
    model = tf.keras.Model(inputs=[decoder_input, encoder_input],
        outputs=output)

    # Call decoder on the inputs.
    decoder_input_data = tf.random.uniform(shape=[1, 10, 64])
    encoder_input_data = tf.random.uniform(shape=[1, 10, 64])
    decoder_output = model([decoder_input_data, encoder_input_data])

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
        **kwargs,
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
        self._self_attention_layer = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self._attention_head_size,
            value_dim=self._attention_head_size,
            dropout=self.dropout,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        self._encoder_decoder_attention_layer = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self._attention_head_size,
            value_dim=feature_size,
            dropout=self.dropout,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )

        self._decoder_attention_layernorm = keras.layers.LayerNormalization()
        self._enc_dec_attention_layernorm = keras.layers.LayerNormalization()
        self._feedforward_layernorm = keras.layers.LayerNormalization()

        self._self_attention_dropout = keras.layers.Dropout(rate=self.dropout)
        self._enc_dec_attentiondropout = keras.layers.Dropout(
            rate=self.dropout,
        )
        # First dense layer in the feedforward network, which maps input
        # feauture size to dimension `self.intermediate_dim`.
        self._intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        # Second dense layer in the feedforward network, which maps input
        # feature size back to the input feature size.
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

    def call(
        self,
        decoder_sequence,
        encoder_sequence,
        decoder_padding_mask=None,
        decoder_attention_mask=None,
        encoder_padding_mask=None,
        encoder_attention_mask=None,
        use_causal_mask=False,
    ):
        """Forward pass of the TransformerDecoder.

        Args:
            decoder_sequence: a Tensor. The decoder input sequence.
            encoder_sequence: a Tensor. The decoder input sequence.
            decoder_padding_mask: a boolean Tensor, the padding mask of decoder
                sequence, must of shape [batch_size, decoder_sequence_length].
            decoder_attention_mask: a boolean Tensor. Customized decoder
                sequence mask, must of shape
                [batch_size, decoder_sequence_length, decoder_sequence_length].
            encoder_padding_mask: a boolean Tensor, the padding mask of encoder
                sequence, must of shape [batch_size, encoder_sequence_length].
            encoder_attention_mask: a boolean Tensor. Customized encoder
                sequence mask, must of shape
                [batch_size, encoder_sequence_length, encoder_sequence_length].
            use_causal_mask: bool, defaults to False. If true, causal mask
                (masking out future input) is applied on the decoder sequence.

        Returns:
            A Tensor of the same shape as the `decoder_sequence`.
        """
        if not self._built:
            self._build(decoder_sequence.shape)
        encoder_mask = merge_padding_and_attention_mask(
            encoder_sequence, encoder_padding_mask, encoder_attention_mask
        )
        decoder_mask = merge_padding_and_attention_mask(
            decoder_sequence, decoder_padding_mask, decoder_attention_mask
        )
        if use_causal_mask:
            causal_mask = tf.cast(
                compute_causal_mask(decoder_sequence),
                dtype=tf.int32,
            )
            if decoder_mask is None:
                decoder_mask = causal_mask
            else:
                decoder_mask = tf.minimum(decoder_mask, causal_mask)

        # Decoder input self-attention.
        self_attended = self._self_attention_layer(
            decoder_sequence,
            decoder_sequence,
            decoder_sequence,
            attention_mask=decoder_mask,
        )
        self_attended = self._self_attention_dropout(self_attended)
        self_attended = self._add_and_norm(
            self_attended, decoder_sequence, self._decoder_attention_layernorm
        )
        # Encoder-decoder attention.
        encoder_decoder_attended = self._encoder_decoder_attention_layer(
            query=self_attended,
            value=encoder_sequence,
            key=encoder_sequence,
            attention_mask=encoder_mask,
        )
        encoder_decoder_attended = self._enc_dec_attentiondropout(
            encoder_decoder_attended,
        )
        encoder_decoder_attended = self._add_and_norm(
            encoder_decoder_attended,
            self_attended,
            self._enc_dec_attention_layernorm,
        )
        # Feedforward.
        feed_forward_output = self._feed_forward(encoder_decoder_attended)
        return self._add_and_norm(
            encoder_decoder_attended,
            feed_forward_output,
            self._feedforward_layernorm,
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
