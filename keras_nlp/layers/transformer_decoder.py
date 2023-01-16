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

from keras_nlp.utils.keras_utils import clone_initializer

from keras_nlp.layers.transformer_layer_utils import (  # isort:skip
    compute_causal_mask,
    merge_padding_and_attention_mask,
)


@keras.utils.register_keras_serializable(package="keras_nlp")
class TransformerDecoder(keras.layers.Layer):
    """Transformer decoder.

    This class follows the architecture of the transformer decoder layer in the
    paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). Users
    can instantiate multiple instances of this class to stack up a decoder.

    This layer will always apply a causal mask to the decoder attention layer.
    This layer will correctly compute an attention mask from an implicit
    Keras padding mask (for example, by passing `mask_zero=True` to a
    `keras.layers.Embedding` layer). See the Masking and Padding
    [guide](https://keras.io/guides/understanding_masking_and_padding/)
    for more details.

    This layer can be called with either one or two inputs. The number of inputs
    must be consistent across all calls. The options are as follows:
        `layer(decoder_sequence)`: no cross-attention will be built into the
            decoder block. This is useful when building a "decoder-only"
            transformer such as GPT-2.
        `layer(decoder_sequence, encoder_sequence)`: cross-attention will be
            built into the decoder block. This is useful when building an
            "encoder-decoder" transformer, such as the original transformer
            model described in Attention is All You Need.

    Args:
        intermediate_dim: int, the hidden size of feedforward network.
        num_heads: int, the number of heads in MultiHeadAttention.
        dropout: float, defaults to 0. the dropout value, shared by
            MultiHeadAttention and feedforward network.
        activation: string or `keras.activations`, defaults to "relu". the
            activation function of feedforward network.
        layer_norm_epsilon: float, defaults to 1e-5. The eps value in layer
            normalization components.
        kernel_initializer: string or `keras.initializers` initializer,
            defaults to "glorot_uniform". The kernel initializer for
            the dense and multiheaded attention layers.
        bias_initializer: string or `keras.initializers` initializer,
            defaults to "zeros". The bias initializer for
            the dense and multiheaded attention layers.
        normalize_first: bool. Defaults to False. If True, the inputs to the
            attention layer(s) and the intermediate dense layer are normalized
            (similar to GPT-2). If set to False, outputs of attention layer and
            intermediate dense layer are normalized (similar to BERT).
        name: string, defaults to None. The name of the layer.
        **kwargs: other keyword arguments.

    Examples:
    ```python
    # Create a single transformer decoder layer.
    decoder = keras_nlp.layers.TransformerDecoder(
        intermediate_dim=64, num_heads=8)

    # Create a simple model containing the decoder.
    decoder_input = keras.Input(shape=[10, 64])
    encoder_input = keras.Input(shape=[10, 64])
    output = decoder(decoder_input, encoder_input)
    model = keras.Model(inputs=[decoder_input, encoder_input],
        outputs=output)

    # Call decoder on the inputs.
    decoder_input_data = tf.random.uniform(shape=[2, 10, 64])
    encoder_input_data = tf.random.uniform(shape=[2, 10, 64])
    decoder_output = model([decoder_input_data, encoder_input_data])

    ```

    References:
     - [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)

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
        normalize_first=False,
        name=None,
        **kwargs,
    ):
        # Work around for model saving, we need to ensure our model is built
        # immediately after restoring from config.
        self._input_shape = kwargs.pop("build_input_shape", None)
        self._has_cross_attention = kwargs.pop("has_cross_attention", False)

        super().__init__(name=name, **kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.normalize_first = normalize_first
        self._built = False
        self.supports_masking = True

        if self._input_shape is not None:
            self._build(self._input_shape, self._has_cross_attention)

    def _build(self, input_shape, has_cross_attention):
        # Create layers based on input shape.
        self._built = True
        self._input_shape = input_shape
        self._has_cross_attention = has_cross_attention
        # Infer the dimension of our hidden feature size from the build shape.
        hidden_dim = input_shape[-1]
        # Attention head size is `hidden_dim` over the number of heads.
        head_dim = int(hidden_dim // self.num_heads)

        # Self attention layers.
        self._self_attention_layer = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=head_dim,
            dropout=self.dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._self_attention_layer._build_from_signature(
            query=input_shape,
            value=input_shape,
        )
        self._self_attention_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )
        self._self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
        )

        # Cross attention layers are optional.
        self._cross_attention_layer = None
        if has_cross_attention:
            self._cross_attention_layer = keras.layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=head_dim,
                value_dim=hidden_dim,
                dropout=self.dropout,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
            )
            self._cross_attention_layer._build_from_signature(
                query=input_shape,
                value=input_shape,
            )
            self._cross_attention_layernorm = keras.layers.LayerNormalization(
                epsilon=self.layer_norm_epsilon,
            )
            self._cross_attention_dropout = keras.layers.Dropout(
                rate=self.dropout,
            )

        # Feedforward layers.
        self._feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._feedforward_output_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
        )
        self._feedforward_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )
        self._feedforward_dropout = keras.layers.Dropout(
            rate=self.dropout,
        )

    def call(
        self,
        decoder_sequence,
        encoder_sequence=None,
        decoder_padding_mask=None,
        decoder_attention_mask=None,
        encoder_padding_mask=None,
        encoder_attention_mask=None,
    ):
        """Forward pass of the TransformerDecoder.

        Args:
            decoder_sequence: a Tensor. The decoder input sequence.
            encoder_sequence: a Tensor. The encoder input sequence. For decoder
                only models (like GPT2), this should be left None. Once the
                model is called once without an encoder_sequence, you cannot
                call it again with encoder_sequence.
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
        Returns:
            A Tensor of the same shape as the `decoder_sequence`.
        """

        has_encoder_sequence = encoder_sequence is not None
        if not self._built:
            self._build(decoder_sequence.shape, has_encoder_sequence)

        is_cross_attention = self._cross_attention_layer is not None
        if not is_cross_attention and has_encoder_sequence:
            raise ValueError(
                "The number of call arguments to "
                "`keras_nlp.layers.TransformerDecoder` should not change. "
                "Use `layer(decoder_sequence, encoder_sequence)` to "
                "build a layer with cross attention, or "
                "`layer(decoder_sequence)` to build a layer without. "
                "This layer has been built without cross attention, but "
                "you are trying to call it with encoder_sequence."
            )
        elif is_cross_attention and not has_encoder_sequence:
            raise ValueError(
                "The number of call arguments to "
                "`keras_nlp.layers.TransformerDecoder` should not change. "
                "Use `layer(decoder_sequence, encoder_sequence)` to "
                "build a layer with cross attention, or "
                "`layer(decoder_sequence)` to build a layer without. "
                "This layer has been built with cross attention, but "
                "you did not provide encoder_sequence."
            )

        x = decoder_sequence  # Intermediate result.

        # Compute self attention mask.
        self_attention_mask = compute_causal_mask(decoder_sequence)
        decoder_mask = merge_padding_and_attention_mask(
            decoder_sequence, decoder_padding_mask, decoder_attention_mask
        )
        if decoder_mask is not None:
            self_attention_mask = tf.minimum(decoder_mask, self_attention_mask)

        # Self attention block.
        residual = x
        if self.normalize_first:
            x = self._self_attention_layernorm(x)
        x = self._self_attention_layer(
            query=x,
            value=x,
            attention_mask=self_attention_mask,
        )
        x = self._self_attention_dropout(x)
        x = x + residual
        if not self.normalize_first:
            x = self._self_attention_layernorm(x)

        # Cross attention is optional.
        if self._cross_attention_layer is not None:
            # Compute cross attention mask.
            cross_attention_mask = merge_padding_and_attention_mask(
                encoder_sequence, encoder_padding_mask, encoder_attention_mask
            )

            # Cross attention block.
            residual = x
            if self.normalize_first:
                x = self._cross_attention_layernorm(x)
            x = self._cross_attention_layer(
                query=x,
                value=encoder_sequence,
                attention_mask=cross_attention_mask,
            )
            x = self._cross_attention_dropout(x)
            x = x + residual
            if not self.normalize_first:
                x = self._cross_attention_layernorm(x)

        # Feedforward block.
        residual = x
        if self.normalize_first:
            x = self._feedforward_layernorm(x)
        x = self._feedforward_intermediate_dense(x)
        x = self._feedforward_output_dense(x)
        x = self._feedforward_dropout(x)
        x = x + residual
        if not self.normalize_first:
            x = self._feedforward_layernorm(x)

        return x

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
                "normalize_first": self.normalize_first,
                "build_input_shape": self._input_shape,
                "has_cross_attention": self._has_cross_attention,
            }
        )
        return config
