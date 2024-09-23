# Copyright 2024 The KerasHub Authors
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

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.cached_multi_head_attention import (
    CachedMultiHeadAttention,
)
from keras_hub.src.utils.keras_utils import clone_initializer

from keras_hub.src.layers.modeling.transformer_layer_utils import (  # isort:skip
    compute_causal_mask,
    merge_padding_and_attention_mask,
)


@keras_hub_export("keras_hub.layers.TransformerDecoder")
class TransformerDecoder(keras.layers.Layer):
    """Transformer decoder.

    This class follows the architecture of the transformer decoder layer in the
    paper [Attention is All You Need](https://arxiv.org/abs/1706.03762). Users
    can instantiate multiple instances of this class to stack up a decoder.

    By default, this layer will apply a causal mask to the decoder attention
    layer. You can also pass padding or attention masks directly to the layer
    during call, e.g. with `decoder_padding_mask` or `decoder_attention_mask`.

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
        dropout: float. the dropout value, shared by
            MultiHeadAttention and feedforward network. Defaults to `0.`.
        activation: string or `keras.activations`. the
            activation function of feedforward network.
            Defaults to `"relu"`.
        layer_norm_epsilon: float. The eps value in layer
            normalization components. Defaults to `1e-5`.
        kernel_initializer: string or `keras.initializers` initializer.
            The kernel initializer for the dense and multiheaded
            attention layers. Defaults to `"glorot_uniform"`.
        bias_initializer: string or `keras.initializers` initializer.
            The bias initializer for the dense and multiheaded
            attention layers. Defaults to `"zeros"`.
        normalize_first: bool. If True, the inputs to the
            attention layer(s) and the intermediate dense layer are normalized
            (similar to GPT-2). If set to False, outputs of attention layer and
            intermediate dense layer are normalized (similar to BERT).
            Defaults to `False`.
        **kwargs: other keyword arguments passed to `keras.layers.Layer`,
            including `name`, `trainable`, `dtype` etc.

    Example:
    ```python
    # Create a single transformer decoder layer.
    decoder = keras_hub.layers.TransformerDecoder(
        intermediate_dim=64, num_heads=8)

    # Create a simple model containing the decoder.
    decoder_input = keras.Input(shape=(10, 64))
    encoder_input = keras.Input(shape=(10, 64))
    output = decoder(decoder_input, encoder_input)
    model = keras.Model(
        inputs=(decoder_input, encoder_input),
        outputs=output,
    )

    # Call decoder on the inputs.
    decoder_input_data = np.random.uniform(size=(2, 10, 64))
    encoder_input_data = np.random.uniform(size=(2, 10, 64))
    decoder_output = model((decoder_input_data, encoder_input_data))
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
        **kwargs,
    ):
        # Work around for model saving, we need to ensure our model is built
        # immediately after restoring from config.
        decoder_sequence_shape = kwargs.pop("decoder_sequence_shape", None)
        encoder_sequence_shape = kwargs.pop("encoder_sequence_shape", None)

        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.normalize_first = normalize_first
        self.supports_masking = True
        self._decoder_sequence_shape = None
        self._encoder_sequence_shape = None

        if decoder_sequence_shape:
            self.build(decoder_sequence_shape, encoder_sequence_shape)

    def build(
        self,
        decoder_sequence_shape,
        encoder_sequence_shape=None,
    ):
        self._decoder_sequence_shape = decoder_sequence_shape
        self._encoder_sequence_shape = encoder_sequence_shape
        # Infer the dimension of our hidden feature size from the build shape.
        hidden_dim = decoder_sequence_shape[-1]
        # Attention head size is `hidden_dim` over the number of heads.
        head_dim = int(hidden_dim // self.num_heads)
        if head_dim == 0:
            raise ValueError(
                "Attention `head_dim` computed cannot be zero. "
                f"The `hidden_dim` value of {hidden_dim} has to be equal to "
                f"or greater than `num_heads` value of {self.num_heads}."
            )

        # Self attention layers.
        self._self_attention_layer = CachedMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=head_dim,
            dropout=self.dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="self_attention",
        )
        if hasattr(self._self_attention_layer, "_build_from_signature"):
            self._self_attention_layer._build_from_signature(
                query=decoder_sequence_shape,
                value=decoder_sequence_shape,
            )
        else:
            self._self_attention_layer.build(
                query_shape=decoder_sequence_shape,
                value_shape=decoder_sequence_shape,
            )
        self._self_attention_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attention_layer_norm",
        )
        self._self_attention_layer_norm.build(decoder_sequence_shape)
        self._self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention_dropout",
        )

        # Cross attention layers are optional.
        self._cross_attention_layer = None
        if encoder_sequence_shape:
            self._cross_attention_layer = CachedMultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=head_dim,
                value_dim=head_dim,
                dropout=self.dropout,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                dtype=self.dtype_policy,
                name="cross_attention",
            )
            if hasattr(self._cross_attention_layer, "_build_from_signature"):
                self._cross_attention_layer._build_from_signature(
                    query=decoder_sequence_shape,
                    value=encoder_sequence_shape,
                )
            else:
                self._cross_attention_layer.build(
                    query_shape=decoder_sequence_shape,
                    value_shape=encoder_sequence_shape,
                )
            self._cross_attention_layer_norm = keras.layers.LayerNormalization(
                epsilon=self.layer_norm_epsilon,
                dtype=self.dtype_policy,
                name="cross_attention_layer_norm",
            )
            self._cross_attention_layer_norm.build(decoder_sequence_shape)
            self._cross_attention_dropout = keras.layers.Dropout(
                rate=self.dropout,
                dtype=self.dtype_policy,
                name="cross_attention_dropout",
            )

        # Feedforward layers.
        self._feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self._feedforward_intermediate_dense.build(decoder_sequence_shape)
        self._feedforward_output_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="feedforward_output_dense",
        )
        intermediate_shape = list(decoder_sequence_shape)
        intermediate_shape[-1] = self.intermediate_dim
        self._feedforward_output_dense.build(tuple(intermediate_shape))
        self._feedforward_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="feedforward_layer_norm",
        )
        self._feedforward_layer_norm.build(decoder_sequence_shape)
        self._feedforward_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="feedforward_dropout",
        )
        # Create layers based on input shape.
        self.built = True

    def call(
        self,
        decoder_sequence,
        encoder_sequence=None,
        decoder_padding_mask=None,
        decoder_attention_mask=None,
        encoder_padding_mask=None,
        encoder_attention_mask=None,
        self_attention_cache=None,
        self_attention_cache_update_index=None,
        cross_attention_cache=None,
        cross_attention_cache_update_index=None,
        use_causal_mask=True,
        training=None,
    ):
        """Forward pass of the TransformerDecoder.

        Args:
            decoder_sequence: a Tensor. The decoder input sequence.
            encoder_sequence: a Tensor. The encoder input sequence. For decoder
                only models (like GPT2), this should be left `None`. Once the
                model is called once without an encoder_sequence, you cannot
                call it again with encoder_sequence.
            decoder_padding_mask: a boolean Tensor, the padding mask of decoder
                sequence, must be of shape
                `[batch_size, decoder_sequence_length]`.
            decoder_attention_mask: a boolean Tensor. Customized decoder
                sequence mask, must be of shape
                `[batch_size, decoder_sequence_length, decoder_sequence_length]`.
            encoder_padding_mask: a boolean Tensor, the padding mask of encoder
                sequence, must be of shape
                `[batch_size, encoder_sequence_length]`.
            encoder_attention_mask: a boolean Tensor. Customized encoder
                sequence mask, must be of shape
                `[batch_size, encoder_sequence_length, encoder_sequence_length]`.
            self_attention_cache: a dense float Tensor. The cache of key/values
                pairs in the self-attention layer. Has shape
                `[batch_size, 2, max_seq_len, num_heads, key_dims]`.
            self_attention_cache_update_index: an int or int Tensor, the index
                at which to update the `self_attention_cache`. Usually, this is
                the index of the current token being processed during decoding.
            cross_attention_cache: a dense float Tensor. The cache of
                key/value pairs in the cross-attention layer. Has shape
                `[batch_size, 2, S, num_heads, key_dims]`.
            cross_attention_cache_update_index:  an int or int Tensor, the index
                at which to update the `cross_attention_cache`. Usually, this is
                either `0` (compute the entire `cross_attention_cache`), or
                `None` (reuse a previously computed `cross_attention_cache`).
            use_causal_mask: bool, defaults to `True`. If true, a causal mask
                (masking out future input) is applied `on the decoder sequence.
            training: a boolean indicating whether the layer should behave in
                training mode or in inference mode.

        Returns:
            One of three things, depending on call arguments:
            - `outputs`, if `self_attention_cache` is `None.
            - `(outputs, self_attention_cache)`, if `self_attention_cache` is
              set and the layer has no cross-attention.
            - `(outputs, self_attention_cache, cross_attention_cache)`, if
              `self_attention_cache` and `cross_attention_cache` are set and
              the layer has cross-attention.
        """

        has_encoder_sequence = encoder_sequence is not None

        has_cross_attention = self._cross_attention_layer is not None
        if not has_cross_attention and has_encoder_sequence:
            raise ValueError(
                "The number of call arguments to "
                "`keras_hub.layers.TransformerDecoder` should not change. "
                "Use `layer(decoder_sequence, encoder_sequence)` to "
                "build a layer with cross attention, or "
                "`layer(decoder_sequence)` to build a layer without. "
                "This layer has been built without cross attention, but "
                "you are trying to call it with encoder_sequence."
            )
        elif has_cross_attention and not has_encoder_sequence:
            raise ValueError(
                "The number of call arguments to "
                "`keras_hub.layers.TransformerDecoder` should not change. "
                "Use `layer(decoder_sequence, encoder_sequence)` to "
                "build a layer with cross attention, or "
                "`layer(decoder_sequence)` to build a layer without. "
                "This layer has been built with cross attention, but "
                "you did not provide encoder_sequence."
            )

        has_self_attention_cache = self_attention_cache is not None
        has_cross_attention_cache = cross_attention_cache is not None
        if has_cross_attention and (
            has_self_attention_cache != has_cross_attention_cache
        ):
            raise ValueError(
                "When calling `keras_hub.layers.TransformerDecoder` with "
                "cross-attention (with both `encoder_sequence` and "
                "`decoder_sequence`), `self_attention_cache` and "
                "`cross_attention_cache` should both be set or both be `None`. "
                "One cannot be `None` while the other is not. Received: "
                f"self_attention_cache={self_attention_cache}, "
                f"cross_attention_cache={cross_attention_cache}."
            )

        self_attention_mask = self._compute_self_attention_mask(
            decoder_sequence=decoder_sequence,
            decoder_padding_mask=decoder_padding_mask,
            decoder_attention_mask=decoder_attention_mask,
            use_causal_mask=use_causal_mask,
            self_attention_cache=self_attention_cache,
            self_attention_cache_update_index=self_attention_cache_update_index,
        )

        x = decoder_sequence  # Intermediate result.

        # Self attention block.
        residual = x
        if self.normalize_first:
            x = self._self_attention_layer_norm(x)
        attention_output = self._self_attention_layer(
            query=x,
            value=x,
            attention_mask=self_attention_mask,
            cache=self_attention_cache,
            cache_update_index=self_attention_cache_update_index,
            training=training,
        )
        if self_attention_cache is None:
            x = attention_output
        else:
            x, self_attention_cache = attention_output
        x = self._self_attention_dropout(x, training=training)
        x = x + residual
        if not self.normalize_first:
            x = self._self_attention_layer_norm(x)

        # Cross attention is optional.
        if has_cross_attention:
            # Compute cross attention mask.
            cross_attention_mask = merge_padding_and_attention_mask(
                encoder_sequence, encoder_padding_mask, encoder_attention_mask
            )

            # Cross attention block.
            residual = x
            if self.normalize_first:
                x = self._cross_attention_layer_norm(x)
            attention_output = self._cross_attention_layer(
                query=x,
                value=encoder_sequence,
                attention_mask=cross_attention_mask,
                cache=cross_attention_cache,
                cache_update_index=cross_attention_cache_update_index,
                training=training,
            )
            if cross_attention_cache is None:
                x = attention_output
            else:
                x, cross_attention_cache = attention_output
            x = self._cross_attention_dropout(x, training=training)
            x = x + residual
            if not self.normalize_first:
                x = self._cross_attention_layer_norm(x)

        # Feedforward block.
        residual = x
        if self.normalize_first:
            x = self._feedforward_layer_norm(x)
        x = self._feedforward_intermediate_dense(x)
        x = self._feedforward_output_dense(x)
        x = self._feedforward_dropout(x, training=training)
        x = x + residual
        if not self.normalize_first:
            x = self._feedforward_layer_norm(x)

        if self_attention_cache is not None:
            if has_cross_attention:
                return (x, self_attention_cache, cross_attention_cache)
            else:
                return (x, self_attention_cache)
        else:
            return x

    def _compute_self_attention_mask(
        self,
        decoder_sequence,
        decoder_padding_mask,
        decoder_attention_mask,
        use_causal_mask,
        self_attention_cache,
        self_attention_cache_update_index,
    ):
        decoder_mask = merge_padding_and_attention_mask(
            decoder_sequence, decoder_padding_mask, decoder_attention_mask
        )
        if use_causal_mask:
            batch_size = ops.shape(decoder_sequence)[0]
            input_length = output_length = ops.shape(decoder_sequence)[1]
            # We need to handle a rectangular causal mask when doing cached
            # decoding. For generative inference, `decoder_sequence` will
            # generally be length 1, and `cache` will be the full generation length.
            if self_attention_cache is not None:
                input_length = ops.shape(self_attention_cache)[2]

            causal_mask = compute_causal_mask(
                batch_size,
                input_length,
                output_length,
                (
                    0
                    if self_attention_cache_update_index is None
                    else self_attention_cache_update_index
                ),
            )
            return (
                ops.minimum(decoder_mask, causal_mask)
                if decoder_mask is not None
                else causal_mask
            )
        return decoder_mask

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
                "decoder_sequence_shape": self._decoder_sequence_shape,
                "encoder_sequence_shape": self._encoder_sequence_shape,
            }
        )
        return config

    def compute_output_shape(self, decoder_sequence_shape):
        return decoder_sequence_shape
