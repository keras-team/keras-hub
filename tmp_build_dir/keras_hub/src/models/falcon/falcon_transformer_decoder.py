import math

import keras
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.falcon.falcon_attention import FalconAttention


class FalconTransformerDecoder(keras.layers.Layer):
    def __init__(
        self,
        num_attention_heads,
        intermediate_dim,
        layer_norm_epsilon=1e-5,
        attention_dropout_rate=0,
        feedforward_dropout_rate=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.intermediate_dim = intermediate_dim
        self.layer_norm_epsilon = layer_norm_epsilon
        self.attention_dropout_rate = attention_dropout_rate
        self.feedforward_dropout_rate = feedforward_dropout_rate

    def build(self, decoder_sequence_shape):
        self.hidden_dim = decoder_sequence_shape[-1]
        self.input_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="input_layernorm",
        )
        self.input_layernorm.build(decoder_sequence_shape)

        # Attention layers.
        self.key_dim = self.hidden_dim // self.num_attention_heads
        self.attention_layer = FalconAttention(
            num_heads=self.num_attention_heads,
            attention_dropout_rate=self.attention_dropout_rate,
            dtype=self.dtype_policy,
            name="attention",
        )
        self.attention_layer.build(
            decoder_sequence_shape,
        )

        self.attention_dropout = keras.layers.Dropout(
            rate=self.attention_dropout_rate,
            dtype=self.dtype_policy,
            name="attention_dropout",
        )

        self.post_attention_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="post_attention_layernorm",
        )
        self.post_attention_layernorm.build(decoder_sequence_shape)

        # Feedforward layers.
        # TODO: use_bias should be an argument to the transformer to support
        # other sizes of models, e.g. 7B, that don't use bias.
        self.dense_h_to_4h = keras.layers.Dense(
            self.intermediate_dim,
            activation=keras.activations.gelu,
            use_bias=True,
            dtype=self.dtype_policy,
            name="dense_h_to_4h",
        )
        self.dense_h_to_4h.build(decoder_sequence_shape)

        self.dense_4h_to_h = keras.layers.Dense(
            self.hidden_dim,
            use_bias=True,
            dtype=self.dtype_policy,
            name="dense_4h_to_h",
        )
        self.dense_4h_to_h.build(
            (
                decoder_sequence_shape[0],
                decoder_sequence_shape[1],
                self.intermediate_dim,
            )
        )

        self.feedforward_dropout = keras.layers.Dropout(
            rate=self.feedforward_dropout_rate,
            dtype=self.dtype_policy,
            name="feedforward_dropout",
        )

        self.built = True

    def call(
        self,
        inputs,
        decoder_padding_mask=None,
        decoder_attention_mask=None,
        attention_cache=None,
        attention_cache_update_index=None,
        training=None,
    ):
        attention_mask = self._compute_attention_mask(
            decoder_sequence=inputs,
            decoder_padding_mask=decoder_padding_mask,
            decoder_attention_mask=decoder_attention_mask,
            attention_cache=attention_cache,
            attention_cache_update_index=attention_cache_update_index,
        )

        residual = inputs

        x = self.input_layernorm(inputs)

        mask = decoder_padding_mask
        if mask is None:
            batch_size, seq_length = ops.shape(inputs)[:2]
            mask = ops.ones((batch_size, seq_length), dtype="int32")
        alibi = self._build_alibi_tensor(self.num_attention_heads, mask)

        # Attention block.
        attention_output = self.attention_layer(
            inputs=x,
            alibi=alibi,
            attention_mask=attention_mask,
            cache=attention_cache,
            cache_update_index=attention_cache_update_index,
        )

        if attention_cache is None:
            x = attention_output
        else:
            x, attention_cache = attention_output

        x = self.attention_dropout(x, training=training)

        x = x + residual
        residual = x

        x = self.post_attention_layernorm(x)

        x = self.dense_h_to_4h(x)
        x = self.dense_4h_to_h(x)

        x = self.feedforward_dropout(x, training=training)

        x = x + residual

        if attention_cache is not None:
            return x, attention_cache
        else:
            return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_attention_heads": self.num_attention_heads,
                "intermediate_dim": self.intermediate_dim,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "attention_dropout_rate": self.attention_dropout_rate,
                "feedforward_dropout_rate": self.feedforward_dropout_rate,
            }
        )
        return config

    def compute_output_shape(self, decoder_sequence_shape):
        return decoder_sequence_shape

    def _compute_attention_mask(
        self,
        decoder_sequence,
        decoder_padding_mask,
        decoder_attention_mask,
        attention_cache=None,
        attention_cache_update_index=None,
    ):
        decoder_mask = merge_padding_and_attention_mask(
            decoder_sequence, decoder_padding_mask, decoder_attention_mask
        )
        batch_size = ops.shape(decoder_sequence)[0]
        input_length = output_length = ops.shape(decoder_sequence)[1]
        # We need to handle a rectangular causal mask when doing cached
        # decoding. For generative inference, `decoder_sequence` will
        # generally be length 1, and `cache` will be the full generation length.
        if attention_cache is not None:
            input_length = ops.shape(attention_cache)[2]

        causal_mask = compute_causal_mask(
            batch_size,
            input_length,
            output_length,
            (
                0
                if attention_cache_update_index is None
                else attention_cache_update_index
            ),
        )
        return (
            ops.minimum(decoder_mask, causal_mask)
            if decoder_mask is not None
            else causal_mask
        )

    def _build_alibi_tensor(self, num_heads, mask):
        slopes = ops.convert_to_tensor(
            self._get_slopes(num_heads),
            dtype=self.compute_dtype,
        )  # num_heads
        mask = ops.cast(mask, dtype="int32")
        # TODO: cumsum always outputs int64 in Keras 2 so the casting of cumsum
        # result to int32 can be removed when keras 2 support is removed.
        cumsum_mask = ops.cast(ops.cumsum(mask, axis=-1) - 1, "int32")
        arange_tensor = (cumsum_mask * mask)[:, None, :]
        alibi = slopes[..., None] * ops.cast(arange_tensor, self.compute_dtype)
        alibi = ops.expand_dims(
            alibi, 0
        )  # [None, batch_size, num_heads, seq_length]
        return ops.transpose(alibi, [1, 2, 0, 3])

    def _get_slopes(self, num_heads):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][
                    : num_heads - closest_power_of_2
                ]
            )
