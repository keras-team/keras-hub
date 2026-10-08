# import keras
# from keras import ops
import keras
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.bloom.bloom_attention import BloomAttention
from keras_hub.src.utils.keras_utils import clone_initializer


class BloomDecoder(keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        intermediate_dim,
        dropout=0.0,
        layer_norm_epsilon=1e-5,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

    def build(self, decoder_sequence_shape):
        hidden_dim = decoder_sequence_shape[-1]
        head_dim = int(hidden_dim // self.num_heads)

        if head_dim * self.num_heads != hidden_dim:
            raise ValueError(
                f"`hidden_dim` must be divisible by num_heads (got `hidden_dim`"
                f": {hidden_dim} and `num_heads`: {self.num_heads})."
            )

        self._pre_attention_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="pre_attention_layernorm",
        )
        self._pre_attention_layernorm.build(decoder_sequence_shape)

        self._self_attention_layer = BloomAttention(
            num_heads=self.num_heads,
            dropout=self.dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="self_attention",
        )
        self._self_attention_layer.build(decoder_sequence_shape)

        self._post_attention_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="post_attention_layernorm",
        )
        self._post_attention_layernorm.build(decoder_sequence_shape)

        self._mlp_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="mlp_intermediate_dense",
        )
        self._mlp_intermediate_dense.build(decoder_sequence_shape)

        self._mlp_output_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="mlp_output_dense",
        )
        intermediate_shape = list(decoder_sequence_shape)
        intermediate_shape[-1] = self.intermediate_dim
        self._mlp_output_dense.build(tuple(intermediate_shape))

        self._dropout_layer = keras.layers.Dropout(
            rate=self.dropout, dtype=self.dtype_policy, name="dropout"
        )

        self.built = True

    def call(
        self,
        decoder_sequence,
        decoder_padding_mask=None,
        decoder_attention_mask=None,
        cache=None,
        cache_update_index=None,
        use_causal_mask=True,
    ):
        self_attention_mask = self._compute_attention_mask(
            decoder_sequence=decoder_sequence,
            decoder_padding_mask=decoder_padding_mask,
            decoder_attention_mask=decoder_attention_mask,
            use_causal_mask=use_causal_mask,
            cache=cache,
            cache_update_index=cache_update_index,
        )

        residual = decoder_sequence
        x = self._pre_attention_layernorm(decoder_sequence)

        attention_output = self._self_attention_layer(
            hidden_states=x,
            attention_mask=self_attention_mask,
            cache=cache,
            cache_update_index=cache_update_index,
        )

        if cache is None:
            x = attention_output
        else:
            x, cache = attention_output

        x = x + residual
        residual = x
        x = self._post_attention_layernorm(x)
        x = self._mlp_intermediate_dense(x)
        x = keras.activations.gelu(x, approximate=True)
        x = self._mlp_output_dense(x)
        x = self._dropout_layer(x)
        x = x + residual

        if cache is not None:
            return x, cache
        else:
            return x

    def _compute_attention_mask(
        self,
        decoder_sequence,
        decoder_padding_mask,
        decoder_attention_mask,
        use_causal_mask,
        cache,
        cache_update_index,
    ):
        decoder_mask = merge_padding_and_attention_mask(
            decoder_sequence, decoder_padding_mask, decoder_attention_mask
        )
        if use_causal_mask:
            batch_size = ops.shape(decoder_sequence)[0]
            input_length = output_length = ops.shape(decoder_sequence)[1]
            if cache is not None:
                input_length = ops.shape(cache)[2]

            causal_mask = compute_causal_mask(
                batch_size,
                input_length,
                output_length,
                (0 if cache_update_index is None else cache_update_index),
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
                "num_heads": self.num_heads,
                "intermediate_dim": self.intermediate_dim,
                "dropout": self.dropout,
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

    def compute_output_shape(self, decoder_sequence_shape):
        return decoder_sequence_shape
