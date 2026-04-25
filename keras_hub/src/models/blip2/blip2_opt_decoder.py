"""Cached OPT decoder layers for BLIP-2.

Provides ``CachedOPTAttention`` and ``OPTDecoderBlock`` — the attention and
transformer block used by ``Blip2CustomOPT``. Both support an optional
KV-cache path for autoregressive generation.
"""

import keras
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)


@keras.saving.register_keras_serializable(package="keras_hub")
class CachedOPTAttention(keras.layers.Layer):
    """Multi-head self-attention with an optional KV cache."""

    def __init__(self, num_heads, hidden_dim, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.dropout_rate = dropout

        self.q_proj = keras.layers.EinsumDense(
            equation="btd,df->btf",
            output_shape=(None, hidden_dim),
            bias_axes="f",
            dtype=self.dtype_policy,
            name="q_proj",
        )
        self.k_proj = keras.layers.EinsumDense(
            equation="btd,df->btf",
            output_shape=(None, hidden_dim),
            bias_axes="f",
            dtype=self.dtype_policy,
            name="k_proj",
        )
        self.v_proj = keras.layers.EinsumDense(
            equation="btd,df->btf",
            output_shape=(None, hidden_dim),
            bias_axes="f",
            dtype=self.dtype_policy,
            name="v_proj",
        )
        self.out_proj = keras.layers.EinsumDense(
            equation="btd,df->btf",
            output_shape=(None, hidden_dim),
            bias_axes="f",
            dtype=self.dtype_policy,
            name="out_proj",
        )
        if dropout > 0:
            self.attn_dropout = keras.layers.Dropout(
                dropout, dtype=self.dtype_policy, name="attn_dropout"
            )

    def _split_heads(self, x):
        """``(B, T, D)`` -> ``(B, H, T, head_dim)``."""
        b, t = ops.shape(x)[0], ops.shape(x)[1]
        return ops.transpose(
            ops.reshape(x, (b, t, self.num_heads, self.head_dim)),
            (0, 2, 1, 3),
        )

    def _merge_heads(self, x):
        """``(B, H, T, head_dim)`` -> ``(B, T, D)``."""
        b, t = ops.shape(x)[0], ops.shape(x)[2]
        return ops.reshape(
            ops.transpose(x, (0, 2, 1, 3)),
            (b, t, self.hidden_dim),
        )

    # ── FIX: Keras cannot infer output shape when call() has a conditional
    #         return (tensor vs tuple). compute_output_spec() short-circuits
    #         symbolic tracing so variables are never double-initialized.
    def compute_output_spec(
        self,
        x,
        attention_mask=None,
        cache=None,
        cache_update_index=0,
        training=None,
    ):
        # Output always mirrors input shape (B, T, hidden_dim)
        output = keras.KerasTensor(x.shape, dtype=x.dtype)

        # Use explicit None check — KerasTensor truthiness is unreliable
        if cache is not None:
            new_cache = keras.KerasTensor(cache.shape, dtype=cache.dtype)
            return output, new_cache

        return output

    def call(
        self,
        x,
        attention_mask=None,
        cache=None,
        cache_update_index=0,
        training=None,
    ):
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if cache is not None:
            k_for_cache = ops.transpose(k, (0, 2, 1, 3))
            v_for_cache = ops.transpose(v, (0, 2, 1, 3))
            k_cache = ops.slice_update(
                cache[:, 0], [0, cache_update_index, 0, 0], k_for_cache
            )
            v_cache = ops.slice_update(
                cache[:, 1], [0, cache_update_index, 0, 0], v_for_cache
            )
            new_cache = ops.stack([k_cache, v_cache], axis=1)
            k = ops.transpose(k_cache, (0, 2, 1, 3))
            v = ops.transpose(v_cache, (0, 2, 1, 3))
        else:
            new_cache = None

        scale = ops.cast(self.head_dim, q.dtype) ** -0.5
        attn_weights = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * scale

        if attention_mask is not None:
            attn_weights = attn_weights + ops.cast(
                attention_mask, attn_weights.dtype
            )

        attn_weights = ops.softmax(attn_weights, axis=-1)
        if self.dropout_rate > 0:
            attn_weights = self.attn_dropout(attn_weights, training=training)

        out = self.out_proj(self._merge_heads(ops.matmul(attn_weights, v)))

        if cache is not None:
            return out, new_cache
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout_rate,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class OPTDecoderBlock(keras.layers.Layer):
    """OPT pre-norm transformer decoder block with optional KV-cache."""

    def __init__(
        self,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout=0.1,
        layer_norm_epsilon=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.dropout_rate = dropout
        self.layer_norm_epsilon = layer_norm_epsilon

        self.self_attn_layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attn_layer_norm",
        )
        self.self_attn = CachedOPTAttention(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            dropout=dropout,
            dtype=self.dtype_policy,
            name="self_attn",
        )
        self.final_layer_norm = keras.layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="final_layer_norm",
        )
        self.fc1 = keras.layers.EinsumDense(
            equation="btd,df->btf",
            output_shape=(None, intermediate_dim),
            bias_axes="f",
            dtype=self.dtype_policy,
            name="fc1",
        )
        self.fc2 = keras.layers.EinsumDense(
            equation="btf,fd->btd",
            output_shape=(None, hidden_dim),
            bias_axes="d",
            dtype=self.dtype_policy,
            name="fc2",
        )
        if dropout > 0:
            self.residual_dropout = keras.layers.Dropout(
                dropout, dtype=self.dtype_policy, name="residual_dropout"
            )

    def _compute_attention_mask(
        self, x, padding_mask, cache, cache_update_index
    ):
        batch_size = ops.shape(x)[0]
        output_length = ops.shape(x)[1]
        input_length = (
            ops.shape(cache)[2] if cache is not None else output_length
        )

        causal_mask = compute_causal_mask(
            batch_size=batch_size,
            input_length=input_length,
            output_length=output_length,
            cache_index=cache_update_index,
        )

        decoder_mask = merge_padding_and_attention_mask(
            inputs=x, padding_mask=padding_mask, attention_mask=None
        )

        if decoder_mask is not None:
            bool_mask = ops.logical_and(
                causal_mask, ops.cast(decoder_mask, "bool")
            )
        else:
            bool_mask = causal_mask

        bool_mask = ops.expand_dims(bool_mask, axis=1)
        return ops.cast(ops.logical_not(bool_mask), x.dtype) * -1e9

    # ── FIX: Same conditional-return problem as CachedOPTAttention.
    #         Declaring compute_output_spec() prevents Keras from re-tracing
    #         call() symbolically after variables are already initialized,
    #         which is what caused "Variable ... is already initialized".
    def compute_output_spec(
        self,
        x,
        padding_mask=None,
        cache=None,
        cache_update_index=0,
        training=None,
    ):
        # Output shape always matches input shape (B, T, hidden_dim)
        output = keras.KerasTensor(x.shape, dtype=x.dtype)

        # Explicit None check — never rely on KerasTensor truthiness
        if cache is not None:
            new_cache = keras.KerasTensor(cache.shape, dtype=cache.dtype)
            return output, new_cache

        return output

    def call(
        self,
        x,
        padding_mask=None,
        cache=None,
        cache_update_index=0,
        training=None,
    ):
        attention_mask = self._compute_attention_mask(
            x, padding_mask, cache, cache_update_index
        )

        residual = x
        x = self.self_attn_layer_norm(x)
        if cache is not None:
            x, new_cache = self.self_attn(
                x,
                attention_mask=attention_mask,
                cache=cache,
                cache_update_index=cache_update_index,
                training=training,
            )
        else:
            x = self.self_attn(
                x, attention_mask=attention_mask, training=training
            )
            new_cache = None
        if self.dropout_rate > 0:
            x = self.residual_dropout(x, training=training)
        x = residual + ops.cast(x, residual.dtype)

        residual = x
        x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = keras.activations.relu(x)
        x = self.fc2(x)
        if self.dropout_rate > 0:
            x = self.residual_dropout(x, training=training)
        x = residual + ops.cast(x, residual.dtype)

        if cache is not None:
            return x, new_cache
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "dropout": self.dropout_rate,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config
