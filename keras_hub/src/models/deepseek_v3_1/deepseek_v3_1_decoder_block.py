"""Transformer decoder block for DeepSeek V3.1."""

import keras
from keras import ops

from keras_hub.src.models.deepseek_v3_1.deepseek_v3_1_attention import (
    DeepSeekV3_1Attention,
)
from keras_hub.src.models.deepseek_v3_1.deepseek_v3_1_moe import (
    DeepSeekV3_1MoE,
)


class DeepSeekV3_1RMSNorm(keras.layers.Layer):
    """RMS Normalization layer.

    Computes normalization in float32 for numerical stability, then casts
    back to compute dtype. This matches the paper's implementation and
    avoids precision issues with fp16/bf16.
    """

    def __init__(self, epsilon=1e-6, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=input_shape[-1:],
            initializer="ones",
        )

    def call(self, x):
        x_f32 = ops.cast(x, "float32")
        mean_square = ops.mean(ops.square(x_f32), axis=-1, keepdims=True)
        normalized = ops.cast(
            x_f32 * ops.rsqrt(mean_square + self.epsilon), self.compute_dtype
        )
        return normalized * ops.cast(self.scale, self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


class DeepSeekV3_1DenseFeedForward(keras.layers.Layer):
    """Dense feed-forward network with SwiGLU architecture.

    Used for the first `first_k_dense_replace` layers before MoE layers begin.
    SwiGLU: output = down_proj(silu(gate_proj(x)) * up_proj(x))
    """

    def __init__(
        self,
        intermediate_dim,
        hidden_dim,
        kernel_initializer="glorot_uniform",
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.intermediate_dim = intermediate_dim
        self.hidden_dim = hidden_dim

        self.gate_proj = keras.layers.Dense(
            intermediate_dim,
            activation="silu",
            kernel_initializer=kernel_initializer,
            dtype=dtype,
            name="gate_proj",
        )
        self.up_proj = keras.layers.Dense(
            intermediate_dim,
            kernel_initializer=kernel_initializer,
            dtype=dtype,
            name="up_proj",
        )
        self.down_proj = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=kernel_initializer,
            dtype=dtype,
            name="down_proj",
        )

    def call(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "hidden_dim": self.hidden_dim,
            }
        )
        return config


class DeepSeekV3_1DecoderBlock(keras.layers.Layer):
    """Transformer decoder block with MLA attention and MoE/Dense FFN.

    Structure (per paper Figure 2):
        x -> RMSNorm -> MLA Attention -> residual
          -> RMSNorm -> FFN (MoE or Dense) -> residual

    Cache format: tuple of (c_kv, k_rope) tensors, both of shape
    (batch, max_seq_len, dim), matching MLA's compressed KV representation.
    """

    def __init__(
        self,
        hidden_dim,
        num_query_heads,
        num_key_value_heads,
        intermediate_dim,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        num_routed_experts=256,
        num_shared_experts=1,
        num_experts_per_tok=8,
        use_moe=True,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        layer_norm_epsilon=1e-6,
        dropout=0.0,
        kernel_initializer="glorot_uniform",
        dtype=None,
        **kwargs,
    ):
        super().__init__(name=kwargs.pop("name", None), dtype=dtype)

        self.hidden_dim = hidden_dim
        self.use_moe = use_moe
        self.num_shared_experts = num_shared_experts

        self.pre_attention_norm = DeepSeekV3_1RMSNorm(
            epsilon=layer_norm_epsilon,
            name=(
                f"{self.name}_pre_attention_norm"
                if getattr(self, "name", None)
                else "pre_attention_norm"
            ),
            dtype=dtype,
        )

        self.attention = DeepSeekV3_1Attention(
            hidden_dim=hidden_dim,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            rope_max_wavelength=rope_max_wavelength,
            rope_scaling_factor=rope_scaling_factor,
            attention_dropout=dropout,
            kernel_initializer=kernel_initializer,
            dtype=dtype,
            name=(
                f"{self.name}_attention" if getattr(self, "name", None) else "attention"
            ),
        )

        self.pre_ffn_norm = DeepSeekV3_1RMSNorm(
            epsilon=layer_norm_epsilon,
            dtype=dtype,
            name=(
                f"{self.name}_pre_ffn_norm"
                if getattr(self, "name", None)
                else "pre_ffn_norm"
            ),
        )

        if use_moe:
            self.ffn = DeepSeekV3_1MoE(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_routed_experts=num_routed_experts,
                num_shared_experts=num_shared_experts,
                num_experts_per_tok=num_experts_per_tok,
                kernel_initializer=kernel_initializer,
                dtype=dtype,
                name=(
                    f"{self.name}_moe_ffn" if getattr(self, "name", None) else "moe_ffn"
                ),
            )
        else:
            self.ffn = DeepSeekV3_1DenseFeedForward(
                intermediate_dim=intermediate_dim,
                hidden_dim=hidden_dim,
                kernel_initializer=kernel_initializer,
                dtype=dtype,
                name=(
                    f"{self.name}_dense_ffn"
                    if getattr(self, "name", None)
                    else "dense_ffn"
                ),
            )

        self.dropout = keras.layers.Dropout(
            dropout,
            dtype=dtype,
            name=f"{self.name}_dropout" if getattr(self, "name", None) else None,
        )

    def call(
        self,
        hidden_states,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
        training=False,
    ):
        # Pre-attention norm + MLA
        normed = self.pre_attention_norm(hidden_states)
        attn_output = self.attention(
            normed,
            attention_mask=attention_mask,
            cache=cache,
            cache_update_index=cache_update_index,
            training=training,
        )

        if isinstance(attn_output, tuple):
            attn_output, new_cache = attn_output
        else:
            new_cache = None

        attn_output = self.dropout(attn_output, training=training)
        hidden_states = hidden_states + attn_output

        # Pre-FFN norm + FFN (MoE or Dense)
        normed = self.pre_ffn_norm(hidden_states)
        ffn_output = self.ffn(normed, training=training)
        ffn_output = self.dropout(ffn_output, training=training)
        hidden_states = hidden_states + ffn_output

        if new_cache is not None:
            return hidden_states, new_cache

        return hidden_states

    def compute_output_spec(
        self,
        hidden_states,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
        training=None,
    ):
        """Return symbolic output spec for Keras graph tracing.

        FIX: When cache is provided, the layer returns a tuple
        (hidden_states, new_cache). Previously this always returned a
        single tensor, which broke symbolic tracing for cached generation.
        """
        try:
            input_shape = hidden_states.shape
        except Exception:
            input_shape = None

        hidden_out = keras.KerasTensor(input_shape, dtype=self.compute_dtype)

        if cache is not None:
            # Cache is a tuple of (c_kv, k_rope) KerasTensors
            # Return matching spec so Keras can trace through cached paths
            c_kv, k_rope = cache
            try:
                c_kv_shape = c_kv.shape
                k_rope_shape = k_rope.shape
            except Exception:
                c_kv_shape = None
                k_rope_shape = None
            new_cache_spec = (
                keras.KerasTensor(c_kv_shape, dtype=self.compute_dtype),
                keras.KerasTensor(k_rope_shape, dtype=self.compute_dtype),
            )
            return hidden_out, new_cache_spec

        return hidden_out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "use_moe": self.use_moe,
            }
        )
        return config
