"""DeepSeek V31 transformer decoder block, RMSNorm, and dense FFN."""

import keras
from keras import ops

from keras_hub.src.models.deepseek_v31.deepseek_v31_attention import (
    DeepSeekV31Attention,
)
from keras_hub.src.models.deepseek_v31.deepseek_v31_moe import DeepSeekV31MoE


class DeepSeekV31RMSNorm(keras.layers.Layer):
    """Root Mean Square Layer Normalization for DeepSeek V31.

    Applies RMS normalization using float32 precision internally to avoid
    numerical instability with fp16/bf16 training, then casts back to the
    layer's compute dtype. This matches the reference DeepSeek implementation.

    Unlike `LayerNormalization`, RMSNorm does not subtract the mean, which
    reduces computation while preserving re-scaling performance.

    Args:
        epsilon: float. Small value added to the RMS denominator for numerical
            stability. Defaults to `1e-6`.

    Example:

    ```python
    norm = keras_hub.layers.DeepSeekV31RMSNorm(epsilon=1e-6)
    x = keras.random.normal((2, 16, 512))
    output = norm(x)  # (2, 16, 512)
    ```

    Reference:
     - [DeepSeek-AI et al., 2024](https://arxiv.org/abs/2412.19437)
     - [Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)
    """

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=input_shape[-1:],
            initializer="ones",
        )
        super().build(input_shape)

    def call(self, x):
        x_fp32 = ops.cast(x, "float32")
        rms = ops.rsqrt(
            ops.mean(ops.square(x_fp32), axis=-1, keepdims=True) + self.epsilon
        )
        return ops.cast(x_fp32 * rms, self.compute_dtype) * ops.cast(
            self.scale, self.compute_dtype
        )

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


class DeepSeekV31DenseFeedForward(keras.layers.Layer):
    """Dense SwiGLU feed-forward network for DeepSeek V31.

    Used for the first `first_k_dense_replace` transformer layers before the
    MoE layers begin. Implements the gated activation function:

        output = down_proj(silu(gate_proj(x)) * up_proj(x))

    Args:
        hidden_dim: int. Input and output dimensionality.
        intermediate_dim: int. Inner dimensionality of the gated projection.
        kernel_initializer: string or initializer. Initializer for Dense kernel
            weights. Defaults to `"glorot_uniform"`.

    Example:

    ```python
    ffn = keras_hub.layers.DeepSeekV31DenseFeedForward(
        hidden_dim=512,
        intermediate_dim=1024,
    )
    x = keras.random.normal((2, 16, 512))
    output = ffn(x)  # (2, 16, 512)
    ```

    Reference:
     - [DeepSeek-AI et al., 2024](https://arxiv.org/abs/2412.19437)
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.kernel_initializer = kernel_initializer

        self.gate_proj = keras.layers.Dense(
            intermediate_dim,
            activation="silu",
            use_bias=False,
            kernel_initializer=kernel_initializer,
            name="gate_proj",
            dtype=self.dtype_policy,
        )
        self.up_proj = keras.layers.Dense(
            intermediate_dim,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            name="up_proj",
            dtype=self.dtype_policy,
        )
        self.down_proj = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            name="down_proj",
            dtype=self.dtype_policy,
        )

    def build(self, input_shape):
        self.gate_proj.build(input_shape)
        self.up_proj.build(input_shape)
        inner_shape = list(input_shape[:-1]) + [self.intermediate_dim]
        self.down_proj.build(inner_shape)
        super().build(input_shape)

    def call(self, x):
        return self.down_proj(self.gate_proj(x) * self.up_proj(x))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "kernel_initializer": self.kernel_initializer,
            }
        )
        return config


class DeepSeekV31DecoderBlock(keras.layers.Layer):
    """Transformer decoder block for DeepSeek V31.

    Implements the pre-norm residual block structure from Figure 2 of the paper:

        x = x + Attention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    where FFN is either a dense SwiGLU network (for the first
    `first_k_dense_replace` layers) or a `DeepSeekV31MoE` layer.

    The KV cache format matches the MLA architecture: each layer stores a tuple
    `(c_kv, k_rope)` of compressed latents rather than full K/V tensors.

    Args:
        hidden_dim: int. Dimensionality of hidden states.
        num_query_heads: int. Number of query attention heads.
        num_key_value_heads: int. Number of key/value heads.
        intermediate_dim: int. Inner dimensionality of the FFN.
        q_lora_rank: int. Query down-projection rank.
        kv_lora_rank: int. KV latent rank (controls KV cache size per token).
        qk_nope_head_dim: int. Per-head content (non-RoPE) dimension.
        qk_rope_head_dim: int. Per-head RoPE dimension.
        v_head_dim: int. Per-head value dimension.
        num_routed_experts: int. Total routed experts in the MoE layer.
            Defaults to `256`.
        num_shared_experts: int. Always-active shared experts. Defaults to `1`.
        num_experts_per_tok: int. Top-K experts activated per token. Defaults
            to `8`.
        use_moe: bool. If `True`, uses `DeepSeekV31MoE` as the FFN; otherwise
            uses `DeepSeekV31DenseFeedForward`. Defaults to `True`.
        rope_max_wavelength: int. RoPE base wavelength. Defaults to `10000`.
        rope_scaling_factor: float. YaRN context extension factor. Defaults to
            `1.0`.
        yarn_original_max_position_embeddings: int. Pre-training context length
            for YaRN ramp computation. Defaults to `4096`.
        layer_norm_epsilon: float. RMSNorm epsilon. Defaults to `1e-6`.
        dropout: float. Dropout rate for attention weights and residuals.
            Defaults to `0.0`.
        kernel_initializer: string or initializer. Initializer for all
            sub-layer kernel weights. Defaults to `"glorot_uniform"`.

    Example:

    ```python
    block = keras_hub.layers.DeepSeekV31DecoderBlock(
        hidden_dim=512,
        num_query_heads=8,
        num_key_value_heads=8,
        intermediate_dim=1024,
        q_lora_rank=256,
        kv_lora_rank=128,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        num_routed_experts=8,
        num_experts_per_tok=2,
        use_moe=True,
    )
    x = keras.random.normal((2, 16, 512))
    output = block(x)  # (2, 16, 512)
    ```

    Reference:
     - [DeepSeek-AI et al., 2024](https://arxiv.org/abs/2412.19437)
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
        yarn_original_max_position_embeddings=4096,
        layer_norm_epsilon=1e-6,
        dropout=0.0,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Store every __init__ argument as
        # an attribute (style guide requirement).
        self.hidden_dim = hidden_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_dim = intermediate_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.use_moe = use_moe
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.yarn_original_max_position_embeddings = (
            yarn_original_max_position_embeddings
        )
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout
        self.kernel_initializer = kernel_initializer

        self.pre_attention_norm = DeepSeekV31RMSNorm(
            epsilon=layer_norm_epsilon,
            name="pre_attention_norm",
            dtype=self.dtype_policy,
        )
        self.attention = DeepSeekV31Attention(
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
            yarn_original_max_position_embeddings=yarn_original_max_position_embeddings,  # noqa: E501
            attention_dropout=dropout,
            kernel_initializer=kernel_initializer,
            name="attention",
            dtype=self.dtype_policy,
        )
        self.pre_ffn_norm = DeepSeekV31RMSNorm(
            epsilon=layer_norm_epsilon,
            name="pre_ffn_norm",
            dtype=self.dtype_policy,
        )

        if use_moe:
            self.ffn = DeepSeekV31MoE(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_routed_experts=num_routed_experts,
                num_shared_experts=num_shared_experts,
                num_experts_per_tok=num_experts_per_tok,
                kernel_initializer=kernel_initializer,
                name="ffn",
                dtype=self.dtype_policy,
            )
        else:
            self.ffn = DeepSeekV31DenseFeedForward(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                kernel_initializer=kernel_initializer,
                name="ffn",
                dtype=self.dtype_policy,
            )

        self.residual_dropout = keras.layers.Dropout(
            dropout,
            dtype=self.dtype_policy,
        )

    def call(
        self,
        hidden_states,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
        training=False,
    ):
        # Pre-attention norm + MLA.
        attn_out = self.attention(
            self.pre_attention_norm(hidden_states),
            attention_mask=attention_mask,
            cache=cache,
            cache_update_index=cache_update_index,
            training=training,
        )

        if isinstance(attn_out, tuple):
            attn_out, new_cache = attn_out
        else:
            new_cache = None

        hidden_states = hidden_states + self.residual_dropout(
            attn_out, training=training
        )

        # Pre-FFN norm + FFN (MoE or Dense).
        ffn_out = self.ffn(self.pre_ffn_norm(hidden_states), training=training)
        hidden_states = hidden_states + self.residual_dropout(
            ffn_out, training=training
        )

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
        # When cache is provided the layer returns (hidden_states, new_cache).
        # This override is required so Keras symbolic tracing handles the tuple
        # output correctly rather than assuming a single tensor output.
        try:
            input_shape = hidden_states.shape
        except Exception:
            input_shape = None

        hidden_out = keras.KerasTensor(input_shape, dtype=self.compute_dtype)

        if cache is not None:
            c_kv, k_rope = cache
            new_cache_spec = (
                keras.KerasTensor(
                    getattr(c_kv, "shape", None), dtype=self.compute_dtype
                ),
                keras.KerasTensor(
                    getattr(k_rope, "shape", None), dtype=self.compute_dtype
                ),
            )
            return hidden_out, new_cache_spec

        return hidden_out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "intermediate_dim": self.intermediate_dim,
                "q_lora_rank": self.q_lora_rank,
                "kv_lora_rank": self.kv_lora_rank,
                "qk_nope_head_dim": self.qk_nope_head_dim,
                "qk_rope_head_dim": self.qk_rope_head_dim,
                "v_head_dim": self.v_head_dim,
                "num_routed_experts": self.num_routed_experts,
                "num_shared_experts": self.num_shared_experts,
                "num_experts_per_tok": self.num_experts_per_tok,
                "use_moe": self.use_moe,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "yarn_original_max_position_embeddings": (
                    self.yarn_original_max_position_embeddings
                ),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "kernel_initializer": self.kernel_initializer,
            }
        )
        return config
