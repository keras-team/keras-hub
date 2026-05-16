"""OpenAI Privacy Filter encoder-only backbone.

An encoder-only MoE transformer with bidirectional sliding-window attention,
interleaved YaRN RoPE, attention sinks, and GLU experts.

References:
- HF: transformers/models/openai_privacy_filter/modular_openai_privacy_filter.py
- Parent: transformers/models/gpt_oss/modeling_gpt_oss.py
"""

import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.utils.keras_utils import clone_initializer


# ---------------------------------------------------------------------------
# RMSNorm — weight * x (NOT (1+weight)*x)
# ---------------------------------------------------------------------------
class OpenAIPrivacyFilterRMSNorm(keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=(input_shape[-1],),
            initializer="ones",
            dtype=self.variable_dtype,
        )
        self.built = True

    def call(self, x):
        x = ops.cast(x, "float32")
        var = ops.mean(ops.power(x, 2), axis=-1, keepdims=True)
        x = x * ops.rsqrt(var + self.epsilon)
        return ops.cast(x * self.scale, self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


# ---------------------------------------------------------------------------
# Bidirectional sliding-window attention with sinks & interleaved RoPE
# ---------------------------------------------------------------------------
class OpenAIPrivacyFilterAttention(keras.layers.Layer):
    """Bidirectional sliding-window GQA with sinks and interleaved RoPE.

    Key differences from GptOssAttention:
    - Bidirectional (not causal) attention mask
    - Interleaved RoPE layout (even/odd indices)
    - Dual Q/K scaling: Q*s, K*s where s = head_dim^-0.25
    """

    def __init__(
        self,
        num_query_heads,
        num_key_value_heads,
        head_dim=64,
        rope_max_wavelength=150000.0,
        rope_scaling_factor=1.0,
        sliding_window=128,
        attention_dropout=0.0,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_dim = num_query_heads * head_dim
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.num_key_value_groups = num_query_heads // num_key_value_heads
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        # Dual scaling: head_dim^-0.25 applied to Q and K separately
        self._scaling = head_dim**-0.25
        self._kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )

    def build(self, inputs_shape):
        hidden_dim = inputs_shape[-1]

        self.query_dense = keras.layers.EinsumDense(
            equation="bqm,muh->bquh",
            output_shape=(None, self.num_query_heads, self.head_dim),
            bias_axes="uh",
            kernel_initializer=self._kernel_initializer,
            bias_initializer="zeros",
            dtype=self.dtype_policy,
            name="query",
        )
        self.query_dense.build(inputs_shape)

        self.key_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            bias_axes="vh",
            kernel_initializer=self._kernel_initializer,
            bias_initializer="zeros",
            dtype=self.dtype_policy,
            name="key",
        )
        self.key_dense.build(inputs_shape)

        self.value_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            bias_axes="vh",
            kernel_initializer=self._kernel_initializer,
            bias_initializer="zeros",
            dtype=self.dtype_policy,
            name="value",
        )
        self.value_dense.build(inputs_shape)

        self.output_dense = keras.layers.EinsumDense(
            equation="bquh,uhm->bqm",
            output_shape=(None, hidden_dim),
            bias_axes="m",
            kernel_initializer=self._kernel_initializer,
            bias_initializer="zeros",
            dtype=self.dtype_policy,
            name="attention_output",
        )
        self.output_dense.build(
            (None, None, self.num_query_heads, self.head_dim)
        )

        self.dropout_layer = keras.layers.Dropout(
            rate=self.attention_dropout,
            dtype=self.dtype_policy,
        )

        self.rotary_embedding_layer = RotaryEmbedding(
            max_wavelength=self.rope_max_wavelength,
            scaling_factor=self.rope_scaling_factor,
            rope_type="yarn",
            beta_fast=32.0,
            beta_slow=1.0,
            original_max_position_embeddings=4096,
            dtype=self.dtype_policy,
        )

        self.sinks = self.add_weight(
            shape=(self.num_query_heads,),
            initializer="random_normal",
            dtype=self.variable_dtype,
            name="sinks",
        )

        self.built = True

    def _apply_interleaved_rotary(self, x, cos, sin):
        """Interleaved RoPE: operates on even/odd indices."""
        first_half = x[..., ::2]
        second_half = x[..., 1::2]
        first_ = first_half * cos - second_half * sin
        second_ = second_half * cos + first_half * sin
        return ops.reshape(
            ops.stack([first_, second_], axis=-1),
            ops.shape(x),
        )

    def call(self, hidden_states, attention_mask=None, training=None):
        query = self.query_dense(hidden_states)
        key = self.key_dense(hidden_states)
        value = self.value_dense(hidden_states)

        # Compute cos/sin from the RotaryEmbedding layer, then apply
        # interleaved layout manually (KerasHub default uses split-half).
        cos, sin = self.rotary_embedding_layer._compute_cos_sin_embedding(
            query, start_index=0
        )
        # cos/sin shape: (1, seq, 1, head_dim) with split-half layout
        # [f0,f1,...,f_{d/2-1}, f0,f1,...,f_{d/2-1}].
        # For interleaved RoPE we need [f0,...,f_{d/2-1}] of size head_dim/2
        # to match x[..., ::2] and x[..., 1::2] which each have head_dim/2.
        half = self.head_dim // 2
        cos = cos[..., :half]
        sin = sin[..., :half]
        query = self._apply_interleaved_rotary(query, cos, sin)
        key = self._apply_interleaved_rotary(key, cos, sin)

        # Dual scaling (Q*s, K*s instead of attn_scores/sqrt(d))
        scaling = ops.cast(self._scaling, self.compute_dtype)
        query = query * scaling
        key = key * scaling

        # Repeat KV heads for GQA
        key = ops.repeat(key, repeats=self.num_key_value_groups, axis=2)
        value = ops.repeat(value, repeats=self.num_key_value_groups, axis=2)

        # Attention scores: [b, heads, q, k]
        attn_scores = ops.einsum("bquh,bkuh->buqk", query, key)

        # Bidirectional sliding window mask: |i - j| <= sliding_window
        q_len = ops.shape(attn_scores)[2]
        kv_len = ops.shape(attn_scores)[3]
        q_pos = ops.cast(ops.arange(q_len), "int32")
        kv_pos = ops.cast(ops.arange(kv_len), "int32")
        distance = ops.abs(q_pos[:, None] - kv_pos[None, :])
        sw_mask = distance <= self.sliding_window

        # Apply padding mask (bidirectional)
        large_neg = ops.cast(
            -1e9 if self.compute_dtype == "float32" else -1e4,
            self.compute_dtype,
        )
        attn_scores = ops.where(
            sw_mask[None, None, :, :], attn_scores, large_neg
        )
        if attention_mask is not None:
            attn_scores = ops.where(attention_mask, attn_scores, large_neg)

        # Append sink logits
        b = ops.shape(attn_scores)[0]
        q = ops.shape(attn_scores)[2]
        sink_logits = ops.reshape(self.sinks, (1, self.num_query_heads, 1, 1))
        sink_logits = ops.broadcast_to(
            sink_logits, (b, self.num_query_heads, q, 1)
        )
        combined = ops.concatenate([attn_scores, sink_logits], axis=-1)

        # Stabilize and softmax
        max_logits = ops.stop_gradient(
            ops.max(combined, axis=-1, keepdims=True)
        )
        combined = combined - max_logits
        probs = ops.softmax(ops.cast(combined, "float32"), axis=-1)

        # Drop sink probability, keep only real token probs
        attn_weights = ops.cast(probs[..., :-1], self.compute_dtype)
        attn_weights = self.dropout_layer(attn_weights, training=training)

        attn_output = ops.einsum("buqk,bkuh->bquh", attn_weights, value)
        return self.output_dense(attn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "sliding_window": self.sliding_window,
                "attention_dropout": self.attention_dropout,
            }
        )
        return config


# ---------------------------------------------------------------------------
# Experts — GLU with alpha=1.702, limit=7.0, has biases
# ---------------------------------------------------------------------------
class OpenAIPrivacyFilterExperts(keras.layers.Layer):
    def __init__(
        self,
        num_experts,
        hidden_dim,
        intermediate_dim,
        kernel_initializer="glorot_uniform",
        alpha=1.702,
        limit=7.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.alpha = alpha
        self.limit = limit

    def build(self, _):
        self.gate_up_proj = self.add_weight(
            shape=(
                self.num_experts,
                self.hidden_dim,
                2 * self.intermediate_dim,
            ),
            initializer=self.kernel_initializer,
            name="gate_up_proj",
        )
        self.gate_up_proj_bias = self.add_weight(
            shape=(self.num_experts, 2 * self.intermediate_dim),
            initializer="zeros",
            name="gate_up_proj_bias",
        )
        self.down_proj = self.add_weight(
            shape=(self.num_experts, self.intermediate_dim, self.hidden_dim),
            initializer=self.kernel_initializer,
            name="down_proj",
        )
        self.down_proj_bias = self.add_weight(
            shape=(self.num_experts, self.hidden_dim),
            initializer="zeros",
            name="down_proj_bias",
        )
        self.built = True

    def call(self, hidden_states):
        # hidden_states: (num_tokens, hidden_dim)
        gate_up = ops.einsum("th,ehm->etm", hidden_states, self.gate_up_proj)
        gate_up = gate_up + self.gate_up_proj_bias[:, None, :]

        # Concatenated layout: first half = gate, second half = up
        gate = gate_up[..., : self.intermediate_dim]
        up = gate_up[..., self.intermediate_dim :]

        gate = ops.clip(gate, -1e9, self.limit)
        up = ops.clip(up, -self.limit, self.limit)

        glu = gate * ops.sigmoid(gate * self.alpha)
        gated_output = (up + 1) * glu

        out = ops.einsum("etm,emh->eth", gated_output, self.down_proj)
        out = out + self.down_proj_bias[:, None, :]
        return out


# ---------------------------------------------------------------------------
# Router — Dense with bias, fp32 cast, softmax+top_k, div by top_k
# ---------------------------------------------------------------------------
class OpenAIPrivacyFilterTopKRouter(keras.layers.Layer):
    def __init__(
        self,
        num_experts,
        top_k,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.top_k = top_k
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

    def build(self, hidden_states_shape):
        self.router_dense = keras.layers.Dense(
            self.num_experts,
            use_bias=True,
            kernel_initializer=self.kernel_initializer,
            dtype=self.dtype_policy,
            name="router_dense",
        )
        self.router_dense.build(hidden_states_shape)
        self.built = True

    def call(self, hidden_states):
        # Force fp32 for routing
        router_logits = self.router_dense(ops.cast(hidden_states, "float32"))
        routing_weights, selected_experts = ops.top_k(
            router_logits, k=self.top_k
        )
        routing_weights = ops.softmax(routing_weights, axis=-1)
        # Additional scaling: divide by top_k
        routing_weights = routing_weights / self.top_k

        expert_mask = ops.one_hot(selected_experts, self.num_experts)
        expert_mask = ops.cast(expert_mask, dtype=routing_weights.dtype)
        weighted_mask = expert_mask * ops.expand_dims(routing_weights, axis=-1)
        router_scores = ops.sum(weighted_mask, axis=1)
        return router_scores


# ---------------------------------------------------------------------------
# Sparse MoE block — router + experts, output *= num_experts_per_tok
# ---------------------------------------------------------------------------
class OpenAIPrivacyFilterSparseMoeBlock(keras.layers.Layer):
    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_experts,
        top_k=4,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.kernel_initializer = kernel_initializer

    def build(self, decoder_sequence_shape):
        self.router = OpenAIPrivacyFilterTopKRouter(
            num_experts=self.num_experts,
            top_k=self.top_k,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="router",
        )
        self.router.build(decoder_sequence_shape)

        self.experts = OpenAIPrivacyFilterExperts(
            num_experts=self.num_experts,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="experts",
        )
        self.experts.build(decoder_sequence_shape)
        self.built = True

    def call(self, hidden_states):
        batch_size, seq_len, _ = ops.shape(hidden_states)
        flat = ops.reshape(hidden_states, (-1, self.hidden_dim))

        router_scores = self.router(flat)
        expert_outputs = self.experts(flat)

        # Weight expert outputs by router scores and sum
        router_scores_t = ops.transpose(router_scores)
        router_scores_expanded = ops.expand_dims(router_scores_t, axis=-1)
        weighted = expert_outputs * router_scores_expanded
        final = ops.sum(weighted, axis=0)

        # Additional scaling: multiply by num_experts_per_tok
        final = final * self.top_k

        return ops.reshape(final, (batch_size, seq_len, self.hidden_dim))


# ---------------------------------------------------------------------------
# Encoder layer — pre-norm, attn, residual, post-norm, MoE, residual
# ---------------------------------------------------------------------------
class OpenAIPrivacyFilterEncoderLayer(keras.layers.Layer):
    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_query_heads,
        num_key_value_heads,
        head_dim=64,
        num_experts=128,
        top_k=4,
        rope_max_wavelength=150000.0,
        rope_scaling_factor=1.0,
        rms_norm_eps=1e-5,
        sliding_window=128,
        attention_dropout=0.0,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.rms_norm_eps = rms_norm_eps
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

    def build(self, input_shape):
        self.input_layernorm = OpenAIPrivacyFilterRMSNorm(
            epsilon=self.rms_norm_eps,
            dtype=self.dtype_policy,
            name="input_layernorm",
        )
        self.input_layernorm.build(input_shape)

        self.self_attention = OpenAIPrivacyFilterAttention(
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            rope_max_wavelength=self.rope_max_wavelength,
            rope_scaling_factor=self.rope_scaling_factor,
            sliding_window=self.sliding_window,
            attention_dropout=self.attention_dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="self_attention",
        )
        self.self_attention.build(input_shape)

        self.post_attention_layernorm = OpenAIPrivacyFilterRMSNorm(
            epsilon=self.rms_norm_eps,
            dtype=self.dtype_policy,
            name="post_attention_layernorm",
        )
        self.post_attention_layernorm.build(input_shape)

        self.sparse_moe_block = OpenAIPrivacyFilterSparseMoeBlock(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="sparse_moe_block",
        )
        self.sparse_moe_block.build(input_shape)
        self.built = True

    def call(self, hidden_states, attention_mask=None, training=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attention(
            hidden_states, attention_mask=attention_mask, training=training
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.sparse_moe_block(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def compute_output_spec(
        self, hidden_states, attention_mask=None, training=None
    ):
        return keras.KerasTensor(
            shape=hidden_states.shape, dtype=hidden_states.dtype
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "rms_norm_eps": self.rms_norm_eps,
                "sliding_window": self.sliding_window,
                "attention_dropout": self.attention_dropout,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
@keras_hub_export("keras_hub.models.OpenAIPrivacyFilterBackbone")
class OpenAIPrivacyFilterBackbone(Backbone):
    """OpenAI Privacy Filter encoder-only backbone.

    A compact (~400M param) encoder-only MoE transformer for PII detection.
    Uses bidirectional sliding-window attention with sinks, interleaved YaRN
    RoPE, and GLU experts.

    Args:
        vocabulary_size: int. Size of the token vocabulary.
        num_layers: int. Number of encoder layers.
        num_query_heads: int. Number of query attention heads.
        num_key_value_heads: int. Number of KV heads (GQA).
        hidden_dim: int. Hidden dimension size.
        intermediate_dim: int. FFN intermediate dimension.
        head_dim: int. Per-head dimension. Defaults to `64`.
        num_experts: int. Number of MoE experts. Defaults to `128`.
        top_k: int. Number of active experts per token. Defaults to `4`.
        rope_max_wavelength: float. RoPE base theta. Defaults to `150000.0`.
        rope_scaling_factor: float. YaRN scaling factor. Defaults to `1.0`.
        rms_norm_eps: float. RMSNorm epsilon. Defaults to `1e-5`.
        sliding_window: int. Bidirectional sliding window size.
            Defaults to `128`.
        attention_dropout: float. Attention dropout. Defaults to `0.0`.
        dtype: string or `keras.mixed_precision.DTypePolicy`.

    Example:
    ```python
    import numpy as np
    import keras_hub

    model = keras_hub.models.OpenAIPrivacyFilterBackbone(
        vocabulary_size=200064,
        num_layers=8,
        num_query_heads=14,
        num_key_value_heads=2,
        hidden_dim=640,
        intermediate_dim=640,
    )
    input_data = {
        "token_ids": np.ones((1, 12), dtype="int32"),
        "padding_mask": np.ones((1, 12), dtype="int32"),
    }
    model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_query_heads,
        num_key_value_heads,
        hidden_dim,
        intermediate_dim,
        head_dim=64,
        num_experts=128,
        top_k=4,
        rope_max_wavelength=150000.0,
        rope_scaling_factor=1.0,
        rms_norm_eps=1e-5,
        sliding_window=128,
        attention_dropout=0.0,
        dtype=None,
        **kwargs,
    ):
        # === Layers ===
        self.token_embedding = keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            dtype=dtype,
            name="token_embedding",
        )
        self.encoder_layers = []
        for i in range(num_layers):
            layer = OpenAIPrivacyFilterEncoderLayer(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_query_heads=num_query_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                num_experts=num_experts,
                top_k=top_k,
                rope_max_wavelength=rope_max_wavelength,
                rope_scaling_factor=rope_scaling_factor,
                rms_norm_eps=rms_norm_eps,
                sliding_window=sliding_window,
                attention_dropout=attention_dropout,
                dtype=dtype,
                name=f"encoder_layer_{i}",
            )
            self.encoder_layers.append(layer)
        self.final_layernorm = OpenAIPrivacyFilterRMSNorm(
            epsilon=rms_norm_eps,
            dtype=dtype,
            name="final_layernorm",
        )

        # === Functional Model ===
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask_input = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        x = self.token_embedding(token_id_input)

        # Build padding attention mask: (batch, 1, seq, seq)
        # True where both query and key positions are non-padding
        pm_float = ops.cast(padding_mask_input, x.dtype)
        attn_mask = pm_float[:, None, :] * pm_float[:, :, None]
        attn_mask = ops.cast(attn_mask[:, None, :, :], "bool")

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, attention_mask=attn_mask)

        x = self.final_layernorm(x)

        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask_input,
            },
            outputs=x,
            dtype=dtype,
            **kwargs,
        )

        # === Config ===
        self.vocabulary_size = vocabulary_size
        self.num_layers = num_layers
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.rms_norm_eps = rms_norm_eps
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "num_layers": self.num_layers,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "head_dim": self.head_dim,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "rms_norm_eps": self.rms_norm_eps,
                "sliding_window": self.sliding_window,
                "attention_dropout": self.attention_dropout,
            }
        )
        return config
