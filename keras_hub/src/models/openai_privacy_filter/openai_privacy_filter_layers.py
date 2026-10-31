import keras
from keras import ops

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.gpt_oss.gpt_oss_layer_norm import (
    GptOssLayerNormalization,
)
from keras_hub.src.utils.keras_utils import clone_initializer


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
        rope_type="yarn",
        beta_fast=32.0,
        beta_slow=1.0,
        original_max_position_embeddings=4096,
        truncate=False,
        sliding_window=128,
        attention_dropout=0.0,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.num_key_value_groups = num_query_heads // num_key_value_heads
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_type = rope_type
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.original_max_position_embeddings = original_max_position_embeddings
        self.truncate = truncate
        # Dual scaling: head_dim^-0.25 applied to Q and K separately
        self._scaling = head_dim**-0.25
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

    def build(self, inputs_shape):
        hidden_dim = inputs_shape[-1]

        self.query_dense = keras.layers.EinsumDense(
            equation="bqm,muh->bquh",
            output_shape=(None, self.num_query_heads, self.head_dim),
            bias_axes="uh",
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer="zeros",
            dtype=self.dtype_policy,
            name="query",
        )
        self.query_dense.build(inputs_shape)

        self.key_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            bias_axes="vh",
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer="zeros",
            dtype=self.dtype_policy,
            name="key",
        )
        self.key_dense.build(inputs_shape)

        self.value_dense = keras.layers.EinsumDense(
            equation="bkm,mvh->bkvh",
            output_shape=(None, self.num_key_value_heads, self.head_dim),
            bias_axes="vh",
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer="zeros",
            dtype=self.dtype_policy,
            name="value",
        )
        self.value_dense.build(inputs_shape)

        self.output_dense = keras.layers.EinsumDense(
            equation="bquh,uhm->bqm",
            output_shape=(None, hidden_dim),
            bias_axes="m",
            kernel_initializer=clone_initializer(self.kernel_initializer),
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
            rope_type=self.rope_type,
            beta_fast=self.beta_fast,
            beta_slow=self.beta_slow,
            original_max_position_embeddings=self.original_max_position_embeddings,
            truncate=self.truncate,
            dtype=self.dtype_policy,
        )

        self.sinks = self.add_weight(
            shape=(self.num_query_heads,),
            initializer="random_normal",
            dtype="float32",
            name="sinks",
        )

        self.built = True

    def _deinterleave(self, x):
        """Reorder `[x0, x1, x2, x3, ...]` to `[x0, x2, ..., x1, x3, ...]`.

        HF applies RoPE to interleaved (even, odd) channel pairs. After this
        reorder, the split-half rotation in `RotaryEmbedding` rotates the
        same pairs. Q and K share the reorder, so attention scores are
        unchanged.
        """
        # `stack` + `reshape` rather than `concatenate`, as in
        # `RotaryEmbedding`, to avoid an XLA compilation bug on jax.
        reordered = ops.stack((x[..., ::2], x[..., 1::2]), axis=-2)
        return ops.reshape(reordered, ops.shape(x))

    def call(self, hidden_states, attention_mask=None, training=None):
        query = self.query_dense(hidden_states)
        key = self.key_dense(hidden_states)
        value = self.value_dense(hidden_states)

        query = self.rotary_embedding_layer(self._deinterleave(query))
        key = self.rotary_embedding_layer(self._deinterleave(key))

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
        combined = ops.concatenate(
            [ops.cast(attn_scores, "float32"), sink_logits], axis=-1
        )

        # Softmax in float32 for numerical stability
        probs = ops.softmax(combined, axis=-1)

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
                "rope_type": self.rope_type,
                "beta_fast": self.beta_fast,
                "beta_slow": self.beta_slow,
                "original_max_position_embeddings": (
                    self.original_max_position_embeddings
                ),
                "truncate": self.truncate,
                "sliding_window": self.sliding_window,
                "attention_dropout": self.attention_dropout,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config


# ---------------------------------------------------------------------------
# Experts — GLU with swiglu_alpha=1.702, swiglu_limit=7.0, has biases
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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "alpha": self.alpha,
                "limit": self.limit,
            }
        )
        return config


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
        router_logits = self.router_dense(hidden_states)
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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config


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
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

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
            alpha=self.swiglu_alpha,
            limit=self.swiglu_limit,
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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "swiglu_alpha": self.swiglu_alpha,
                "swiglu_limit": self.swiglu_limit,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config


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
        rope_type="yarn",
        beta_fast=32.0,
        beta_slow=1.0,
        original_max_position_embeddings=4096,
        truncate=False,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
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
        self.rope_type = rope_type
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.original_max_position_embeddings = original_max_position_embeddings
        self.truncate = truncate
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit
        self.rms_norm_eps = rms_norm_eps
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

    def build(self, input_shape):
        self.input_layernorm = GptOssLayerNormalization(
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
            rope_type=self.rope_type,
            beta_fast=self.beta_fast,
            beta_slow=self.beta_slow,
            original_max_position_embeddings=self.original_max_position_embeddings,
            truncate=self.truncate,
            sliding_window=self.sliding_window,
            attention_dropout=self.attention_dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="self_attention",
        )
        self.self_attention.build(input_shape)

        self.post_attention_layernorm = GptOssLayerNormalization(
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
            swiglu_alpha=self.swiglu_alpha,
            swiglu_limit=self.swiglu_limit,
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
                "rope_type": self.rope_type,
                "beta_fast": self.beta_fast,
                "beta_slow": self.beta_slow,
                "original_max_position_embeddings": (
                    self.original_max_position_embeddings
                ),
                "truncate": self.truncate,
                "swiglu_alpha": self.swiglu_alpha,
                "swiglu_limit": self.swiglu_limit,
                "rms_norm_eps": self.rms_norm_eps,
                "sliding_window": self.sliding_window,
                "attention_dropout": self.attention_dropout,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config
