"""Portable core layers for HRM-Text."""

import keras
from keras import ops


class HrmTextAttentionMask(keras.layers.Layer):
    """Builds the dynamic causal or PrefixLM attention mask."""

    def call(self, token_type_ids, padding_mask):
        sequence_length = ops.shape(token_type_ids)[1]
        positions = ops.arange(sequence_length)
        causal = positions[None, :] <= positions[:, None]
        prefix = ops.cast(token_type_ids == 1, "bool")
        prefix_mask = ops.logical_and(prefix[:, :, None], prefix[:, None, :])
        allowed = ops.logical_or(causal[None, :, :], prefix_mask)
        valid = ops.cast(padding_mask, "bool")
        valid_mask = ops.logical_and(valid[:, :, None], valid[:, None, :])
        return ops.logical_and(allowed, valid_mask)


class HrmTextRMSNorm(keras.layers.Layer):
    """Parameterless RMS normalization used by HRM-Text."""

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        variance = ops.mean(ops.square(inputs), axis=-1, keepdims=True)
        outputs = inputs * ops.rsqrt(variance + self.epsilon)
        return ops.cast(outputs, inputs.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


class HrmTextInitialState(keras.layers.Layer):
    """Broadcasts HRM-Text's learned low-level initial state."""

    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def build(self, inputs_shape):
        self.z_L_init = self.add_weight(
            name="z_L_init",
            shape=(self.hidden_dim,),
            initializer="zeros",
            trainable=True,
        )
        super().build(inputs_shape)

    def call(self, inputs):
        initial_state = ops.cast(self.z_L_init, inputs.dtype)
        return ops.broadcast_to(initial_state, ops.shape(inputs))

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


class HrmTextMLP(keras.layers.Layer):
    """SwiGLU feed-forward network."""

    def __init__(self, intermediate_dim, **kwargs):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim

    def build(self, inputs_shape):
        hidden_dim = inputs_shape[-1]
        self.gate_proj = keras.layers.Dense(
            self.intermediate_dim,
            use_bias=False,
            name="gate_proj",
            dtype=self.dtype_policy,
        )
        self.up_proj = keras.layers.Dense(
            self.intermediate_dim,
            use_bias=False,
            name="up_proj",
            dtype=self.dtype_policy,
        )
        self.down_proj = keras.layers.Dense(
            hidden_dim,
            use_bias=False,
            name="down_proj",
            dtype=self.dtype_policy,
        )
        super().build(inputs_shape)

    def call(self, inputs):
        hidden_states = ops.silu(self.gate_proj(inputs)) * self.up_proj(inputs)
        return self.down_proj(hidden_states)

    def get_config(self):
        config = super().get_config()
        config.update({"intermediate_dim": self.intermediate_dim})
        return config


class HrmTextAttention(keras.layers.Layer):
    """Multi-head RoPE attention with HRM-Text's sigmoid output gate."""

    def __init__(self, num_heads, head_dim, rope_theta=10000.0, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.hidden_dim = num_heads * head_dim

    def build(self, inputs_shape):
        for name in ("q_proj", "k_proj", "v_proj", "gate_proj", "o_proj"):
            setattr(
                self,
                name,
                keras.layers.Dense(
                    self.hidden_dim,
                    use_bias=False,
                    name=name,
                    dtype=self.dtype_policy,
                ),
            )
        super().build(inputs_shape)

    def _apply_rope(self, inputs, start_index):
        sequence_length = ops.shape(inputs)[1]
        positions = ops.arange(sequence_length, dtype="int32") + start_index
        frequencies = ops.arange(0, self.head_dim, 2, dtype="float32")
        frequencies = 1.0 / (self.rope_theta ** (frequencies / self.head_dim))
        angles = (
            ops.expand_dims(ops.cast(positions, "float32"), -1) * frequencies
        )
        angles = ops.concatenate((angles, angles), axis=-1)
        cos = ops.cast(ops.cos(angles)[None, :, None, :], inputs.dtype)
        sin = ops.cast(ops.sin(angles)[None, :, None, :], inputs.dtype)
        half = self.head_dim // 2
        rotated = ops.concatenate(
            (-inputs[..., half:], inputs[..., :half]), axis=-1
        )
        return inputs * cos + rotated * sin

    def call(
        self,
        hidden_states,
        attention_mask,
        cache=None,
        cache_update_index=None,
    ):
        shape = ops.shape(hidden_states)
        batch_size, sequence_length = shape[0], shape[1]
        reshape_shape = (
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )
        query = ops.reshape(self.q_proj(hidden_states), reshape_shape)
        key_update = ops.reshape(self.k_proj(hidden_states), reshape_shape)
        value_update = ops.reshape(self.v_proj(hidden_states), reshape_shape)
        gate = ops.reshape(self.gate_proj(hidden_states), reshape_shape)
        start_index = 0 if cache_update_index is None else cache_update_index
        query = self._apply_rope(query, start_index)
        key_update = self._apply_rope(key_update, start_index)

        if cache is None:
            key, value = key_update, value_update
        else:
            key_cache, value_cache = cache[:, 0], cache[:, 1]
            key_cache = ops.slice_update(
                key_cache, [0, start_index, 0, 0], key_update
            )
            value_cache = ops.slice_update(
                value_cache, [0, start_index, 0, 0], value_update
            )
            # Attend over the full, statically shaped cache. The causal mask
            # excludes its not-yet-written positions. This avoids dynamic
            # slice sizes in JAX's compiled generation loop.
            key, value = key_cache, value_cache
            cache = ops.stack((key_cache, value_cache), axis=1)

        scores = ops.einsum("bqhd,bkhd->bhqk", query, key)
        scores = scores * (self.head_dim**-0.5)
        mask = ops.cast(attention_mask[:, None, :, :], "bool")
        scores = ops.where(mask, scores, ops.cast(-1e30, scores.dtype))
        weights = ops.softmax(scores, axis=-1)
        output = ops.einsum("bhqk,bkhd->bqhd", weights, value)
        output = ops.sigmoid(gate) * output
        output = ops.reshape(
            output, (batch_size, sequence_length, self.hidden_dim)
        )
        output = self.o_proj(output)
        return (output, cache) if cache is not None else output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "rope_theta": self.rope_theta,
            }
        )
        return config


class HrmTextDecoderBlock(keras.layers.Layer):
    """Pre-norm HRM-Text transformer block."""

    def __init__(
        self,
        num_heads,
        head_dim,
        intermediate_dim,
        rope_theta=10000.0,
        rms_norm_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.intermediate_dim = intermediate_dim
        self.rope_theta = rope_theta
        self.rms_norm_epsilon = rms_norm_epsilon

    def build(self, inputs_shape):
        self.input_layernorm = HrmTextRMSNorm(
            self.rms_norm_epsilon,
            name="input_layernorm",
            dtype=self.dtype_policy,
        )
        self.self_attn = HrmTextAttention(
            self.num_heads,
            self.head_dim,
            self.rope_theta,
            name="self_attn",
            dtype=self.dtype_policy,
        )
        self.post_attention_layernorm = HrmTextRMSNorm(
            self.rms_norm_epsilon,
            name="post_attention_layernorm",
            dtype=self.dtype_policy,
        )
        self.mlp = HrmTextMLP(
            self.intermediate_dim, name="mlp", dtype=self.dtype_policy
        )
        super().build(inputs_shape)

    def call(
        self, hidden_states, attention_mask, cache=None, cache_update_index=None
    ):
        residual = hidden_states
        output = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask,
            cache=cache,
            cache_update_index=cache_update_index,
        )
        if cache is not None:
            output, cache = output
        hidden_states = residual + output
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return (hidden_states, cache) if cache is not None else hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "intermediate_dim": self.intermediate_dim,
                "rope_theta": self.rope_theta,
                "rms_norm_epsilon": self.rms_norm_epsilon,
            }
        )
        return config


class HrmTextStack(keras.layers.Layer):
    """One shared H or L stack with logical cache-slot indexing."""

    def __init__(self, num_layers, **block_kwargs):
        layer_kwargs = {}
        for key in ("name", "dtype"):
            if key in block_kwargs:
                layer_kwargs[key] = block_kwargs.pop(key)
        super().__init__(**layer_kwargs)
        self.num_layers = num_layers
        self.block_kwargs = block_kwargs
        self.layers = [
            HrmTextDecoderBlock(
                name=f"layers_{index}", dtype=self.dtype_policy, **block_kwargs
            )
            for index in range(num_layers)
        ]
        self.final_norm = HrmTextRMSNorm(
            block_kwargs["rms_norm_epsilon"],
            name="final_norm",
            dtype=self.dtype_policy,
        )

    def call(
        self,
        hidden_states,
        attention_mask,
        cache=None,
        cache_update_index=None,
        cycle_offset=0,
    ):
        updated_cache = [] if cache is not None else None
        for index, layer in enumerate(self.layers):
            if cache is None:
                hidden_states = layer(hidden_states, attention_mask)
            else:
                hidden_states, layer_cache = layer(
                    hidden_states,
                    attention_mask,
                    cache=cache[:, cycle_offset + index],
                    cache_update_index=cache_update_index,
                )
                updated_cache.append(layer_cache)
        hidden_states = self.final_norm(hidden_states)
        if cache is None:
            return hidden_states
        return hidden_states, updated_cache
