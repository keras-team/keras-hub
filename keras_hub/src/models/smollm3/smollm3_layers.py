from keras import activations
from keras import initializers
from keras import layers
from keras import ops

from keras_hub.src.models.smollm3.smollm3_utils import apply_rotary_pos_emb
from keras_hub.src.models.smollm3.smollm3_utils import eager_attention_forward
from keras_hub.src.models.smollm3.smollm3_utils import rope_init


class SmolLM3Attention(layers.Layer):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_bias: bool,
        attention_dropout: float,
        rope_layer_enabled_list: list[bool],
        layer_types: list[str],
        layer_idx: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_layer_enabled_list = rope_layer_enabled_list
        self.layer_types = layer_types

        self.layer_idx = layer_idx

        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_groups = (
            self.num_attention_heads // self.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.is_causal = True

        self.q_proj = layers.Dense(
            self.num_attention_heads * self.head_dim,
            use_bias=self.attention_bias,
            name="q_proj",
        )
        self.k_proj = layers.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=self.attention_bias,
            name="k_proj",
        )
        self.v_proj = layers.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=self.attention_bias,
            name="v_proj",
        )
        self.o_proj = layers.Dense(
            self.hidden_size, use_bias=self.attention_bias, name="o_proj"
        )

        self.use_rope = (
            self.rope_layer_enabled_list[self.layer_idx]
            if self.layer_idx < len(self.rope_layer_enabled_list)
            else True
        )  # Default to True if index out of bounds

        self._attention_interface = eager_attention_forward

    def call(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        training=False,
        **kwargs,
    ):
        self.training = training

        input_shape = ops.shape(hidden_states)[
            :-1
        ]  # Exclude last dim (hidden_size)

        hidden_shape = (*input_shape, self.num_attention_heads, self.head_dim)

        query_states = ops.reshape(self.q_proj(hidden_states), hidden_shape)
        query_states = ops.transpose(
            query_states, axes=(0, 2, 1, 3)
        )  # (batch, num_heads, seq_len, head_dim)

        # For key and value, the kv_hidden_shape should be based on num_key_value_heads
        kv_hidden_shape = (
            *input_shape,
            self.num_key_value_heads,
            self.head_dim,
        )
        key_states = ops.reshape(self.k_proj(hidden_states), kv_hidden_shape)
        key_states = ops.transpose(
            key_states, axes=(0, 2, 1, 3)
        )  # (batch, num_key_value_heads, seq_len, head_dim)

        value_states = ops.reshape(self.v_proj(hidden_states), kv_hidden_shape)
        value_states = ops.transpose(
            value_states, axes=(0, 2, 1, 3)
        )  # (batch, num_key_value_heads, seq_len, head_dim)

        if self.use_rope:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        attn_output, attn_weights = self._attention_interface(
            module=self,
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
            dropout=self.attention_dropout,
            scaling=self.scaling,
            training=self.training,
            **kwargs,
        )

        attn_output = ops.reshape(attn_output, (*input_shape, self.hidden_size))

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class SmolLM3MLP(layers.Layer):
    def __init__(
        self, hidden_size: int, intermediate_size: int, mlp_bias: bool, **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.mlp_bias = mlp_bias

        self.gate_proj = layers.Dense(
            self.intermediate_size, use_bias=self.mlp_bias, name="gate_proj"
        )
        self.up_proj = layers.Dense(
            self.intermediate_size, use_bias=self.mlp_bias, name="up_proj"
        )
        self.down_proj = layers.Dense(
            self.hidden_size, use_bias=self.mlp_bias, name="down_proj"
        )

    def call(self, x):
        gate_output = activations.silu(self.gate_proj(x))
        up_output = self.up_proj(x)
        intermediate_output = gate_output * up_output
        down_proj_output = self.down_proj(intermediate_output)
        return down_proj_output


class SmolLM3DecoderLayer(layers.Layer):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_bias: bool,
        attention_dropout: float,
        rope_layer_enabled_list: list[bool],
        layer_types: list[str],
        layer_idx: int,
        intermediate_size: int,
        mlp_bias: bool,
        rms_norm_eps: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx

        self.self_attn = SmolLM3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            rope_layer_enabled_list=rope_layer_enabled_list,
            layer_types=layer_types,
            layer_idx=layer_idx,
            name="self_attn",
        )

        self.mlp = SmolLM3MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            mlp_bias=mlp_bias,
            name="mlp",
        )

        self.input_layernorm = layers.RMSNormalization(
            epsilon=rms_norm_eps, axis=-1, name="input_layernorm"
        )
        self.post_attention_layernorm = layers.RMSNormalization(
            epsilon=rms_norm_eps, axis=-1, name="post_attention_layernorm"
        )

        self.attention_type = layer_types[layer_idx]

    def build(self, input_shape):
        # Build sub-layers
        self.self_attn.build(input_shape)
        self.mlp.build(input_shape)
        self.input_layernorm.build(input_shape)
        self.post_attention_layernorm.build(input_shape)

        super().build(input_shape)

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        training=False,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_output, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            training=training,
            **kwargs,
        )
        hidden_states = ops.add(residual, attn_output)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = ops.add(residual, hidden_states)

        return hidden_states


class SmolLM3RotaryEmbedding(layers.Layer):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        max_position_embeddings: int,
        rope_theta: float,
        partial_rotary_factor: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor

        self.head_dim = self.hidden_size // self.num_attention_heads

        inv_freq_tensor, self.attention_scaling = rope_init(
            self.rope_theta, self.partial_rotary_factor, self.head_dim
        )

        self.inv_freq = self.add_weight(
            name="inv_freq",
            shape=ops.shape(inv_freq_tensor),
            dtype=inv_freq_tensor.dtype,
            initializer=initializers.Constant(
                ops.convert_to_numpy(inv_freq_tensor)
            ),
            trainable=False,  # This weight is not trained
        )
        self.original_inv_freq = self.inv_freq

    def call(self, x, position_ids):
        inv_freq_expanded = ops.expand_dims(
            ops.expand_dims(self.inv_freq, axis=0), axis=-1
        )

        batch_size = ops.shape(position_ids)[0]
        inv_freq_expanded = ops.broadcast_to(
            inv_freq_expanded, (batch_size, ops.shape(self.inv_freq)[0], 1)
        )

        position_ids_expanded = ops.expand_dims(position_ids, axis=1)

        freqs = ops.matmul(
            ops.cast(inv_freq_expanded, "float32"),
            ops.cast(position_ids_expanded, "float32"),
        )

        freqs = ops.transpose(freqs, axes=(0, 2, 1))

        emb = ops.concatenate((freqs, freqs), axis=-1)

        cos = ops.cos(emb) * self.attention_scaling
        sin = ops.sin(emb) * self.attention_scaling

        return ops.cast(cos, x.dtype), ops.cast(sin, x.dtype)
