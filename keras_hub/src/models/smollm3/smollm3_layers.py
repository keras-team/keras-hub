import math

from keras import activations
from keras import initializers
from keras import layers
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.smollm3.smollm3_utils import apply_rotary_pos_emb
from keras_hub.src.models.smollm3.smollm3_utils import rope_init


class SmolLM3Attention(layers.Layer):
    """Multi-head attention layer for SmolLM3 model.

    Args:
        hidden_size: int. The hidden size of the attention layer.
        num_attention_heads: int. The number of attention heads.
        num_key_value_heads: int. The number of key-value heads.
        attention_bias: bool. Whether to use bias in attention projections.
        attention_dropout: float. Dropout rate for attention weights.
        rope_layer_enabled_list: list of bool. List indicating if RoPE is
            enabled for each layer.
        layer_types: list of str. List of layer types.
        layer_idx: int. Index of the current layer.
        max_position_embeddings: int. Maximum sequence length for position
            embeddings. Defaults to 2048.
        rope_theta: float. The theta value for RoPE. Defaults to 10000.0.
        partial_rotary_factor: float. The factor for partial rotary embedding.
            Defaults to 1.0.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        attention_bias,
        attention_dropout,
        rope_layer_enabled_list,
        layer_types,
        layer_idx,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        partial_rotary_factor=1.0,
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
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.partial_rotary_factor = partial_rotary_factor

        self._dot_product_equation = "bquh,bkuh->buqk"
        self._combine_equation = "buqk,bkuh->bquh"

        self.head_dim = hidden_size // self.num_attention_heads
        self._inv_norm_factor = 1.0 / math.sqrt(self.head_dim)

        self.layer_idx = layer_idx
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
        self.o_proj = layers.EinsumDense(
            equation="bquh,uhm->bqm",
            output_shape=(None, self.hidden_size),
            name="o_proj",
        )
        self.o_proj.build((None, None, self.num_attention_heads, self.head_dim))

        self.use_rope = (
            self.rope_layer_enabled_list[self.layer_idx]
            if self.layer_idx < len(self.rope_layer_enabled_list)
            else True
        )  # Default to True if index out of bounds

        self.rotary_embedding = SmolLM3RotaryEmbedding(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
            rope_theta=self.rope_theta,
            partial_rotary_factor=self.partial_rotary_factor,
            name="rotary_emb",
        )

        self._softmax = layers.Softmax(
            axis=-1,
            dtype="float32",
            name="attention_softmax",
        )

    def build(self, input_shape):
        """Builds the internal Dense layers.

        Args:
            input_shape: A list/tuple of shapes for the inputs:
                [hidden_states_shape, position_embeddings_shape_tuple,
                attention_mask_shape]
                - hidden_states_shape: (batch_size, seq_len,
                    hidden_size)
        """
        # The input shape to the Dense layers (q_proj, k_proj, v_proj, o_proj)
        # is the same as the hidden_states input to SmolLM3Attention.
        hidden_states_shape = input_shape[0]
        self.q_proj.build(hidden_states_shape)
        self.k_proj.build(hidden_states_shape)
        self.v_proj.build(hidden_states_shape)
        super().build(input_shape)

    def call(
        self,
        hidden_states,
        training=False,
        attention_mask=None,
        **kwargs,
    ):
        """Forward pass for SmolLM3Attention.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len,
                hidden_size).
            position_embeddings: Tuple of (cos, sin) tensors for RoPE.
            attention_mask: Attention mask tensor.
            training: Whether the layer is in training mode.
        """
        self.training = training
        self_attention_cache = kwargs.get("self_attention_cache", None)
        self_attention_cache_update_index = kwargs.get(
            "self_attention_cache_update_index", None
        )
        start_index = (
            self_attention_cache_update_index
            if self_attention_cache_update_index is not None
            else 0
        )

        input_shape = ops.shape(hidden_states)[:-1]
        hidden_shape = (*input_shape, self.num_attention_heads, self.head_dim)

        query = ops.reshape(self.q_proj(hidden_states), hidden_shape)

        def _compute_kv_values(x_input):
            kv_hidden_shape = (
                *input_shape,
                self.num_key_value_heads,
                self.head_dim,
            )

            key = ops.reshape(self.k_proj(x_input), kv_hidden_shape)
            value = ops.reshape(self.v_proj(x_input), kv_hidden_shape)

            return key, value

        if self_attention_cache is not None:
            key_cache = self_attention_cache[:, 0, ...]
            value_cache = self_attention_cache[:, 1, ...]

            if self_attention_cache_update_index is None:
                key = key_cache
                value = value_cache
            else:
                key_update, value_update = _compute_kv_values(hidden_states)

                # Apply RoPE to key_update BEFORE caching
                if self.use_rope:
                    cos, sin = self.rotary_embedding(
                        query, start_index=start_index
                    )
                    query_rope, key_update = apply_rotary_pos_emb(
                        query, key_update, cos, sin, expansion_axis=2
                    )
                    query = query_rope

                start = (0, self_attention_cache_update_index, 0, 0)

                key = ops.slice_update(key_cache, start, key_update)
                value = ops.slice_update(value_cache, start, value_update)
                self_attention_cache = ops.stack((key, value), axis=1)
        else:
            if self_attention_cache_update_index is not None:
                raise ValueError(
                    "`self_attention_cache_update_index` should not be set "
                    "if `self_attention_cache` is `None`. Received: "
                    f"self_attention_cache={self_attention_cache}, "
                    "self_attention_cache_update_index="
                    f"{self_attention_cache_update_index}"
                )
            key, value = _compute_kv_values(hidden_states)

            # Apply RoPE when not using cache
            if self.use_rope:
                cos, sin = self.rotary_embedding(query, start_index=start_index)
                query, key = apply_rotary_pos_emb(
                    query, key, cos, sin, expansion_axis=2
                )

        key = ops.repeat(key, repeats=self.num_key_value_groups, axis=2)
        value = ops.repeat(value, repeats=self.num_key_value_groups, axis=2)

        attn_output = self._compute_attention(
            query,
            key,
            value,
            attention_mask,
            cache_update_index=self_attention_cache_update_index,
        )

        attn_output = self.o_proj(attn_output)

        if self_attention_cache is not None:
            return attn_output, self_attention_cache

        return attn_output

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Args:
            input_shape: A list/tuple of shapes for the inputs:
                [hidden_states_shape, position_embeddings_shape_tuple,
                attention_mask_shape]
                - hidden_states_shape: (batch_size, seq_len,
                  hidden_size)
                - position_embeddings_shape_tuple: (cos_shape,
                  sin_shape) where cos_shape/sin_shape is
                  (batch_size, seq_len, head_dim)
                - attention_mask_shape: (batch_size, 1, seq_len,
                  seq_len)

        Returns:
            A list of output shapes: [output_attn_output_shape,
            output_attn_weights_shape]
        """
        hidden_states_shape = input_shape[0]

        batch_size = hidden_states_shape[0]
        seq_len = hidden_states_shape[1]

        output_attn_output_shape = (batch_size, seq_len, self.hidden_size)

        output_attn_weights_shape = (
            batch_size,
            self.num_attention_heads,
            seq_len,
            seq_len,
        )

        return [output_attn_output_shape, output_attn_weights_shape]

    def _masked_softmax(self, attention_scores, attention_mask=None):
        """Applies softmax with optional masking.

        Args:
            attention_scores: Attention score tensor.
            attention_mask: Optional mask tensor.

        Returns:
            Masked softmax attention weights.
        """
        if attention_mask is not None:
            return self._softmax(
                attention_scores, attention_mask[:, None, :, :]
            )
        return self._softmax(attention_scores)

    def _compute_attention(
        self, query, key, value, attention_mask=None, cache_update_index=None
    ):
        """Computes attention using query, key, and value tensors.

        Uses Flash Attention when available for better performance.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            attention_mask: Optional mask tensor.
            cache_update_index: Index for sliding window computation.

        Returns:
            attention_output: Output tensor after applying attention.
        """
        attention_scores = ops.einsum(self._dot_product_equation, query, key)

        attention_scores = ops.multiply(
            attention_scores,
            ops.cast(self._inv_norm_factor, self.compute_dtype),
        )
        attention_scores = self._masked_softmax(
            attention_scores, attention_mask
        )
        attention_scores = ops.cast(attention_scores, self.compute_dtype)
        attention_output = ops.einsum(
            self._combine_equation, attention_scores, value
        )

        return attention_output


class SmolLM3MLP(layers.Layer):
    """Multi-layer perceptron (MLP) block for SmolLM3 model.

    Args:
        hidden_size: int. The hidden size of the MLP.
        intermediate_size: int. The intermediate size of the MLP.
        mlp_bias: bool. Whether to use bias in MLP dense layers.
    """

    def __init__(self, hidden_size, intermediate_size, mlp_bias, **kwargs):
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

    def build(self, input_shape):
        """
        Builds the internal Dense layers.
        Args:
            input_shape: The shape of the input to this layer
                         (batch_size, seq_len, hidden_size).
        """
        self.gate_proj.build(input_shape)
        self.up_proj.build(input_shape)
        # The down_proj takes intermediate_output, which has shape
        # (batch_size, seq_len, intermediate_size)
        down_proj_input_shape = (
            input_shape[0],
            input_shape[1],
            self.intermediate_size,
        )
        self.down_proj.build(down_proj_input_shape)
        super().build(input_shape)

    def call(self, x):
        """
        Forward pass for SmolLM3MLP.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).
        """
        gate_output = activations.silu(self.gate_proj(x))
        up_output = self.up_proj(x)
        intermediate_output = gate_output * up_output
        down_proj_output = self.down_proj(intermediate_output)
        return down_proj_output

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Args:
            input_shape: The input shape (batch_size, seq_len, hidden_size).

        Returns:
            The output shape, which is the same as the input shape:
            (batch_size, seq_len, hidden_size).
        """
        return input_shape


class SmolLM3DecoderLayer(layers.Layer):
    """Decoder layer for SmolLM3 model, combining self-attention and MLP.

    Args:
        hidden_size: int. The hidden size of the layer.
        num_attention_heads: int. The number of attention heads.
        num_key_value_heads: int. The number of key-value heads.
        attention_bias: bool. Whether to use bias in attention projections.
        attention_dropout: float. Dropout rate for attention weights.
        rope_layer_enabled_list: list of bool. List indicating if RoPE is
            enabled for each layer.
        layer_types: list of str. List of layer types.
        layer_idx: int. Index of the current layer.
        intermediate_size: int. The intermediate size of the MLP.
        mlp_bias: bool. Whether to use bias in MLP dense layers.
        layer_norm_epsilon: float. Epsilon for RMSNormalization.
        max_position_embeddings: int. Maximum sequence length for position
            embeddings. Defaults to 2048.
        rope_theta: float. The theta value for RoPE. Defaults to 10000.0.
        partial_rotary_factor: float. The factor for partial rotary embedding.
            Defaults to 1.0.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        attention_bias,
        attention_dropout,
        rope_layer_enabled_list,
        layer_types,
        layer_idx,
        intermediate_size,
        mlp_bias,
        layer_norm_epsilon,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        partial_rotary_factor=1.0,
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
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            name="self_attn",
        )

        self.mlp = SmolLM3MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            mlp_bias=mlp_bias,
            name="mlp",
        )

        self.input_layernorm = layers.RMSNormalization(
            epsilon=layer_norm_epsilon, axis=-1, name="input_layernorm"
        )
        self.post_attention_layernorm = layers.RMSNormalization(
            epsilon=layer_norm_epsilon, axis=-1, name="post_attention_layernorm"
        )

        self.attention_type = layer_types[layer_idx]

    def _compute_self_attention_mask(
        self,
        decoder_sequence,
        decoder_padding_mask,
        decoder_attention_mask,
        self_attention_cache,
        self_attention_cache_update_index,
    ):
        decoder_mask = merge_padding_and_attention_mask(
            decoder_sequence, decoder_padding_mask, decoder_attention_mask
        )
        batch_size = ops.shape(decoder_sequence)[0]
        input_length = output_length = ops.shape(decoder_sequence)[1]
        # We need to handle a rectangular causal mask when doing cached
        # decoding. For generative inference, `decoder_sequence` will
        # generally be length 1, and `cache` will be the full generation length.
        if self_attention_cache is not None:
            input_length = ops.shape(self_attention_cache)[2]

        cache_update_index = (
            0
            if self_attention_cache_update_index is None
            else self_attention_cache_update_index
        )

        causal_mask = compute_causal_mask(
            batch_size, input_length, output_length, cache_update_index
        )

        return (
            ops.minimum(decoder_mask, causal_mask)
            if decoder_mask is not None
            else causal_mask
        )

    def build(self, input_shape):
        """
        Builds the sub-layers based on the input shape.

        Args:
            input_shape: The input shape to the decoder layer
                (batch_size, seq_len, hidden_size).
        """
        # input_shape for SmolLM3DecoderLayer: (batch_size, seq_len,
        # hidden_size)
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        head_dim = self.self_attn.head_dim
        pos_emb_shape = (batch_size, seq_len, head_dim)

        attn_mask_shape = (batch_size, 1, seq_len, seq_len)

        # Pass the correct input shape to self_attn's build method
        # The input_shape for self_attn.build is a list:
        # [hidden_states_shape, (pos_emb_shape, pos_emb_shape), attn_mask_shape]
        self.self_attn.build(
            [input_shape, (pos_emb_shape, pos_emb_shape), attn_mask_shape]
        )

        self.mlp.build(input_shape)
        self.input_layernorm.build(input_shape)
        self.post_attention_layernorm.build(input_shape)

        super().build(input_shape)

    def call(
        self,
        hidden_states,
        training=False,
        decoder_padding_mask=None,
        decoder_attention_mask=None,
        **kwargs,
    ):
        """
        Forward pass for SmolLM3DecoderLayer.

        Args:
            hidden_states: Input tensor of shape (batch_size,
                seq_len, hidden_size).
            position_embeddings: Optional tuple of (cos, sin)
                tensors for RoPE.
            training: Whether the layer is in training mode.
        """
        self_attention_cache = kwargs.get("self_attention_cache", None)
        self_attention_cache_update_index = kwargs.get(
            "self_attention_cache_update_index", None
        )

        self_attention_mask = self._compute_self_attention_mask(
            decoder_sequence=hidden_states,
            decoder_padding_mask=decoder_padding_mask,
            decoder_attention_mask=decoder_attention_mask,
            self_attention_cache=self_attention_cache,
            self_attention_cache_update_index=self_attention_cache_update_index,
        )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        x = self.self_attn(
            hidden_states=hidden_states,
            training=training,
            attention_mask=self_attention_mask,
            **kwargs,
        )

        if isinstance(x, tuple):
            attn_output, self_attention_cache = x
        else:
            attn_output = x

        hidden_states = ops.add(residual, attn_output)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = ops.add(residual, hidden_states)

        if self_attention_cache is not None:
            return hidden_states, self_attention_cache
        else:
            return hidden_states

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Args:
            input_shape: The input shape (batch_size, seq_len, hidden_size).

        Returns:
            The output shape, which is the same as the input shape:
            (batch_size, seq_len, hidden_size).
        """
        return input_shape


class SmolLM3RotaryEmbedding(layers.Layer):
    """Rotary Position Embedding (RoPE) layer for SmolLM3 model.

    Args:
        hidden_size: int. The hidden size of the model.
        num_attention_heads: int. The number of attention heads.
        max_position_embeddings: int. The maximum sequence length for position
            embeddings.
        rope_theta: float. The theta value for RoPE.
        partial_rotary_factor: float. The factor for partial rotary embedding.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        max_position_embeddings,
        rope_theta,
        partial_rotary_factor,
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

    def build(self, input_shape):
        """
        Builds the layer. For SmolLM3RotaryEmbedding, this mainly
        ensures that the parent layer's build is called.

        Args:
            input_shape: A list/tuple of shapes for the inputs:
                [x_shape, position_ids_shape]
                - x_shape: (batch_size, ..., head_dim)
                - position_ids_shape: (batch_size, seq_len)
        """
        # No internal layers to explicitly build here, as inv_freq is
        # added in __init__
        super().build(input_shape)

    def call(
        self,
        x,
        start_index=0,
    ):
        """
        Forward pass for SmolLM3RotaryEmbedding.

        Args:
            x: Input tensor, typically query or key states.
               Shape can vary, but the last dimension is head_dim.
            position_ids: Tensor of position IDs of shape (batch_size, seq_len).
        """
        batch_size = ops.shape(x)[0]
        seq_len = ops.shape(x)[1]
        positions = ops.arange(seq_len, dtype="float32")
        positions = positions + ops.cast(start_index, dtype="float32")

        # inv_freq: (inv_freq_dim,) -> (1, inv_freq_dim, 1)
        # -> (batch, inv_freq_dim, 1)
        inv_freq_expanded = ops.expand_dims(
            ops.expand_dims(self.inv_freq, axis=0), axis=-1
        )
        inv_freq_expanded = ops.broadcast_to(
            inv_freq_expanded, (batch_size, ops.shape(self.inv_freq)[0], 1)
        )

        # positions: (seq_len,) -> (1, 1, seq_len)
        # -> (batch, 1, seq_len)
        position_ids_expanded = ops.expand_dims(
            ops.expand_dims(positions, axis=0), axis=0
        )
        position_ids_expanded = ops.broadcast_to(
            position_ids_expanded, (batch_size, 1, seq_len)
        )

        # matmul: (batch, inv_freq_dim, 1) @ (batch, 1, seq_len)
        # -> (batch, inv_freq_dim, seq_len)
        freqs = ops.matmul(
            ops.cast(inv_freq_expanded, "float32"),
            ops.cast(position_ids_expanded, "float32"),
        )

        # transpose: (batch, inv_freq_dim, seq_len) ->
        # (batch, seq_len, inv_freq_dim)
        freqs = ops.transpose(freqs, axes=(0, 2, 1))

        emb = ops.concatenate((freqs, freqs), axis=-1)

        cos = ops.cos(emb) * self.attention_scaling
        sin = ops.sin(emb) * self.attention_scaling

        return ops.cast(cos, x.dtype), ops.cast(sin, x.dtype)

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Args:
            input_shape: A list/tuple of shapes for the inputs:
                         [x_shape, position_ids_shape]
                         - x_shape: (batch_size, ..., head_dim)
                         - position_ids_shape: (batch_size, seq_len)

        Returns:
            A list of output shapes for (cos, sin):
            [(batch_size, seq_len, head_dim), (batch_size, seq_len, head_dim)]
        """
        if input_shape[1] is not None and len(input_shape[1]) >= 2:
            batch_size = input_shape[1][0]
            seq_len = input_shape[1][1]
        else:
            # Fallback if position_ids_shape is None or malformed.
            # In this case, the batch_size and seq_len are unknown.
            batch_size = None
            seq_len = None

        # The output cos and sin have shape (batch_size, seq_len, head_dim)
        output_shape = (batch_size, seq_len, self.head_dim)

        return [output_shape, output_shape]
