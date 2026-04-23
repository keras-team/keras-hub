import math

import keras
from keras import activations
from keras import layers
from keras import ops

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)


class SmolVLM2InterleaveEmbeddings(keras.layers.Layer):
    """Scatter vision token embeddings into the text embedding sequence.

    Given a (batch, seq_len, hidden_dim) text embedding tensor and a flat
    list of vision token embeddings, this layer replaces positions indicated
    by `vision_indices` with the corresponding vision embeddings.

    This follows the same pattern as `Qwen3_5InterleaveEmbeddings` and
    replaces the static `_inputs_merger` method previously in the CausalLM.

    Args:
        hidden_dim: int. The embedding dimension (must match both text and
            vision embeddings after projection).
    """

    def __init__(self, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def call(self, image_embeddings, text_embeddings, vision_indices):
        """Interleave vision tokens into the text embedding sequence.

        Args:
            image_embeddings: Tensor with vision token embeddings.
                Batched: (batch, total_vision_tokens, hidden_dim).
                Unbatched: (total_vision_tokens, hidden_dim).
            text_embeddings: Tensor (batch, seq_len, hidden_dim).
            vision_indices: int32 Tensor with flat indices into the
                concatenated (batch × seq_len) sequence.

        Returns:
            Tensor (batch, seq_len, hidden_dim) with vision tokens
            inserted at the specified positions.
        """
        batch_size = ops.shape(text_embeddings)[0]
        seq_len = ops.shape(text_embeddings)[1]

        # Handle batched image_embeddings from the functional graph.
        if len(ops.shape(image_embeddings)) == 3:
            image_embeddings = ops.reshape(
                image_embeddings, (-1, self.hidden_dim)
            )
        if len(ops.shape(vision_indices)) == 2:
            vision_indices = ops.reshape(vision_indices, (-1,))

        # Slice image_embeddings to match the number of vision indices.
        # This handles text-only calls where the vision encoder
        # produces embeddings but there are 0 vision positions.
        num_indices = ops.shape(vision_indices)[0]
        image_embeddings = image_embeddings[:num_indices]

        image_embeddings = ops.cast(image_embeddings, text_embeddings.dtype)

        # Flatten text to (batch * seq_len, hidden_dim).
        flat_text = ops.reshape(text_embeddings, (-1, self.hidden_dim))

        # Reshape indices for scatter_update.
        vision_indices = ops.cast(vision_indices, "int32")
        vision_indices = ops.expand_dims(vision_indices, axis=-1)

        # Scatter vision embeddings into the flat text tensor.
        flat_out = ops.scatter_update(
            flat_text, vision_indices, image_embeddings
        )

        return ops.reshape(flat_out, (batch_size, seq_len, self.hidden_dim))

    def compute_output_spec(
        self, image_embeddings, text_embeddings, vision_indices
    ):
        """Return output shape spec for functional model tracing."""
        return keras.KerasTensor(
            shape=text_embeddings.shape,
            dtype=text_embeddings.dtype,
        )

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim})
        return config


class SmolVLM2Attention(layers.Layer):
    """Multi-head attention layer for SmolVLM2 text decoder.

    This implements standard Llama-style attention with grouped query
    attention (GQA) and rotary position embeddings (RoPE).

    Args:
        hidden_dim: int. Dimensionality of the model hidden states.
        num_query_heads: int. Number of query attention heads.
        num_key_value_heads: int. Number of key/value attention heads.
        rope_max_wavelength: float. Maximum wavelength for RoPE.
        layer_norm_epsilon: float. Epsilon for RMS normalization.
        dropout: float. Dropout probability for attention weights.
        dtype: string or keras DTypePolicy. Computation/weight dtype.
    """

    def __init__(
        self,
        hidden_dim,
        num_query_heads,
        num_key_value_heads,
        rope_max_wavelength=10000,
        dropout=0,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_dim = hidden_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.rope_max_wavelength = rope_max_wavelength
        self.dropout = dropout

        self.head_dim = hidden_dim // num_query_heads
        self.num_key_value_groups = num_query_heads // num_key_value_heads
        self._inv_norm_factor = 1.0 / math.sqrt(self.head_dim)

        self._dot_product_equation = "bquh,bkuh->buqk"
        self._combine_equation = "buqk,bkuh->bquh"

    def build(self, input_shape):
        self.q_proj = layers.Dense(
            self.num_query_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype_policy,
            name="q_proj",
        )
        self.q_proj.build(input_shape)

        self.k_proj = layers.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype_policy,
            name="k_proj",
        )
        self.k_proj.build(input_shape)

        self.v_proj = layers.Dense(
            self.num_key_value_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype_policy,
            name="v_proj",
        )
        self.v_proj.build(input_shape)

        self.o_proj = layers.Dense(
            self.hidden_dim,
            use_bias=False,
            dtype=self.dtype_policy,
            name="o_proj",
        )
        self.o_proj.build((None, None, self.num_query_heads * self.head_dim))

        self.rotary_embedding = RotaryEmbedding(
            max_wavelength=self.rope_max_wavelength,
            dtype=self.dtype_policy,
            name="rotary_embedding",
        )

        self._softmax = layers.Softmax(
            axis=-1,
            dtype="float32",
            name="attention_softmax",
        )

        super().build(input_shape)

    def _compute_attention(self, query, key, value, attention_mask=None):
        attention_scores = ops.einsum(self._dot_product_equation, query, key)
        attention_scores = ops.multiply(
            attention_scores,
            ops.cast(self._inv_norm_factor, self.compute_dtype),
        )

        if attention_mask is not None:
            attention_scores = self._softmax(
                attention_scores, attention_mask[:, None, :, :]
            )
        else:
            attention_scores = self._softmax(attention_scores)

        attention_scores = ops.cast(attention_scores, self.compute_dtype)
        attention_output = ops.einsum(
            self._combine_equation, attention_scores, value
        )
        return attention_output

    def call(
        self,
        hidden_states,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
    ):
        batch_size = ops.shape(hidden_states)[0]
        seq_len = ops.shape(hidden_states)[1]

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # Reshape to (batch, seq, num_heads, head_dim)
        query = ops.reshape(
            query, (batch_size, seq_len, self.num_query_heads, self.head_dim)
        )
        key = ops.reshape(
            key,
            (batch_size, seq_len, self.num_key_value_heads, self.head_dim),
        )
        value = ops.reshape(
            value,
            (batch_size, seq_len, self.num_key_value_heads, self.head_dim),
        )

        # Apply RoPE.
        start_index = (
            cache_update_index if cache_update_index is not None else 0
        )
        query = self.rotary_embedding(query, start_index=start_index)
        key = self.rotary_embedding(key, start_index=start_index)

        # Handle KV cache.
        if cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]

            if cache_update_index is not None:
                start = (0, cache_update_index, 0, 0)
                key = ops.slice_update(key_cache, start, key)
                value = ops.slice_update(value_cache, start, value)
            else:
                key = key_cache
                value = value_cache

            cache = ops.stack((key, value), axis=1)

        # Repeat KV heads for GQA.
        if self.num_key_value_groups > 1:
            key = ops.repeat(key, repeats=self.num_key_value_groups, axis=2)
            value = ops.repeat(value, repeats=self.num_key_value_groups, axis=2)

        attn_output = self._compute_attention(query, key, value, attention_mask)

        # Reshape back to original shape
        attn_output = ops.reshape(
            attn_output, (batch_size, seq_len, self.hidden_dim)
        )
        attn_output = self.o_proj(attn_output)

        if cache is not None:
            return attn_output, cache
        return attn_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "rope_max_wavelength": self.rope_max_wavelength,
                "dropout": self.dropout,
            }
        )
        return config


class SmolVLM2MLP(layers.Layer):
    """SwiGLU feedforward block for SmolVLM2 text decoder.

    Matches HF's LlamaMLP: gate_proj/up_proj → silu(gate) * up → down_proj.

    Args:
        hidden_dim: int. Input/output dimensionality.
        intermediate_dim: int. Inner dimensionality.
        dtype: string or keras DTypePolicy. Computation/weight dtype.
    """

    def __init__(self, hidden_dim, intermediate_dim, dtype=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim

    def build(self, input_shape):
        self.gate_proj = layers.Dense(
            self.intermediate_dim,
            use_bias=False,
            dtype=self.dtype_policy,
            name="gate_proj",
        )
        self.gate_proj.build(input_shape)

        self.up_proj = layers.Dense(
            self.intermediate_dim,
            use_bias=False,
            dtype=self.dtype_policy,
            name="up_proj",
        )
        self.up_proj.build(input_shape)

        self.down_proj = layers.Dense(
            self.hidden_dim,
            use_bias=False,
            dtype=self.dtype_policy,
            name="down_proj",
        )
        self.down_proj.build(
            (input_shape[0], input_shape[1], self.intermediate_dim)
        )

        super().build(input_shape)

    def call(self, x):
        gate_output = activations.silu(self.gate_proj(x))
        up_output = self.up_proj(x)
        return self.down_proj(gate_output * up_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
            }
        )
        return config


class SmolVLM2DecoderBlock(layers.Layer):
    """Pre-norm decoder block for SmolVLM2 text decoder.

    Combines self-attention and MLP with RMSNorm pre-normalization,
    matching HF's LlamaDecoderLayer.

    Args:
        hidden_dim: int. Dimensionality of the model hidden states.
        intermediate_dim: int. Inner dimensionality of the MLP.
        num_query_heads: int. Number of query attention heads.
        num_key_value_heads: int. Number of key/value attention heads.
        rope_max_wavelength: float. Maximum wavelength for RoPE.
        layer_norm_epsilon: float. Epsilon for RMS normalization.
        dropout: float. Dropout probability.
        dtype: string or keras DTypePolicy. Computation/weight dtype.
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_query_heads,
        num_key_value_heads,
        rope_max_wavelength=10000,
        layer_norm_epsilon=1e-5,
        dropout=0,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.rope_max_wavelength = rope_max_wavelength
        self.layer_norm_epsilon = layer_norm_epsilon
        self.dropout = dropout

    def build(self, input_shape):
        self.self_attn = SmolVLM2Attention(
            hidden_dim=self.hidden_dim,
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            rope_max_wavelength=self.rope_max_wavelength,
            dropout=self.dropout,
            dtype=self.dtype_policy,
            name="self_attn",
        )
        self.self_attn.build(input_shape)

        self.mlp = SmolVLM2MLP(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            dtype=self.dtype_policy,
            name="mlp",
        )
        self.mlp.build(input_shape)

        self.input_layernorm = layers.RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="input_layernorm",
        )
        self.input_layernorm.build(input_shape)

        self.post_attention_layernorm = layers.RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="post_attention_layernorm",
        )
        self.post_attention_layernorm.build(input_shape)

        super().build(input_shape)

    def _compute_self_attention_mask(
        self,
        decoder_sequence,
        decoder_padding_mask,
        decoder_attention_mask,
        cache,
        cache_update_index,
    ):
        decoder_mask = merge_padding_and_attention_mask(
            decoder_sequence, decoder_padding_mask, decoder_attention_mask
        )
        batch_size = ops.shape(decoder_sequence)[0]
        input_length = output_length = ops.shape(decoder_sequence)[1]

        if cache is not None:
            input_length = ops.shape(cache)[2]

        cache_idx = 0 if cache_update_index is None else cache_update_index

        causal_mask = compute_causal_mask(
            batch_size, input_length, output_length, cache_idx
        )

        return (
            ops.minimum(decoder_mask, causal_mask)
            if decoder_mask is not None
            else causal_mask
        )

    def call(
        self,
        hidden_states,
        decoder_padding_mask=None,
        decoder_attention_mask=None,
        cache=None,
        cache_update_index=None,
    ):
        self_attention_mask = self._compute_self_attention_mask(
            decoder_sequence=hidden_states,
            decoder_padding_mask=decoder_padding_mask,
            decoder_attention_mask=decoder_attention_mask,
            cache=cache,
            cache_update_index=cache_update_index,
        )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        x = self.self_attn(
            hidden_states,
            attention_mask=self_attention_mask,
            cache=cache,
            cache_update_index=cache_update_index,
        )

        if isinstance(x, tuple):
            attn_output, cache = x
        else:
            attn_output = x
            cache = None

        hidden_states = ops.add(residual, attn_output)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = ops.add(residual, hidden_states)

        if cache is not None:
            return hidden_states, cache
        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "rope_max_wavelength": self.rope_max_wavelength,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
            }
        )
        return config


class SmolVLM2Connector(layers.Layer):
    """Idefics3-style connector: pixel-shuffle + linear projection.

    Reduces the number of visual tokens by `scale_factor²` via a
    space-to-depth (pixel-shuffle) rearrangement, then projects the
    concatenated features to the text decoder's hidden dimension.

    Matches HF's `Idefics3Connector`.

    Args:
        vision_hidden_dim: int. Vision encoder output dimension.
        text_hidden_dim: int. Text decoder hidden dimension.
        scale_factor: int. Spatial downsampling factor.
        dtype: string or keras DTypePolicy. Computation/weight dtype.
    """

    def __init__(
        self,
        vision_hidden_dim,
        text_hidden_dim,
        scale_factor,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.vision_hidden_dim = vision_hidden_dim
        self.text_hidden_dim = text_hidden_dim
        self.scale_factor = scale_factor

    def build(self, input_shape):
        projection_input_dim = self.vision_hidden_dim * self.scale_factor**2
        self.modality_projection = layers.Dense(
            self.text_hidden_dim,
            use_bias=False,
            dtype=self.dtype_policy,
            name="modality_projection",
        )
        self.modality_projection.build((None, None, projection_input_dim))
        super().build(input_shape)

    def pixel_shuffle(self, x, scale_factor):
        """Rearrange spatial tokens via space-to-depth.

        Converts (batch, H*W, D) -> (batch, H*W/s², D*s²) where
        s = scale_factor. Assumes input sequence length is a perfect
        square (H = W = sqrt(seq_len)).

        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim).
            scale_factor: int. Spatial downsampling factor.

        Returns:
            Rearranged tensor of shape
            (batch, seq_len / scale_factor², embed_dim * scale_factor²).
        """
        bsz = ops.shape(x)[0]
        # Use static shape for seq_len and embed_dim — these are known
        # at graph-build time and avoid JAX tracer issues.
        seq_len = x.shape[1]
        embed_dim = x.shape[2]

        height = width = int(seq_len**0.5)

        x = ops.reshape(x, (bsz, height, width, embed_dim))
        x = ops.reshape(
            x,
            (
                bsz,
                height,
                width // scale_factor,
                embed_dim * scale_factor,
            ),
        )
        x = ops.transpose(x, axes=(0, 2, 1, 3))
        x = ops.reshape(
            x,
            (
                bsz,
                width // scale_factor,
                height // scale_factor,
                embed_dim * scale_factor**2,
            ),
        )
        x = ops.transpose(x, axes=(0, 2, 1, 3))
        new_seq_len = seq_len // (scale_factor**2)
        x = ops.reshape(x, (bsz, new_seq_len, embed_dim * scale_factor**2))
        return x

    def call(self, image_hidden_states):
        """Forward pass of the connector.

        Args:
            image_hidden_states: Tensor of shape
                (batch, num_patches, vision_hidden_dim) from the vision
                encoder.

        Returns:
            Projected tensor of shape
            (batch, num_patches / scale_factor², text_hidden_dim).
        """
        image_hidden_states = self.pixel_shuffle(
            image_hidden_states, self.scale_factor
        )
        image_hidden_states = self.modality_projection(image_hidden_states)
        return image_hidden_states

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vision_hidden_dim": self.vision_hidden_dim,
                "text_hidden_dim": self.text_hidden_dim,
                "scale_factor": self.scale_factor,
            }
        )
        return config
