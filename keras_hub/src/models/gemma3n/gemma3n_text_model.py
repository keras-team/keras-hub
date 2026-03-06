import math

import keras

from keras_hub.src.models.gemma3n.gemma3n_text_decoder import (
    Gemma3nTextDecoderBlock,
)
from keras_hub.src.models.gemma3n.gemma3n_text_layers import (
    Gemma3nTextScaledWordEmbedding,
)
from keras_hub.src.models.gemma3n.rms_normalization import Gemma3nRMSNorm


class Gemma3nTextModel(keras.layers.Layer):
    """The core Gemma3n text model layer.

    This layer implements the transformer architecture of the Gemma3n model.
    It includes token embeddings, multiple decoder blocks, and final
    normalization.

    Args:
        pad_token_id: int. The id for the padding token.
        vocab_size: int. The size of the vocabulary.
        hidden_size: int. The size of the hidden states.
        num_hidden_layers: int. The number of hidden layers in the transformer.
        rms_norm_eps: float. The epsilon value for the RMS normalization layers.
        num_attention_heads: int. The number of attention heads.
        num_key_value_heads: int. The number of key-value heads for GQA.
        head_dim: int. The dimension of each attention head.
        attention_bias: bool. Whether to use a bias in the attention mechanism.
        attention_dropout: float. The dropout rate for the attention scores.
        layer_types: list of str. The type of each layer, e.g.,
            "sliding_attention".
        sliding_window: int. The sliding window size for sliding window
            attention.
        rope_theta: float. The base frequency for Rotary Positional Embeddings.
        rope_scaling: float or None. The scaling factor for RoPE.
        rope_local_base_freq: float. The base frequency for local RoPE.
        max_position_embeddings: int. The maximum sequence length.
        intermediate_size: list of int. The size of the intermediate layer in
            each of the feed-forward networks.
        hidden_activation: str. The activation function for the hidden layers.
        activation_sparsity_pattern: list of float or None. The sparsity pattern
            for activations.
        altup_num_inputs: int. The number of inputs for the AltUp mechanism.
        altup_coef_clip: float. The coefficient clipping value for AltUp.
        altup_active_idx: int. The active index for AltUp.
        altup_correct_scale: bool. Whether to correct scaling in AltUp.
        laurel_rank: int. The rank for LAUREL factorization.
        hidden_size_per_layer_input: int. The hidden size for per-layer inputs.
        vocab_size_per_layer_input: int. The vocabulary size for per-layer
            inputs.
        num_kv_shared_layers: int. The number of shared key-value layers.
    """

    def __init__(
        self,
        pad_token_id,
        vocab_size,
        hidden_size,
        num_hidden_layers,
        rms_norm_eps,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        attention_bias,
        attention_dropout,
        layer_types,
        sliding_window,
        rope_theta,
        rope_scaling,
        rope_local_base_freq,
        max_position_embeddings,
        intermediate_size,
        hidden_activation,
        activation_sparsity_pattern,
        altup_num_inputs,
        altup_coef_clip,
        altup_active_idx,
        altup_correct_scale,
        laurel_rank,
        hidden_size_per_layer_input,
        vocab_size_per_layer_input,
        num_kv_shared_layers,
        final_logit_soft_cap=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.layer_types = layer_types
        self.sliding_window = sliding_window
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_local_base_freq = rope_local_base_freq
        self.max_position_embeddings = max_position_embeddings
        self.intermediate_size = intermediate_size
        self.hidden_activation = hidden_activation
        self.activation_sparsity_pattern = activation_sparsity_pattern
        self.altup_num_inputs = altup_num_inputs
        self.altup_coef_clip = altup_coef_clip
        self.altup_active_idx = altup_active_idx
        self.altup_correct_scale = altup_correct_scale
        self.laurel_rank = laurel_rank
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.vocab_size_per_layer_input = vocab_size_per_layer_input
        self.num_kv_shared_layers = num_kv_shared_layers
        self.final_logit_soft_cap = final_logit_soft_cap
        self.first_kv_shared_layer_idx = (
            num_hidden_layers - num_kv_shared_layers
        )
        self.padding_idx = pad_token_id
        self.embed_tokens = Gemma3nTextScaledWordEmbedding(
            vocab_size,
            hidden_size,
            embed_scale=hidden_size**0.5,
            name="embed_tokens",
            dtype=self.dtype_policy,
        )
        if activation_sparsity_pattern is None:
            self.activation_sparsity_pattern = [0.0] * num_hidden_layers
        self.transformer_layers = [
            Gemma3nTextDecoderBlock(
                hidden_size,
                rms_norm_eps,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                attention_bias,
                attention_dropout,
                rope_local_base_freq
                if layer_types[i] == "sliding_attention"
                else rope_theta,
                rope_scaling,
                sliding_window
                if layer_types[i] == "sliding_attention"
                else None,
                intermediate_size[i],
                hidden_activation,
                self.activation_sparsity_pattern[i],
                altup_num_inputs,
                altup_coef_clip,
                altup_active_idx,
                altup_correct_scale,
                laurel_rank,
                hidden_size_per_layer_input,
                i >= self.first_kv_shared_layer_idx > 0,
                name=f"decoder_block_{i}",
                dtype=self.dtype_policy,
            )
            for i in range(num_hidden_layers)
        ]
        self.final_normalization = Gemma3nRMSNorm(
            hidden_size, eps=rms_norm_eps, name="norm", dtype=self.dtype_policy
        )
        self.embed_tokens_per_layer = Gemma3nTextScaledWordEmbedding(
            vocab_size_per_layer_input,
            num_hidden_layers * hidden_size_per_layer_input,
            embed_scale=hidden_size_per_layer_input**0.5,
            name="embed_tokens_per_layer",
            dtype=self.dtype_policy,
        )
        self.per_layer_model_projection = keras.layers.Dense(
            num_hidden_layers * hidden_size_per_layer_input,
            use_bias=False,
            name="per_layer_model_projection",
            dtype=self.dtype_policy,
        )
        self.per_layer_projection_norm = Gemma3nRMSNorm(
            hidden_size_per_layer_input,
            eps=rms_norm_eps,
            name="per_layer_projection_norm",
            dtype=self.dtype_policy,
        )
        self.altup_projections = [
            keras.layers.Dense(
                hidden_size,
                use_bias=False,
                name=f"altup_projection_{i}",
                dtype=self.dtype_policy,
            )
            for i in range(1, altup_num_inputs)
        ]
        self.altup_unembed_projections = [
            keras.layers.Dense(
                hidden_size,
                use_bias=False,
                name=f"altup_unembed_projection_{i}",
                dtype=self.dtype_policy,
            )
            for i in range(1, altup_num_inputs)
        ]
        self.per_layer_projection_scale = hidden_size**-0.5
        self.per_layer_input_scale = 1.0 / math.sqrt(2.0)

    def build(self, input_shape):
        if isinstance(input_shape, (list, tuple)) and isinstance(
            input_shape[0], (list, tuple)
        ):
            input_ids_shape, _, inputs_embeds_shape, _ = input_shape
        else:
            input_ids_shape = input_shape
            hidden_size = self.embed_tokens.embedding_dim
            inputs_embeds_shape = input_ids_shape[:-1] + (hidden_size,)
        self.embed_tokens.build(input_ids_shape)
        self.embed_tokens_per_layer.build(input_ids_shape)
        if not self.per_layer_model_projection.built:
            self.per_layer_model_projection.build(inputs_embeds_shape)
        per_layer_projection_norm_shape = (
            None,
            None,
            None,
            self.hidden_size_per_layer_input,
        )
        if not self.per_layer_projection_norm.built:
            self.per_layer_projection_norm.build(
                per_layer_projection_norm_shape
            )
        for proj in self.altup_projections:
            proj.build(inputs_embeds_shape)
        for proj in self.altup_unembed_projections:
            proj.build(inputs_embeds_shape)
        decoder_hidden_states_shape = (
            self.altup_num_inputs,
        ) + inputs_embeds_shape
        decoder_per_layer_input_shape = input_ids_shape + (
            self.hidden_size_per_layer_input,
        )
        decoder_input_shape = (
            decoder_hidden_states_shape,
            decoder_per_layer_input_shape,
            None,  # attention_mask
        )
        for layer in self.transformer_layers:
            layer.build(decoder_input_shape)
        self.final_normalization.build(inputs_embeds_shape)
        super().build(input_shape)

    def get_per_layer_inputs(self, input_ids):
        embeds = self.embed_tokens_per_layer(input_ids)
        return keras.ops.reshape(
            embeds,
            keras.ops.shape(input_ids)
            + (self.num_hidden_layers, self.hidden_size_per_layer_input),
        )

    def project_per_layer_inputs(self, inputs_embeds, per_layer_inputs=None):
        per_layer_projection = self.per_layer_model_projection(inputs_embeds)
        per_layer_projection = (
            per_layer_projection * self.per_layer_projection_scale
        )
        per_layer_projection = keras.ops.reshape(
            per_layer_projection,
            keras.ops.shape(inputs_embeds)[:-1]
            + (self.num_hidden_layers, self.hidden_size_per_layer_input),
        )
        per_layer_projection = self.per_layer_projection_norm(
            per_layer_projection
        )
        if per_layer_inputs is None:
            return per_layer_projection
        return (
            per_layer_projection + per_layer_inputs
        ) * self.per_layer_input_scale

    def token_embedding(self, inputs, reverse=False):
        """Apply or reverse the token embedding.

        Args:
            inputs: If `reverse=False`, token IDs to embed. If `reverse=True`,
                hidden states to convert to logits.
            reverse: bool. If False, performs embedding lookup. If True,
                computes logits by projecting hidden states through
                the transpose of the embedding matrix.
        """
        if not reverse:
            return self.embed_tokens(inputs)
        else:
            embedding_weights = self.embed_tokens.embedding.embeddings
            logits = keras.ops.matmul(
                inputs, keras.ops.transpose(embedding_weights)
            )
            if self.final_logit_soft_cap is not None:
                logits = logits / self.final_logit_soft_cap
                logits = keras.ops.tanh(logits)
                logits = logits * self.final_logit_soft_cap
            return logits

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (list, tuple)) and isinstance(
            input_shape[0], (list, tuple)
        ):
            input_ids_shape = input_shape[0]
        else:
            input_ids_shape = input_shape
        hidden_size = self.embed_tokens.embedding_dim
        return input_ids_shape + (hidden_size,)

    def call(
        self,
        input_ids,
        attention_mask,
        inputs_embeds,
        per_layer_inputs,
        cache=None,
        cache_update_index=0,
        cache_update_mask=None,
    ):
        hidden_states_0 = inputs_embeds
        target_magnitude = keras.ops.sqrt(
            keras.ops.mean(hidden_states_0**2, axis=-1, keepdims=True)
        )
        epsilon = 1e-5
        temp_hidden_states = [hidden_states_0]
        for proj in self.altup_projections:
            altup_proj = proj(hidden_states_0)
            new_magnitude = keras.ops.sqrt(
                keras.ops.maximum(
                    keras.ops.mean(altup_proj**2, axis=-1, keepdims=True),
                    epsilon,
                )
            )
            current_hidden_state = altup_proj * target_magnitude / new_magnitude
            temp_hidden_states.append(current_hidden_state)
        hidden_states = keras.ops.stack(temp_hidden_states, axis=0)

        new_caches = []
        last_nonshared_layer_idx = {}
        if cache is not None:
            for i, decoder_layer in enumerate(self.transformer_layers):
                is_kv_shared_layer = i >= self.first_kv_shared_layer_idx > 0
                layer_type = self.layer_types[i]
                per_layer_input = per_layer_inputs[:, :, i, :]
                if is_kv_shared_layer:
                    # For shared layer, use kv cache from last
                    # non-shared layer of the same type
                    current_cache = new_caches[
                        last_nonshared_layer_idx[layer_type]
                    ]
                else:
                    current_cache = cache[:, i, ...]
                    last_nonshared_layer_idx[layer_type] = i
                hidden_states, new_cache = decoder_layer(
                    (
                        hidden_states,
                        per_layer_input,
                        attention_mask,
                    ),
                    cache=current_cache,
                    cache_update_index=cache_update_index,
                    cache_update_mask=cache_update_mask,
                )
                new_caches.append(new_cache)
            cache = keras.ops.stack(new_caches, axis=1)
        else:
            for i, decoder_layer in enumerate(self.transformer_layers):
                is_kv_shared_layer = i >= self.first_kv_shared_layer_idx > 0
                layer_type = self.layer_types[i]
                per_layer_input = per_layer_inputs[:, :, i, :]
                if is_kv_shared_layer:
                    # For shared layer, use kv cache from last
                    # non-shared layer of the same type
                    current_cache = new_caches[
                        last_nonshared_layer_idx[layer_type]
                    ]
                else:
                    current_cache = None
                    last_nonshared_layer_idx[layer_type] = i
                hidden_states, new_cache = decoder_layer(
                    (
                        hidden_states,
                        per_layer_input,
                        attention_mask,
                    ),
                    cache=current_cache,
                )
                new_caches.append(new_cache)
        target_magnitude = keras.ops.sqrt(
            keras.ops.mean(hidden_states[0] ** 2, axis=-1, keepdims=True)
        )
        temp_hidden_states = [hidden_states[0]]
        for i, proj in enumerate(self.altup_unembed_projections):
            altup_unemb_proj = proj(hidden_states[i + 1])
            new_magnitude = keras.ops.sqrt(
                keras.ops.maximum(
                    keras.ops.mean(altup_unemb_proj**2, axis=-1, keepdims=True),
                    epsilon,
                )
            )
            current_hidden_state = (
                altup_unemb_proj * target_magnitude / new_magnitude
            )
            temp_hidden_states.append(current_hidden_state)
        hidden_states = keras.ops.stack(temp_hidden_states)
        hidden_states = keras.ops.mean(hidden_states, axis=0)
        normalized = self.final_normalization(hidden_states)
        if cache is not None:
            return normalized, cache
        return normalized

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pad_token_id": self.pad_token_id,
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_hidden_layers": self.num_hidden_layers,
                "rms_norm_eps": self.rms_norm_eps,
                "num_attention_heads": self.num_attention_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "attention_bias": self.attention_bias,
                "attention_dropout": self.attention_dropout,
                "layer_types": self.layer_types,
                "sliding_window": self.sliding_window,
                "rope_theta": self.rope_theta,
                "rope_scaling": self.rope_scaling,
                "rope_local_base_freq": self.rope_local_base_freq,
                "max_position_embeddings": self.max_position_embeddings,
                "intermediate_size": self.intermediate_size,
                "hidden_activation": self.hidden_activation,
                "activation_sparsity_pattern": self.activation_sparsity_pattern,
                "altup_num_inputs": self.altup_num_inputs,
                "altup_coef_clip": self.altup_coef_clip,
                "altup_active_idx": self.altup_active_idx,
                "altup_correct_scale": self.altup_correct_scale,
                "laurel_rank": self.laurel_rank,
                "hidden_size_per_layer_input": self.hidden_size_per_layer_input,
                "vocab_size_per_layer_input": self.vocab_size_per_layer_input,
                "num_kv_shared_layers": self.num_kv_shared_layers,
                "final_logit_soft_cap": self.final_logit_soft_cap,
            }
        )
        return config
