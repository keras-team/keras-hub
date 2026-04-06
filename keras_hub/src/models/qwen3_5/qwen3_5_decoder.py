import keras
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.qwen3_5.qwen3_5_attention import Qwen3_5Attention
from keras_hub.src.models.qwen3_5.qwen3_5_gated_delta_net import (
    Qwen3_5GatedDeltaNet,
)
from keras_hub.src.models.qwen3_5.qwen3_5_layernorm import Qwen3_5LayerNorm
from keras_hub.src.utils.keras_utils import clone_initializer


class Qwen3_5TransformerDecoder(keras.layers.Layer):
    """A Transformer decoder layer for Qwen3.5.

    Dispatches between full self-attention and linear attention
    (GatedDeltaNet) based on ``layer_type``.

    Args:
        layer_type: One of ``"full_attention"`` or ``"linear_attention"``.
        intermediate_dim: FFN intermediate dimension.
        num_query_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value attention heads (GQA).
        head_dim: Dimension of each attention head.
        partial_rotary_factor: Fraction of head_dim that gets RoPE.
        rope_max_wavelength: Maximum wavelength for rotary embeddings.
        rope_scaling_factor: Scaling factor for rotary embeddings.
        activation: Activation function for the FFN.
        layer_norm_epsilon: Epsilon for layer norms.
        kernel_initializer: Initializer for projection kernels.
        dropout: Dropout rate.
        sliding_window_size: Sliding window size (full_attention only).
        linear_num_key_heads: Number of key heads (linear_attention).
        linear_num_value_heads: Number of value heads (linear_attention).
        linear_key_head_dim: Key head dim (linear_attention).
        linear_value_head_dim: Value head dim (linear_attention).
        linear_conv_kernel_dim: Conv kernel size (linear_attention).
    """

    def __init__(
        self,
        layer_type,
        intermediate_dim,
        num_query_heads,
        num_key_value_heads,
        head_dim,
        partial_rotary_factor=0.25,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        activation="silu",
        layer_norm_epsilon=1e-6,
        kernel_initializer="glorot_uniform",
        dropout=0.0,
        sliding_window_size=None,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        mrope_section=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer_type = layer_type
        self.intermediate_dim = intermediate_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.dropout = dropout
        self.sliding_window_size = sliding_window_size
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.mrope_section = mrope_section
        self.supports_masking = True

    def build(self, decoder_sequence_shape):
        self._decoder_sequence_shape = decoder_sequence_shape
        self.hidden_dim = decoder_sequence_shape[-1]

        # Token mixer — dispatched by layer_type.
        if self.layer_type == "linear_attention":
            self._linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=self.hidden_dim,
                linear_num_key_heads=self.linear_num_key_heads,
                linear_num_value_heads=self.linear_num_value_heads,
                linear_key_head_dim=self.linear_key_head_dim,
                linear_value_head_dim=self.linear_value_head_dim,
                linear_conv_kernel_dim=self.linear_conv_kernel_dim,
                layer_norm_epsilon=self.layer_norm_epsilon,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                dtype=self.dtype_policy,
                name="linear_attn",
            )
            self._linear_attn.build(decoder_sequence_shape)
        elif self.layer_type == "full_attention":
            self._self_attention_layer = Qwen3_5Attention(
                num_query_heads=self.num_query_heads,
                num_key_value_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                partial_rotary_factor=self.partial_rotary_factor,
                rope_max_wavelength=self.rope_max_wavelength,
                rope_scaling_factor=self.rope_scaling_factor,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                dropout=self.dropout,
                layer_norm_epsilon=self.layer_norm_epsilon,
                sliding_window_size=self.sliding_window_size,
                mrope_section=self.mrope_section,
                dtype=self.dtype_policy,
                name="self_attention",
            )
            self._self_attention_layer.build(decoder_sequence_shape)
        else:
            raise ValueError(
                f"Unknown layer_type '{self.layer_type}'. "
                "Expected 'full_attention' or 'linear_attention'."
            )

        # Pre-norm.
        self._input_layernorm = Qwen3_5LayerNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="input_layernorm",
        )
        self._input_layernorm.build(decoder_sequence_shape)

        self._self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention_dropout",
        )

        # Feedforward layers (SwiGLU).
        self._feedforward_gate_dense = keras.layers.Dense(
            self.intermediate_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            use_bias=False,
            dtype=self.dtype_policy,
            name="feedforward_gate_dense",
        )
        self._feedforward_gate_dense.build(decoder_sequence_shape)

        self._feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            use_bias=False,
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self._feedforward_intermediate_dense.build(decoder_sequence_shape)

        self._feedforward_output_dense = keras.layers.Dense(
            self.hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            use_bias=False,
            dtype=self.dtype_policy,
            name="feedforward_output_dense",
        )
        self._feedforward_output_dense.build(
            self._feedforward_gate_dense.compute_output_shape(
                decoder_sequence_shape
            )
        )

        self._post_attention_layernorm = Qwen3_5LayerNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="post_attention_layernorm",
        )
        self._post_attention_layernorm.build(decoder_sequence_shape)

        self.built = True

    def call(
        self,
        decoder_sequence,
        decoder_padding_mask=None,
        decoder_attention_mask=None,
        self_attention_cache=None,
        self_attention_cache_update_index=None,
        position_ids=None,
        training=None,
    ):
        residual = decoder_sequence
        x = self._input_layernorm(decoder_sequence)

        # Token mixer.
        if self.layer_type == "linear_attention":
            # GatedDeltaNet uses only a 2D padding mask.
            x = self._linear_attn(
                x,
                attention_mask=decoder_padding_mask,
                cache=self_attention_cache,
                cache_update_index=self_attention_cache_update_index,
                training=training,
            )
            if self_attention_cache is not None:
                x, self_attention_cache = x
        elif self.layer_type == "full_attention":
            self_attention_mask = self._compute_self_attention_mask(
                decoder_sequence=decoder_sequence,
                decoder_padding_mask=decoder_padding_mask,
                decoder_attention_mask=decoder_attention_mask,
                self_attention_cache=self_attention_cache,
                self_attention_cache_update_index=(
                    self_attention_cache_update_index
                ),
            )
            x = self._self_attention_layer(
                hidden_states=x,
                attention_mask=self_attention_mask,
                cache=self_attention_cache,
                cache_update_index=self_attention_cache_update_index,
                position_ids=position_ids,
            )
            if self_attention_cache is not None:
                x, self_attention_cache = x

        x = self._self_attention_dropout(x, training=training)
        x = x + residual

        # Feedforward block (SwiGLU).
        residual = x
        x = self._post_attention_layernorm(x)
        gate_output = self._feedforward_gate_dense(x)

        gate_output = ops.cast(gate_output, "float32")
        gate_output = self.activation(gate_output)
        gate_output = ops.cast(gate_output, self.compute_dtype)

        x = self._feedforward_intermediate_dense(x)
        x = self._feedforward_output_dense(ops.multiply(x, gate_output))

        decoder_output = x + residual

        if self_attention_cache is not None:
            if self.layer_type == "linear_attention":
                return (
                    decoder_output,
                    self_attention_cache[0],
                    self_attention_cache[1],
                )
            return decoder_output, self_attention_cache
        return decoder_output

    def call_and_update_cache(
        self,
        decoder_sequence,
        kv_cache,
        conv_cache,
        recurrent_cache,
        cache_update_index,
        decoder_padding_mask=None,
        position_ids=None,
    ):
        """Forward pass with a uniform cache interface.

        Each layer type updates only its own cache and passes the others
        through unchanged. This allows the caller to iterate over layers
        without branching on ``layer_type``.

        Args:
            decoder_sequence: Hidden states (batch, seq_len, hidden_dim).
            kv_cache: KV cache slice for this layer
                (batch, 2, seq_len, num_kv_heads, head_dim).
            conv_cache: Conv cache slice for this layer
                (batch, conv_dim, conv_kernel_size - 1).
            recurrent_cache: Recurrent cache slice for this layer
                (batch, num_value_heads, key_head_dim, value_head_dim).
            cache_update_index: Int, current step index.
            decoder_padding_mask: Optional padding mask.
            position_ids: Optional M-RoPE position IDs (full_attention
                only).

        Returns:
            Tuple of (output, updated_kv_cache, updated_conv_cache,
            updated_recurrent_cache).
        """
        if self.layer_type == "full_attention":
            output, updated_kv = self(
                decoder_sequence,
                decoder_padding_mask=decoder_padding_mask,
                self_attention_cache=kv_cache,
                self_attention_cache_update_index=cache_update_index,
                position_ids=position_ids,
            )
            return output, updated_kv, conv_cache, recurrent_cache
        else:
            output, updated_conv, updated_recurrent = self(
                decoder_sequence,
                decoder_padding_mask=decoder_padding_mask,
                self_attention_cache=(conv_cache, recurrent_cache),
                self_attention_cache_update_index=cache_update_index,
            )
            return output, kv_cache, updated_conv, updated_recurrent

    def _compute_self_attention_mask(
        self,
        decoder_sequence,
        decoder_padding_mask,
        decoder_attention_mask,
        self_attention_cache,
        self_attention_cache_update_index,
    ):
        decoder_mask = merge_padding_and_attention_mask(
            decoder_sequence,
            decoder_padding_mask,
            decoder_attention_mask,
        )
        batch_size = ops.shape(decoder_sequence)[0]
        input_length = output_length = ops.shape(decoder_sequence)[1]
        if self_attention_cache is not None:
            input_length = ops.shape(self_attention_cache)[2]

        cache_update_index = (
            0
            if self_attention_cache_update_index is None
            else self_attention_cache_update_index
        )
        causal_mask = compute_causal_mask(
            batch_size,
            input_length,
            output_length,
            cache_update_index,
        )
        return (
            ops.minimum(decoder_mask, causal_mask)
            if decoder_mask is not None
            else causal_mask
        )

    def compute_output_shape(self, decoder_sequence_shape):
        return decoder_sequence_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layer_type": self.layer_type,
                "intermediate_dim": self.intermediate_dim,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "head_dim": self.head_dim,
                "partial_rotary_factor": self.partial_rotary_factor,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "dropout": self.dropout,
                "sliding_window_size": self.sliding_window_size,
                "linear_num_key_heads": self.linear_num_key_heads,
                "linear_num_value_heads": self.linear_num_value_heads,
                "linear_key_head_dim": self.linear_key_head_dim,
                "linear_value_head_dim": self.linear_value_head_dim,
                "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
                "mrope_section": self.mrope_section,
            }
        )
        return config
