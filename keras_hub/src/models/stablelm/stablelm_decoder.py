import keras
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.utils.keras_utils import clone_initializer
from keras_hub.src.models.stablelm.stablelm_attention import StableLMAttention 

class StableLMTransformerDecoder(keras.layers.Layer):
    """StableLM-3B4E1T Transformer decoder layer.

    This layer implements the decoder for StableLM-3B4E1T, a decoder-only transformer
    with multi-head self-attention using partial rotary position embeddings (RoPE)
    and LayerNorm with learned bias terms.

    Args:
        intermediate_dim (int): Hidden size of the feedforward network.
        num_query_heads (int): Number of query attention heads (32 for StableLM-3B4E1T).
        num_key_value_heads (int): Number of key/value attention heads (32 for StableLM-3B4E1T).
        rope_max_wavelength (float, optional): Maximum wavelength for RoPE. Defaults to 10000.
        rope_scaling_factor (float, optional): Scaling factor for RoPE. Defaults to 1.0.
        rotary_percentage (float, optional): Percentage of head dimensions for RoPE (0.25 for StableLM).
        activation (str or callable, optional): Activation for the feedforward network. Defaults to "silu".
        layer_norm_epsilon (float, optional): Epsilon for LayerNorm. Defaults to 1e-5.
        kernel_initializer (str or initializer, optional): Initializer for dense layers. Defaults to "glorot_uniform".
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        **kwargs: Additional keyword arguments for the parent class.
    """

    def __init__(
        self,
        intermediate_dim,
        num_query_heads,
        num_key_value_heads,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        rotary_percentage=0.25,
        activation="silu",
        layer_norm_epsilon=1e-5,
        kernel_initializer="glorot_uniform",
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.rotary_percentage = rotary_percentage
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.dropout = dropout
        self.supports_masking = True

    def build(self, decoder_sequence_shape):
        self.hidden_dim = decoder_sequence_shape[-1]

        # Self-attention layer with partial RoPE
        self._self_attention_layer = StableLMAttention(
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            hidden_dim=self.hidden_dim,
            rope_max_wavelength=self.rope_max_wavelength,
            rope_scaling_factor=self.rope_scaling_factor,
            rotary_percentage=self.rotary_percentage,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dropout=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention",
        )
        self._self_attention_layer.build(decoder_sequence_shape)

        # LayerNorm for self-attention (with learned bias)
        self._self_attention_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attention_layernorm",
        )
        self._self_attention_layernorm.build(decoder_sequence_shape)

        # Dropout for self-attention
        self._self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention_dropout",
        )

        # Feedforward layers (gated MLP)
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
            self._feedforward_gate_dense.compute_output_shape(decoder_sequence_shape)
        )

        # LayerNorm for feedforward (with learned bias)
        self._feedforward_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="feedforward_layernorm",
        )
        self._feedforward_layernorm.build(decoder_sequence_shape)

        self.built = True

    def call(
        self,
        decoder_sequence,
        decoder_padding_mask=None,
        decoder_attention_mask=None,
        self_attention_cache=None,
        self_attention_cache_update_index=None,
        training=None,
    ):
        # Compute the attention mask
        self_attention_mask = self._compute_self_attention_mask(
            decoder_sequence=decoder_sequence,
            decoder_padding_mask=decoder_padding_mask,
            decoder_attention_mask=decoder_attention_mask,
            self_attention_cache=self_attention_cache,
            self_attention_cache_update_index=self_attention_cache_update_index,
        )

        residual = decoder_sequence

        # Self-attention block
        x = self._self_attention_layernorm(decoder_sequence)
        x, self_attention_cache = self._self_attention_layer(
            hidden_states=x,
            attention_mask=self_attention_mask,
            cache=self_attention_cache,
            cache_update_index=self_attention_cache_update_index,
        )
        x = self._self_attention_dropout(x, training=training)
        x = x + residual

        residual = x

        # Feedforward block
        x = self._feedforward_layernorm(x)
        gate_output = self._feedforward_gate_dense(x)
        gate_output = self.activation(gate_output)
        intermediate_output = self._feedforward_intermediate_dense(x)
        x = self._feedforward_output_dense(ops.multiply(intermediate_output, gate_output))
        decoder_output = x + residual

        if self_attention_cache is not None:
            return decoder_output, self_attention_cache
        return decoder_output

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
        if self_attention_cache is not None:
            input_length = ops.shape(self_attention_cache)[2]
        cache_update_index = 0 if self_attention_cache_update_index is None else self_attention_cache_update_index
        causal_mask = compute_causal_mask(
            batch_size, input_length, output_length, cache_update_index
        )
        return ops.minimum(decoder_mask, causal_mask) if decoder_mask is not None else causal_mask

    def compute_output_shape(self, decoder_sequence_shape):
        return decoder_sequence_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "rotary_percentage": self.rotary_percentage,
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
                "dropout": self.dropout,
            }
        )
        return config