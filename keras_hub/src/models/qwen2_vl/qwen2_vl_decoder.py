import keras
from keras import ops

from keras_hub.src.models.qwen.qwen_decoder import QwenTransformerDecoder
from keras_hub.src.models.qwen.qwen_layernorm import QwenLayerNorm
from keras_hub.src.models.qwen2_vl.qwen2_vl_attention import Qwen2VLAttention
from keras_hub.src.utils.keras_utils import clone_initializer


class Qwen2VLTransformerDecoder(QwenTransformerDecoder):
    """A Transformer decoder layer for the Qwen2-VL backbone.

    Identical to ``QwenTransformerDecoder`` (same sublayers, same weight
    names) but threads ``position_ids`` through to the attention layer for
    M-RoPE support.

    Args:
        mrope_section: list of int or None. M-RoPE section sizes passed to
            ``Qwen2VLAttention``. ``None`` disables M-RoPE.
    """

    def __init__(self, *args, mrope_section=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mrope_section = mrope_section

    def build(self, decoder_sequence_shape):
        self._decoder_sequence_shape = decoder_sequence_shape
        self.hidden_dim = decoder_sequence_shape[-1]

        # Self attention layer.
        self._self_attention_layer = Qwen2VLAttention(
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            rope_max_wavelength=self.rope_max_wavelength,
            rope_scaling_factor=self.rope_scaling_factor,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dropout=self.dropout,
            use_sliding_window_attention=self.use_sliding_window_attention,
            sliding_window_size=self.sliding_window_size,
            mrope_section=self.mrope_section,
            dtype=self.dtype_policy,
            name="self_attention",
        )
        self._self_attention_layer.build(decoder_sequence_shape)

        self._self_attention_layernorm = QwenLayerNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attention_layernorm",
        )
        self._self_attention_layernorm.build(decoder_sequence_shape)

        self._self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention_dropout",
        )

        # Feedforward layers.
        self._feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            use_bias=False,
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self._feedforward_intermediate_dense.build(decoder_sequence_shape)

        self._feedforward_gate_dense = keras.layers.Dense(
            self.intermediate_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            use_bias=False,
            dtype=self.dtype_policy,
            name="feedforward_gate_dense",
        )
        self._feedforward_gate_dense.build(decoder_sequence_shape)

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

        self._feedforward_layernorm = QwenLayerNorm(
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
        position_ids=None,
        training=None,
    ):
        """Forward pass for the decoder layer.

        Same signature as ``QwenTransformerDecoder.call`` plus:

        Args:
            position_ids: Optional int tensor ``(batch, 3, seq_len)`` with
                per-channel M-RoPE positions for the current tokens.
        """
        self_attention_mask = self._compute_self_attention_mask(
            decoder_sequence=decoder_sequence,
            decoder_padding_mask=decoder_padding_mask,
            decoder_attention_mask=decoder_attention_mask,
            self_attention_cache=self_attention_cache,
            self_attention_cache_update_index=self_attention_cache_update_index,
        )
        residual = decoder_sequence

        x = self._self_attention_layernorm(decoder_sequence)

        # Self attention block.
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
        residual = x

        x = self._feedforward_layernorm(x)
        gate_output = self._feedforward_gate_dense(x)

        # Run the activation function in full 32-bit precision, matching
        # `torch.nn.functional.silu` behavior.
        gate_output = ops.cast(gate_output, "float32")
        gate_output = self.activation(gate_output)
        gate_output = ops.cast(gate_output, self.compute_dtype)

        x = self._feedforward_intermediate_dense(x)

        x = self._feedforward_output_dense(ops.multiply(x, gate_output))

        decoder_output = x + residual

        if self_attention_cache is not None:
            return decoder_output, self_attention_cache
        return decoder_output

    def get_config(self):
        config = super().get_config()
        config.update({"mrope_section": self.mrope_section})
        return config
