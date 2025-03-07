import keras

from keras_hub.src.models.llama.llama_decoder import LlamaTransformerDecoder
from keras_hub.src.models.llama.llama_layernorm import LlamaLayerNorm
from keras_hub.src.models.llama31.llama31_attention import Llama31Attention
from keras_hub.src.utils.keras_utils import clone_initializer


class Llama31TransformerDecoder(LlamaTransformerDecoder):
    """A Transformer decoder layer for the Llama backbone."""

    def __init__(
        self,
        intermediate_dim,
        num_query_heads,
        num_key_value_heads,
        rope_max_wavelength=10000,
        rope_scaling_factor=1.0,
        rope_factor=8,
        rope_low_freq_factor=1,
        rope_high_freq_factor=4,
        rope_old_context_len=8192,
        activation="silu",
        layer_norm_epsilon=1e-5,
        kernel_initializer="glorot_uniform",
        dropout=0,
        **kwargs,
    ):
        super().__init__(
            intermediate_dim, num_query_heads, num_key_value_heads, **kwargs
        )
        self.intermediate_dim = intermediate_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads

        self.rope_max_wavelength = rope_max_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_factor = rope_factor
        self.rope_low_freq_factor = rope_low_freq_factor
        self.rope_high_freq_factor = rope_high_freq_factor
        self.rope_old_context_len = rope_old_context_len

        self.dropout = dropout

        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)

        self.supports_masking = True

    def build(self, decoder_sequence_shape):
        self._decoder_sequence_shape = decoder_sequence_shape
        self.hidden_dim = decoder_sequence_shape[-1]

        # Self attention layer.
        self._self_attention_layer = Llama31Attention(
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            rope_max_wavelength=self.rope_max_wavelength,
            rope_scaling_factor=self.rope_scaling_factor,
            rope_factor=self.rope_factor,
            rope_high_freq_factor=self.rope_high_freq_factor,
            rope_low_freq_factor=self.rope_low_freq_factor,
            rope_old_context_len=self.rope_old_context_len,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dropout=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention",
        )

        self._self_attention_layer.build(decoder_sequence_shape)

        self._self_attention_layernorm = LlamaLayerNorm(
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

        self._feedforward_layernorm = LlamaLayerNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="feedforward_layernorm",
        )
        self._feedforward_layernorm.build(decoder_sequence_shape)

        self.built = True

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_query_heads": self.num_query_heads,
                "rope_max_wavelength": self.rope_max_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "num_key_value_heads": self.num_key_value_heads,
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "dropout": self.dropout,
            }
        )
        return config
