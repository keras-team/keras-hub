import keras
from keras import ops

from keras_hub.src.models.qwen3.qwen3_decoder import Qwen3TransformerDecoder
from keras_hub.src.models.qwen3_vl.qwen3_vl_attention import Qwen3VLAttention
from keras_hub.src.models.qwen3_vl.qwen3_vl_layers import Qwen3VLLayerNorm
from keras_hub.src.utils.keras_utils import clone_initializer


class Qwen3VLTransformerDecoder(Qwen3TransformerDecoder):
    """A Transformer decoder layer for the Qwen3-VL text model.

    The layer is a pre-norm block with interleaved multimodal RoPE
    attention and a SwiGLU feed-forward network. It reuses the layers of
    `Qwen3TransformerDecoder` and swaps in `Qwen3VLAttention` and
    `Qwen3VLLayerNorm`.

    Args:
        intermediate_dim: int. Hidden size of the feed-forward network.
        head_dim: int. Size of each attention head.
        num_query_heads: int. Number of query heads.
        num_key_value_heads: int. Number of key/value heads.
        rope_max_wavelength: float. Base of the rotary frequencies.
            Defaults to `5000000.0`.
        mrope_section: list of 3 ints. Frequencies used by the temporal,
            height and width position channels. Defaults to `(24, 20, 20)`.
        layer_norm_epsilon: float. Epsilon of the RMSNorm layers. Defaults
            to `1e-6`.
        activation: Activation of the feed-forward gate. Defaults to
            `"silu"`.
        kernel_initializer: Initializer for the dense kernels. Defaults to
            `"glorot_uniform"`.
        dropout: float. Dropout rate of the attention probabilities. As in
            HF, the residual branches have no dropout. Defaults to `0.0`.
    """

    def __init__(
        self,
        intermediate_dim,
        head_dim,
        num_query_heads,
        num_key_value_heads,
        rope_max_wavelength=5000000.0,
        mrope_section=(24, 20, 20),
        layer_norm_epsilon=1e-6,
        activation="silu",
        kernel_initializer="glorot_uniform",
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(
            intermediate_dim=intermediate_dim,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rope_max_wavelength=rope_max_wavelength,
            activation=activation,
            layer_norm_epsilon=layer_norm_epsilon,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
            **kwargs,
        )
        self.mrope_section = list(mrope_section)

    def _make_self_attention(self):
        return Qwen3VLAttention(
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            rope_max_wavelength=self.rope_max_wavelength,
            mrope_section=self.mrope_section,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dropout=self.dropout,
            layer_norm_epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attention",
        )

    def _make_self_attention_dropout(self):
        # HF only drops the attention probabilities.
        return None

    def _make_layer_norm(self, name):
        return Qwen3VLLayerNorm(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name=name,
        )

    def call(
        self,
        decoder_sequence,
        decoder_padding_mask=None,
        decoder_attention_mask=None,
        self_attention_cache=None,
        self_attention_cache_update_index=None,
        position_ids=None,
        rotary_embeddings=None,
        training=None,
    ):
        """Forward pass of the decoder layer.

        Same arguments as `Qwen3TransformerDecoder.call`, plus:

        Args:
            position_ids: Optional int Tensor of shape `(batch, 3, seq_len)`
                with the T, H and W positions. `None` means text-only
                positions.
            rotary_embeddings: Optional `(cos, sin)` tuple shared by all the
                layers of a model, see
                `MultimodalRotaryEmbedding.compute_embeddings`.

        Returns:
            The layer output, plus the updated cache if
            `self_attention_cache` is given.
        """
        x = decoder_sequence
        cache = self_attention_cache
        cache_update_index = self_attention_cache_update_index
        attention_mask = self._compute_self_attention_mask(
            decoder_sequence=x,
            decoder_padding_mask=decoder_padding_mask,
            decoder_attention_mask=decoder_attention_mask,
            self_attention_cache=cache,
            self_attention_cache_update_index=cache_update_index,
        )
        residual = x
        x = self._self_attention_layernorm(x)
        x = self._self_attention_layer(
            hidden_states=x,
            attention_mask=attention_mask,
            cache=cache,
            cache_update_index=cache_update_index,
            position_ids=position_ids,
            rotary_embeddings=rotary_embeddings,
            training=training,
        )
        if cache is not None:
            x, cache = x
        x = x + residual

        residual = x
        x = self._feedforward_layernorm(x)
        # Run the activation in float32, as torch does for SiLU.
        gate = ops.cast(self._feedforward_gate_dense(x), "float32")
        gate = ops.cast(self.activation(gate), self.compute_dtype)
        x = self._feedforward_intermediate_dense(x)
        x = self._feedforward_output_dense(ops.multiply(x, gate))
        x = x + residual

        if cache is not None:
            return x, cache
        return x

    def compute_output_spec(
        self, decoder_sequence, *args, self_attention_cache=None, **kwargs
    ):
        cache = self_attention_cache
        output = keras.KerasTensor(
            decoder_sequence.shape, dtype=self.compute_dtype
        )
        if cache is None:
            return output
        return output, keras.KerasTensor(cache.shape, dtype=cache.dtype)

    def get_config(self):
        config = super().get_config()
        config.pop("rope_scaling_factor")
        config.pop("sliding_window_size")
        config.update({"mrope_section": self.mrope_section})
        return config
