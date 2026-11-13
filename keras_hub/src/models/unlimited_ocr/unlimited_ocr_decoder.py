import keras
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.qwen_moe.qwen_moe_decoder import QwenMoeMLP
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_attention import (
    UnlimitedOCRAttention,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_layers import (
    UnlimitedOCRSparseMoeBlock,
)
from keras_hub.src.utils.keras_utils import clone_initializer


class UnlimitedOCRTransformerDecoder(keras.layers.Layer):
    """A pre-norm transformer decoder block with a mixture of experts.

    The block applies RMS norm, attention, a residual add, RMS norm, a
    feedforward block, and a residual add. The feedforward block is either a
    dense gated MLP or a sparse mixture of experts.

    The block supports the reference sliding window attention (R-SWA) for
    cached decoding. When `prefill_length` is set, a query attends to every
    prefill token and to the last `sliding_window` positions, itself
    included. Without a cache, or without `prefill_length`, the block uses
    full causal attention.

    Args:
        intermediate_dim: int. The intermediate size of the dense MLP.
        num_query_heads: int. The number of attention heads.
        num_key_value_heads: int or `None`. The number of key and value
            heads. If `None`, it equals `num_query_heads`. Defaults to
            `None`.
        moe_intermediate_dim: int. The intermediate size of one routed
            expert.
        num_shared_experts: int. The number of shared experts.
        num_experts: int. The number of routed experts.
        top_k: int. The number of routed experts for each token.
        sparse: bool. If `True`, the feedforward block is a mixture of
            experts. If `False`, the feedforward block is a dense MLP.
        routed_scaling_factor: float. The factor that scales the routing
            weights. Defaults to `1.0`.
        rope_max_wavelength: int. The maximum wavelength of the rotary
            embedding. Defaults to `10000`.
        layer_norm_epsilon: float. The epsilon of the RMS norm layers.
            Defaults to `1e-6`.
        sliding_window: int or `None`. The decode window of the R-SWA. If
            `None`, the block uses full causal attention, even with a cache
            and `prefill_length`. Defaults to `128`.
        kernel_initializer: string or `keras.initializers` initializer. The
            initializer of the kernels. Defaults to `"glorot_uniform"`.
    """

    def __init__(
        self,
        intermediate_dim,
        num_query_heads,
        moe_intermediate_dim,
        num_shared_experts,
        num_experts,
        top_k,
        sparse=True,
        routed_scaling_factor=1.0,
        rope_max_wavelength=10000,
        layer_norm_epsilon=1e-6,
        sliding_window=128,
        num_key_value_heads=None,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.moe_intermediate_dim = moe_intermediate_dim
        self.num_shared_experts = num_shared_experts
        self.num_experts = num_experts
        self.top_k = top_k
        self.sparse = sparse
        self.routed_scaling_factor = routed_scaling_factor
        self.rope_max_wavelength = rope_max_wavelength
        self.layer_norm_epsilon = layer_norm_epsilon
        self.sliding_window = sliding_window
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.supports_masking = True

    def build(self, decoder_sequence_shape):
        self.hidden_dim = decoder_sequence_shape[-1]

        self._self_attention_layer = UnlimitedOCRAttention(
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            rope_max_wavelength=self.rope_max_wavelength,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=self.dtype_policy,
            name="self_attention",
        )
        self._self_attention_layer.build(decoder_sequence_shape)

        self._self_attention_layernorm = keras.layers.RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attention_layernorm",
        )
        self._self_attention_layernorm.build(decoder_sequence_shape)

        if self.sparse:
            self.mlp = UnlimitedOCRSparseMoeBlock(
                hidden_dim=self.hidden_dim,
                moe_intermediate_dim=self.moe_intermediate_dim,
                num_shared_experts=self.num_shared_experts,
                num_experts=self.num_experts,
                top_k=self.top_k,
                routed_scaling_factor=self.routed_scaling_factor,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                dtype=self.dtype_policy,
                name="mlp",
            )
        else:
            self.mlp = QwenMoeMLP(
                intermediate_dim=self.intermediate_dim,
                hidden_dim=self.hidden_dim,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                dtype=self.dtype_policy,
                name="mlp",
            )
        self.mlp.build(decoder_sequence_shape)

        self._feedforward_layernorm = keras.layers.RMSNormalization(
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
        prefill_length=None,
        cache_write_mask=None,
    ):
        """Run the decoder block.

        Args:
            decoder_sequence: Tensor of shape `(batch, length, hidden_dim)`.
            decoder_padding_mask: Optional tensor of shape `(batch, length)`.
            decoder_attention_mask: Optional tensor of shape
                `(batch, length, key_length)`.
            self_attention_cache: Optional key/value cache tensor.
            self_attention_cache_update_index: Optional int. The position of
                the first token in `decoder_sequence`.
            prefill_length: Optional int tensor of shape `(batch,)`. The
                number of prefill tokens of each row. The R-SWA mask applies
                only when the layer has a cache and this value is set.
            cache_write_mask: Optional boolean tensor of shape `(batch,)`.
                A row with `False` keeps its cached keys and values at the
                updated positions.

        Returns:
            The block output. When `self_attention_cache` is set, the block
            returns an `(output, cache)` tuple.
        """
        self_attention_mask = self._compute_self_attention_mask(
            decoder_sequence=decoder_sequence,
            decoder_padding_mask=decoder_padding_mask,
            decoder_attention_mask=decoder_attention_mask,
            self_attention_cache=self_attention_cache,
            self_attention_cache_update_index=self_attention_cache_update_index,
            prefill_length=prefill_length,
        )
        residual = decoder_sequence
        x = self._self_attention_layernorm(decoder_sequence)
        x = self._self_attention_layer(
            hidden_states=x,
            attention_mask=self_attention_mask,
            cache=self_attention_cache,
            cache_update_index=self_attention_cache_update_index,
            cache_write_mask=cache_write_mask,
        )
        if self_attention_cache is not None:
            x, self_attention_cache = x
        x = x + residual

        residual = x
        x = self._feedforward_layernorm(x)
        x = self.mlp(x)
        x = ops.cast(x, ops.dtype(residual))
        decoder_output = ops.cast(x + residual, self.compute_dtype)

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
        prefill_length,
    ):
        decoder_mask = merge_padding_and_attention_mask(
            decoder_sequence, decoder_padding_mask, decoder_attention_mask
        )
        batch_size = ops.shape(decoder_sequence)[0]
        input_length = output_length = ops.shape(decoder_sequence)[1]
        # A cached decode step has a rectangular mask. The query has length
        # `output_length` and the keys have the length of the cache.
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
        mask = (
            ops.minimum(decoder_mask, causal_mask)
            if decoder_mask is not None
            else causal_mask
        )

        if (
            prefill_length is not None
            and self_attention_cache is not None
            and self.sliding_window is not None
        ):
            # R-SWA mask: a query sees every prefill token and the last
            # `sliding_window` positions (modeling_unlimited_ocr.py:1705).
            query_positions = ops.arange(
                output_length, dtype="int32"
            ) + ops.cast(cache_update_index, "int32")
            key_positions = ops.arange(input_length, dtype="int32")
            prefill_length = ops.cast(prefill_length, "int32")
            in_prefill = (
                key_positions[None, None, :] < prefill_length[:, None, None]
            )
            in_window = (
                key_positions[None, None, :]
                > query_positions[None, :, None] - self.sliding_window
            )
            mask = ops.logical_and(
                ops.cast(mask, "bool"), ops.logical_or(in_prefill, in_window)
            )
        return mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "moe_intermediate_dim": self.moe_intermediate_dim,
                "num_shared_experts": self.num_shared_experts,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "sparse": self.sparse,
                "routed_scaling_factor": self.routed_scaling_factor,
                "rope_max_wavelength": self.rope_max_wavelength,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "sliding_window": self.sliding_window,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config
