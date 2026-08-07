import keras
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.gemma4.gemma4_attention import Gemma4TextAttention
from keras_hub.src.models.gemma4.gemma4_attention import Gemma4VisionAttention
from keras_hub.src.models.gemma4.gemma4_layers import RMSNormalization
from keras_hub.src.models.gemma4.gemma4_moe import Gemma4MoEBlock
from keras_hub.src.models.gemma4.gemma4_moe import Gemma4Router


class DiffusionGemmaTransformerLayer(keras.layers.Layer):
    """Transformer layer for DiffusionGemma.

    Identical to `Gemma4TextDecoderBlock` with two additions:

    1. **`encoder_layer_scalar`** — a second non-trainable scalar (init 1.0)
       used when `is_encoder=True` (causal encoder pass).  The existing
       `layer_scalar` is used for decoder passes (`is_encoder=False`).

    2. **Canvas bidirectional attention** — when `canvas_mask` is not `None`,
       canvas query positions attend bidirectionally to all canvas key positions
       in the KV cache, overriding the causal mask.

    The layer acts as both the causal encoder (prompt → KV cache) and the
    bidirectional decoder (canvas denoising), hence "transformer layer" rather
    than "decoder block".

    Args:
        hidden_dim: int. Dimensionality of the model's hidden representations.
        intermediate_dim: int. Dimensionality of the feed-forward intermediate
            layer.
        head_dim: int. Dimensionality of each attention head.
        num_query_heads: int. Number of query attention heads.
        num_key_value_heads: int. Number of key/value attention heads (for
            grouped-query attention).
        logit_soft_cap: float. Optional soft cap applied to attention logits
            before softmax. Defaults to `None`.
        use_sliding_window_attention: bool. Whether to use sliding-window
            (local) attention. Defaults to `False`.
        sliding_window_size: int. Size of the sliding attention window.
            Defaults to `512`.
        layer_norm_epsilon: float. Epsilon value for RMS normalization.
            Defaults to `1e-6`.
        rope_wavelength: float. Base wavelength for rotary position embeddings.
            Defaults to `10000.0`.
        rope_scaling_factor: float. Scaling factor applied to RoPE frequencies.
            Defaults to `1.0`.
        rope_partial_rotary_factor: float. Fraction of head dimensions that
            receive rotary embeddings. Defaults to `1.0`.
        use_bidirectional_attention: bool. Whether to enable bidirectional
            (non-causal) attention. Defaults to `False`.
        use_vision_bidirectional_attention: bool. Whether to apply
            bidirectional attention to vision token positions only. Defaults
            to `False`.
        is_global_attention: bool. Whether this layer uses global (full-
            sequence) attention rather than sliding-window attention. Defaults
            to `False`.
        global_head_dim: int or `None`. Head dimensionality to use for global
            attention layers (may differ from `head_dim`). Defaults to
            `None`.
        dropout: float. Dropout rate applied after attention and FFW
            sub-layers. Defaults to `0`.
        attention_k_eq_v: bool. Whether key and value projections share
            weights. Defaults to `False`.
        num_global_key_value_heads: int or `None`. Number of key/value heads
            used in global attention layers. Defaults to `None`.
        enable_moe_block: bool. Whether to replace the dense FFW block with a
            Mixture-of-Experts block. Defaults to `False`.
        num_experts: int or `None`. Total number of experts when
            `enable_moe_block=True`. Defaults to `None`.
        expert_intermediate_dim: int or `None`. Intermediate dimension of each
            expert MLP when `enable_moe_block=True`. Defaults to `None`.
        num_experts_per_token: int. Number of experts activated per token
            when `enable_moe_block=True`. Defaults to `8`.
        is_text_layer: bool. Whether this block is a text (as opposed to
            vision) decoder layer; controls whether a non-trainable
            `layer_scalar` (or `encoder_layer_scalar`) is applied to the
            output. Defaults to `True`.
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        head_dim,
        num_query_heads,
        num_key_value_heads,
        logit_soft_cap=None,
        use_sliding_window_attention=False,
        sliding_window_size=512,
        layer_norm_epsilon=1e-6,
        rope_wavelength=10_000.0,
        rope_scaling_factor=1.0,
        rope_partial_rotary_factor=1.0,
        use_bidirectional_attention=False,
        use_vision_bidirectional_attention=False,
        is_global_attention=False,
        global_head_dim=None,
        dropout=0,
        attention_k_eq_v=False,
        num_global_key_value_heads=None,
        enable_moe_block=False,
        num_experts=None,
        expert_intermediate_dim=None,
        num_experts_per_token=8,
        is_text_layer=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.head_dim = head_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.logit_soft_cap = logit_soft_cap
        self.use_sliding_window_attention = use_sliding_window_attention
        self.sliding_window_size = sliding_window_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.rope_wavelength = rope_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.rope_partial_rotary_factor = rope_partial_rotary_factor
        self.use_bidirectional_attention = use_bidirectional_attention
        self.use_vision_bidirectional_attention = (
            use_vision_bidirectional_attention
        )
        self.is_global_attention = is_global_attention
        self.global_head_dim = global_head_dim
        self.dropout = dropout
        self.attention_k_eq_v = attention_k_eq_v
        self.num_global_key_value_heads = num_global_key_value_heads
        self.enable_moe_block = enable_moe_block
        self.num_experts = num_experts
        self.expert_intermediate_dim = expert_intermediate_dim
        self.num_experts_per_token = num_experts_per_token
        self.is_text_layer = is_text_layer

        self.pre_attention_norm = RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="pre_attention_norm",
        )
        self.post_attention_norm = RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="post_attention_norm",
        )

        effective_head_dim = (
            global_head_dim
            if is_global_attention and global_head_dim is not None
            else head_dim
        )
        attention_cls = (
            Gemma4TextAttention if self.is_text_layer else Gemma4VisionAttention
        )
        self.attention = attention_cls(
            head_dim=effective_head_dim,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            logit_soft_cap=logit_soft_cap,
            use_sliding_window_attention=use_sliding_window_attention,
            sliding_window_size=sliding_window_size,
            layer_norm_epsilon=layer_norm_epsilon,
            rope_wavelength=rope_wavelength,
            rope_scaling_factor=rope_scaling_factor,
            rope_partial_rotary_factor=rope_partial_rotary_factor,
            use_bidirectional_attention=use_bidirectional_attention,
            is_global_attention=is_global_attention,
            attention_k_eq_v=attention_k_eq_v,
            num_global_key_value_heads=num_global_key_value_heads,
            dropout=dropout,
            dtype=self.dtype_policy,
            name="attention",
        )

        if self.dropout > 0:
            self.attention_dropout = keras.layers.Dropout(rate=dropout)
            self.feedforward_dropout = keras.layers.Dropout(rate=dropout)

        self.pre_ffw_norm = RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="pre_ffw_norm",
        )
        self.post_ffw_norm = RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="post_ffw_norm",
        )

        self.gating_ffw = keras.layers.EinsumDense(
            equation="btd,df->btf",
            output_shape=(None, intermediate_dim),
            dtype=self.dtype_policy,
            name="ffw_gating",
        )
        self.gating_ffw_2 = keras.layers.EinsumDense(
            equation="btd,df->btf",
            output_shape=(None, intermediate_dim),
            dtype=self.dtype_policy,
            name="ffw_gating_2",
        )
        self.ffw_linear = keras.layers.EinsumDense(
            equation="btf,fd->btd",
            output_shape=(None, self.hidden_dim),
            dtype=self.dtype_policy,
            name="ffw_linear",
        )

        if enable_moe_block:
            assert num_experts is not None, (
                "`num_experts` must be set when `enable_moe_block=True`."
            )
            assert expert_intermediate_dim is not None, (
                "`expert_intermediate_dim` must be set when "
                "`enable_moe_block=True`."
            )
            self.pre_ffw_norm_moe = RMSNormalization(
                epsilon=self.layer_norm_epsilon,
                dtype=self.dtype_policy,
                name="pre_ffw_norm_moe",
            )
            self.post_ffw_norm_dense = RMSNormalization(
                epsilon=self.layer_norm_epsilon,
                dtype=self.dtype_policy,
                name="post_ffw_norm_dense",
            )
            self.post_ffw_norm_moe_path = RMSNormalization(
                epsilon=self.layer_norm_epsilon,
                dtype=self.dtype_policy,
                name="post_ffw_norm_moe_path",
            )
            self.moe_router = Gemma4Router(
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                layer_norm_epsilon=layer_norm_epsilon,
                dtype=self.dtype_policy,
                name="moe_router",
            )
            self.moe_expert_bank = Gemma4MoEBlock(
                num_experts=num_experts,
                hidden_dim=hidden_dim,
                expert_intermediate_dim=expert_intermediate_dim,
                dtype=self.dtype_policy,
                name="moe_expert_bank",
            )

    def build(self, input_shape):
        self.pre_attention_norm.build(input_shape)
        self.attention.build(input_shape)

        attn_out_shape, cache_shape = self.attention.compute_output_shape(
            input_shape
        )
        self.post_attention_norm.build(attn_out_shape)

        self.pre_ffw_norm.build(input_shape)
        self.gating_ffw.build(input_shape)
        self.gating_ffw_2.build(input_shape)

        ffn_shape = self.gating_ffw.compute_output_shape(input_shape)
        self.ffw_linear.build(ffn_shape)

        ffw_out_shape = self.ffw_linear.compute_output_shape(ffn_shape)
        self.post_ffw_norm.build(ffw_out_shape)

        if self.enable_moe_block:
            self.pre_ffw_norm_moe.build(input_shape)
            self.post_ffw_norm_dense.build(input_shape)
            self.post_ffw_norm_moe_path.build(input_shape)
            self.moe_router.build(input_shape)
            self.moe_expert_bank.build(input_shape)

        if self.is_text_layer:
            # Decoder-pass scalar (matches Gemma4 layer_scalar).
            self.layer_scalar = self.add_weight(
                name="layer_scalar",
                shape=(),
                initializer="ones",
                trainable=False,
            )
            # Encoder-pass scalar (causal prompt encoding).
            self.encoder_layer_scalar = self.add_weight(
                name="encoder_layer_scalar",
                shape=(),
                initializer="ones",
                trainable=False,
            )

        self.built = True

    def _compute_image_bidirectional_attention_mask(self, vision_mask):
        """Allow image tokens to attend to each other within the same image."""
        bidirectional_mask = vision_mask

        padded_mask = ops.cast(
            ops.pad(bidirectional_mask, [(0, 0), (1, 0)], constant_values=0),
            dtype="int32",
        )

        boundary = ops.cast(
            ops.greater(padded_mask[..., 1:], padded_mask[..., :-1]),
            dtype="int32",
        )
        numbered_boundary = ops.cumsum(boundary, -1)
        indices = ops.multiply(bidirectional_mask, numbered_boundary)

        indices_expanded_1 = ops.expand_dims(indices, 1)
        indices_expanded_2 = ops.expand_dims(indices, -1)

        mask = ops.logical_and(
            ops.equal(indices_expanded_1, indices_expanded_2),
            indices_expanded_2,
        )
        return mask

    def _compute_attention_mask(
        self,
        x,
        padding_mask,
        vision_mask,
        cache,
        cache_update_index,
    ):
        decoder_mask = merge_padding_and_attention_mask(
            inputs=x, padding_mask=padding_mask, attention_mask=None
        )

        batch_size = ops.shape(x)[0]
        input_length = output_length = ops.shape(x)[1]
        if cache is not None:
            input_length = ops.shape(cache)[2]

        if self.use_bidirectional_attention:
            if decoder_mask is None:
                return None
            mask_1 = decoder_mask
            mask_2 = ops.transpose(mask_1, (0, 2, 1))
            return mask_1 * mask_2

        causal_mask = compute_causal_mask(
            batch_size=batch_size,
            input_length=input_length,
            output_length=output_length,
            cache_index=cache_update_index,
        )

        if self.use_sliding_window_attention and not self.is_global_attention:
            causal_mask = self.attention._mask_sliding_window(
                causal_mask,
                cache_update_index=cache_update_index,
            )

        if (
            vision_mask is not None
            and cache is None
            and self.use_vision_bidirectional_attention
            and not self.is_global_attention
        ):
            bidirectional_image_mask = (
                self._compute_image_bidirectional_attention_mask(vision_mask)
            )
            causal_mask = ops.logical_or(causal_mask, bidirectional_image_mask)

        if decoder_mask is not None:
            causal_mask = ops.minimum(decoder_mask, causal_mask)

        return causal_mask

    def _compute_canvas_bidirectional_attention_mask(
        self, canvas_mask, cache_update_index, output_length, input_length
    ):
        """Return a bool mask that lets canvas queries attend to canvas keys.

        Canvas key positions are `[cache_update_index,
        cache_update_index + output_length)` within the KV cache of length
        `input_length`.  Canvas query positions are marked by `canvas_mask`.

        Args:
            canvas_mask: bool tensor `(B, output_length)`.
            cache_update_index: int or scalar — index of the first canvas key
                in the KV cache.
            output_length: int or scalar — number of canvas tokens.
            input_length: int or scalar — total KV cache sequence length.

        Returns:
            Bool tensor `(B, output_length, input_length)`.
        """
        batch_size = ops.shape(canvas_mask)[0]
        j = ops.arange(input_length)
        canvas_key = ops.logical_and(
            j >= cache_update_index,
            j < cache_update_index + output_length,
        )
        canvas_key = ops.broadcast_to(
            ops.reshape(canvas_key, (1, 1, input_length)),
            (batch_size, output_length, input_length),
        )
        q = ops.broadcast_to(
            ops.reshape(
                ops.cast(canvas_mask, "bool"), (batch_size, output_length, 1)
            ),
            (batch_size, output_length, input_length),
        )
        return ops.logical_and(canvas_key, q)

    def call(
        self,
        x,
        padding_mask=None,
        vision_mask=None,
        cache=None,
        cache_update_index=0,
        cache_update_mask=None,
        positions=None,
        canvas_mask=None,
        is_encoder=False,
    ):
        # Clamp float16 to avoid overflow.
        is_float16 = keras.backend.standardize_dtype(x.dtype) == "float16"
        if is_float16:
            x = ops.clip(x, -65504, 65504)

        # === Attention sub-block ===
        residual = x
        normalized_x = self.pre_attention_norm(x)
        attention_mask = self._compute_attention_mask(
            normalized_x, padding_mask, vision_mask, cache, cache_update_index
        )

        # Canvas bidirectional mask: all canvas queries attend to all canvas
        # keys, overriding the causal restriction for those positions.
        if canvas_mask is not None and cache is not None:
            output_length = ops.shape(normalized_x)[1]
            input_length = ops.shape(cache)[2]
            canvas_bidirec = self._compute_canvas_bidirectional_attention_mask(
                canvas_mask, cache_update_index, output_length, input_length
            )
            if attention_mask is not None:
                attention_mask = ops.logical_or(attention_mask, canvas_bidirec)
            else:
                attention_mask = canvas_bidirec

        if cache is not None:
            attention, new_cache = self.attention(
                normalized_x,
                attention_mask=attention_mask,
                cache=cache,
                cache_update_index=cache_update_index,
                cache_update_mask=cache_update_mask,
                positions=positions,
            )
        else:
            attention, new_cache = self.attention(
                normalized_x,
                attention_mask=attention_mask,
                positions=positions,
            )

        attention = self.post_attention_norm(attention)

        if self.dropout:
            attention = self.attention_dropout(attention)

        if is_float16:
            x = ops.cast(
                ops.clip(
                    ops.add(
                        ops.cast(residual, "float32"),
                        ops.cast(attention, "float32"),
                    ),
                    -65504,
                    65504,
                ),
                "float16",
            )
        else:
            x = residual + attention

        # === Feed-forward sub-block ===
        residual = x

        if self.enable_moe_block:
            normalized_x = self.pre_ffw_norm(x)
            x1 = ops.matmul(normalized_x, self.gating_ffw.kernel)
            x2 = ops.matmul(normalized_x, self.gating_ffw_2.kernel)
            dense_out = keras.activations.gelu(x1, approximate=True) * x2
            dense_out = ops.matmul(dense_out, self.ffw_linear.kernel)
            dense_out = self.post_ffw_norm_dense(dense_out)

            dispatch_weights = self.moe_router(x)
            moe_in = self.pre_ffw_norm_moe(x)
            shape = ops.shape(moe_in)
            moe_in_flat = ops.reshape(moe_in, (-1, shape[-1]))
            moe_out = self.moe_expert_bank(moe_in_flat, dispatch_weights)
            moe_out = ops.reshape(moe_out, shape)
            moe_out = self.post_ffw_norm_moe_path(moe_out)

            x = dense_out + moe_out
        else:
            normalized_x = self.pre_ffw_norm(x)
            x1 = ops.matmul(normalized_x, self.gating_ffw.kernel)
            x2 = ops.matmul(normalized_x, self.gating_ffw_2.kernel)

            x = keras.activations.gelu(x1, approximate=True) * x2
            x = ops.matmul(x, self.ffw_linear.kernel)

        x = self.post_ffw_norm(x)

        if self.dropout:
            x = self.feedforward_dropout(x)

        if is_float16:
            x = ops.cast(
                ops.clip(
                    ops.add(
                        ops.cast(residual, "float32"), ops.cast(x, "float32")
                    ),
                    -65504,
                    65504,
                ),
                "float16",
            )
        else:
            x = residual + x

        # Scale by encoder or decoder scalar depending on the pass type.
        if self.is_text_layer:
            scalar = (
                self.encoder_layer_scalar if is_encoder else self.layer_scalar
            )
            x = x * ops.cast(scalar, x.dtype)

        return x, new_cache

    def compute_output_shape(self, input_shape):
        attn_out_shape, cache_shape = self.attention.compute_output_shape(
            input_shape
        )
        return input_shape, cache_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "head_dim": self.head_dim,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "logit_soft_cap": self.logit_soft_cap,
                "use_sliding_window_attention": (
                    self.use_sliding_window_attention
                ),
                "sliding_window_size": self.sliding_window_size,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "rope_wavelength": self.rope_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
                "rope_partial_rotary_factor": self.rope_partial_rotary_factor,
                "use_bidirectional_attention": self.use_bidirectional_attention,
                "use_vision_bidirectional_attention": (
                    self.use_vision_bidirectional_attention
                ),
                "is_global_attention": self.is_global_attention,
                "global_head_dim": self.global_head_dim,
                "attention_k_eq_v": self.attention_k_eq_v,
                "num_global_key_value_heads": self.num_global_key_value_heads,
                "enable_moe_block": self.enable_moe_block,
                "num_experts": self.num_experts,
                "expert_intermediate_dim": self.expert_intermediate_dim,
                "num_experts_per_token": self.num_experts_per_token,
                "is_text_layer": self.is_text_layer,
            }
        )
        return config
