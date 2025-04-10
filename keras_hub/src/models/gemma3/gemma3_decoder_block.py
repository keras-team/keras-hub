import keras
from keras import ops

from keras_hub.src.layers.modeling.transformer_layer_utils import (
    compute_causal_mask,
)
from keras_hub.src.layers.modeling.transformer_layer_utils import (
    merge_padding_and_attention_mask,
)
from keras_hub.src.models.gemma3.gemma3_attention import CachedGemma3Attention
from keras_hub.src.models.gemma3.rms_normalization import RMSNormalization


class Gemma3DecoderBlock(keras.layers.Layer):
    """Transformer decoder layer for Gemma3.

    This decoder layer is the same as the layer used for Gemma and Gemma2.
    However, there are a few key differences. Firstly, image tokens have
    bidirectional masking. Additionally, this layer exposes the following args:

    `use_query_key_norm`: bool. If True, apply RMS normalization on query
        and key. For Gemma3, this is True.
    `rope_wavelength`: float. Configurable value for RoPE wavelength. Gemma3
        uses 10K for local attention layers and 1M for global attention layers.
    `gate_dim_reduction`: int. In the gating layers, the output dimension is
        `intermediate_dim // gate_dim_reduction`. For Gemma and Gemma2, this
        value is 2. For Gemma3, it is 1.
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        head_dim,
        num_query_heads,
        num_key_value_heads,
        query_head_dim_normalize=True,
        use_query_key_norm=False,
        use_post_ffw_norm=False,
        use_post_attention_norm=False,
        gate_dim_reduction=2,
        logit_soft_cap=None,
        use_sliding_window_attention=False,
        sliding_window_size=4096,
        layer_norm_epsilon=1e-6,
        rope_wavelength=10_000.0,
        rope_scaling_factor=1.0,
        dropout=0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.head_dim = head_dim
        self.num_query_heads = num_query_heads
        self.num_key_value_heads = num_key_value_heads
        self.query_head_dim_normalize = query_head_dim_normalize
        self.use_query_key_norm = use_query_key_norm
        self.use_post_ffw_norm = use_post_ffw_norm
        self.use_post_attention_norm = use_post_attention_norm
        self.gate_dim_reduction = gate_dim_reduction
        self.logit_soft_cap = logit_soft_cap
        self.use_sliding_window_attention = use_sliding_window_attention
        self.sliding_window_size = sliding_window_size
        self.layer_norm_epsilon = layer_norm_epsilon
        self.rope_wavelength = rope_wavelength
        self.rope_scaling_factor = rope_scaling_factor
        self.dropout = dropout

        self.pre_attention_norm = RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="pre_attention_norm",
        )

        if use_post_attention_norm:
            self.post_attention_norm = RMSNormalization(
                epsilon=self.layer_norm_epsilon,
                dtype=self.dtype_policy,
                name="post_attention_norm",
            )

        self.attention = CachedGemma3Attention(
            head_dim=head_dim,
            num_query_heads=num_query_heads,
            num_key_value_heads=num_key_value_heads,
            use_query_key_norm=use_query_key_norm,
            logit_soft_cap=logit_soft_cap,
            use_sliding_window_attention=use_sliding_window_attention,
            sliding_window_size=sliding_window_size,
            query_head_dim_normalize=True,
            rope_wavelength=rope_wavelength,
            rope_scaling_factor=rope_scaling_factor,
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

        if use_post_ffw_norm:
            self.post_ffw_norm = RMSNormalization(
                epsilon=self.layer_norm_epsilon,
                dtype=self.dtype_policy,
                name="post_ffw_norm",
            )

        self.gating_ffw = keras.layers.EinsumDense(
            equation="btd,df->btf",
            output_shape=(None, self.intermediate_dim // gate_dim_reduction),
            dtype=self.dtype_policy,
            name="ffw_gating",
        )

        self.gating_ffw_2 = keras.layers.EinsumDense(
            equation="btd,df->btf",
            output_shape=(None, self.intermediate_dim // gate_dim_reduction),
            dtype=self.dtype_policy,
            name="ffw_gating_2",
        )

        self.ffw_linear = keras.layers.EinsumDense(
            equation="btf,fd->btd",
            output_shape=(None, self.hidden_dim),
            dtype=self.dtype_policy,
            name="ffw_linear",
        )

    def build(self, input_shape):
        self.pre_attention_norm.build(input_shape)
        self.attention.build(input_shape)

        if self.use_post_attention_norm:
            shape = self.attention.compute_output_shape(input_shape)
            self.post_attention_norm.build(shape)

        shape = input_shape
        self.pre_ffw_norm.build(shape)
        self.gating_ffw.build(shape)
        self.gating_ffw_2.build(shape)

        shape = self.gating_ffw.compute_output_shape(shape)
        self.ffw_linear.build(shape)

        if self.use_post_ffw_norm:
            shape = self.ffw_linear.compute_output_shape(shape)
            self.post_ffw_norm.build(shape)

        self.built = True

    def compute_output_shape(self, input_shape):
        # Isometric
        return input_shape

    def _compute_image_bidirectional_attention_mask(self, vision_mask):
        # vision_mask is False for text, True for images. Shape of
        # (bsz, seq_len).
        bidirectional_mask = vision_mask

        # Left pad with 0.
        padded_mask = ops.cast(
            ops.pad(bidirectional_mask, [(0, 0), (1, 0)], constant_values=0),
            dtype="int32",
        )

        # Assign unique indices to every contiguous span of True.
        boundary = ops.cast(
            ops.greater(padded_mask[..., 1:], padded_mask[..., :-1]),
            dtype="int32",
        )
        numbered_boundary = ops.cumsum(boundary, -1)
        indices = ops.multiply(bidirectional_mask, numbered_boundary)

        indices_expanded_1 = ops.expand_dims(indices, 1)
        indices_expanded_2 = ops.expand_dims(indices, -1)

        mask = ops.logical_and(
            ops.equal(
                indices_expanded_1,
                indices_expanded_2,
            ),
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

        causal_mask = compute_causal_mask(
            batch_size=batch_size,
            input_length=input_length,
            output_length=output_length,
            cache_index=cache_update_index,
        )

        # Compute bidirectional mask (image tokens can attend to each other
        # in both directions, within the same image).
        if vision_mask is not None:
            bidirectional_image_mask = (
                self._compute_image_bidirectional_attention_mask(vision_mask)
            )
            causal_mask = ops.logical_or(causal_mask, bidirectional_image_mask)

        # Respect the padding mask.
        if decoder_mask is not None:
            causal_mask = ops.minimum(decoder_mask, causal_mask)

        return causal_mask

    def call(
        self,
        x,
        padding_mask=None,
        vision_mask=None,
        cache=None,
        cache_update_index=0,
        cache_update_mask=None,
    ):
        # Note: `vision_mask` is used only for Gemma3.
        normalized_x = self.pre_attention_norm(x)
        attention_mask = self._compute_attention_mask(
            normalized_x, padding_mask, vision_mask, cache, cache_update_index
        )
        if cache is not None:
            attention, new_cache = self.attention(
                normalized_x,
                attention_mask=attention_mask,
                cache=cache,
                cache_update_index=cache_update_index,
                cache_update_mask=cache_update_mask,
            )
        else:
            attention = self.attention(
                normalized_x,
                attention_mask=attention_mask,
            )

        if self.use_post_attention_norm:
            attention = self.post_attention_norm(attention)

        if self.dropout:
            attention = self.attention_dropout(attention)

        attention_x = x + attention
        normalized_x = self.pre_ffw_norm(attention_x)

        x1 = self.gating_ffw(normalized_x)
        x2 = self.gating_ffw_2(normalized_x)
        x = keras.activations.gelu(x1, approximate=True) * x2
        x = self.ffw_linear(x)

        if self.use_post_ffw_norm:
            x = self.post_ffw_norm(x)

        x = x + attention_x

        if cache is not None:
            return x, new_cache
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "head_dim": self.head_dim,
                "num_query_heads": self.num_query_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "query_head_dim_normalize": self.query_head_dim_normalize,
                "use_query_key_norm": self.use_query_key_norm,
                "use_post_ffw_norm": self.use_post_ffw_norm,
                "use_post_attention_norm": self.use_post_attention_norm,
                "gate_dim_reduction": self.gate_dim_reduction,
                "logit_soft_cap": self.logit_soft_cap,
                "use_sliding_window_attention": (
                    self.use_sliding_window_attention
                ),
                "sliding_window_size": self.sliding_window_size,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "dropout": self.dropout,
                "rope_wavelength": self.rope_wavelength,
                "rope_scaling_factor": self.rope_scaling_factor,
            }
        )
        return config
