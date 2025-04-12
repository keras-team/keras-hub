import keras

from keras_hub.src.layers.modeling.transformer_decoder import TransformerDecoder
from keras_hub.src.models.moonshine.moonshine_layers import MoonshineMLP
from keras_hub.src.models.moonshine.moonshine_layers import (
    moonshine_kernel_initializer,
)
from keras_hub.src.models.moonshine.moonshine_multi_head_attention import (
    MoonshineMultiHeadAttention,
)
from keras_hub.src.utils.keras_utils import clone_initializer
from keras import ops


@keras.saving.register_keras_serializable(package="keras_hub")
class MoonshineDecoderBlock(TransformerDecoder):
    """Moonshine decoder block for sequence processing.

    This layer implements a decoder block that includes self-attention with
    causal masking, cross-attention with precomputed key/value pairs, and a
    feedforward network. It supports both cached and uncached operation modes.

    Args:
        hidden_dim: int. The dimensionality of the model's hidden
            representations.
        intermediate_dim: int. The dimensionality of the intermediate
            representations in the feedforward network.
        num_heads: int. The number of attention heads for multi-head attention
            mechanisms.
        feedforward_expansion_factor: int, optional. A multiplicative factor for
            scaling the feedforward network dimension. Defaults to 4.
        use_swiglu_activation: bool, optional. Whether to use the SwiGLU
            activation in the feedforward network for improved performance.
            Defaults to True.
        pad_head_dim_to_multiple_of: int, optional. If specified, pads the head
            dimension to be a multiple of this value for performance
            optimization. Defaults to None.
        initializer_range: float, optional. The standard deviation of the
            truncated normal distribution used to initialize model weights.
            Defaults to 0.02.
        attention_bias: bool, optional. Whether to add a bias term to the
            attention computations. Defaults to False.
        attention_dropout: float, optional. The dropout rate applied to
            attention weights during training. Defaults to 0.0.
        dtype: str, optional. The data type to use for model computations and
            weights. Defaults to None.
        **kwargs: Additional keyword arguments passed to the base layer.
    """

    # References:
    # Defined and formulated based on the UsefulSensors implementation of the
    # DecoderLayer class (https://github.com/usefulsensors/moonshine/blob/4a000427bd36a1c2c6d20a86c672dbd850b44c88/moonshine/model.py#L348-L466).

    # <<< START ADDITION >>>
    _debug_print_count = 0
    _max_debug_prints = 5
    # <<< END ADDITION >>>

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_heads,
        feedforward_expansion_factor=4,
        use_swiglu_activation=True,
        pad_head_dim_to_multiple_of=None,
        initializer_range=0.02,
        attention_bias=False,
        attention_dropout=0.0,
        dtype=None,
        **kwargs,
    ):
        kwargs.pop("dropout", None)
        kwargs.pop("activation", None)
        kwargs.pop("kernel_initializer", None)
        self.kernel_initializer = moonshine_kernel_initializer(
            initializer_range=initializer_range
        )
        super().__init__(
            intermediate_dim=intermediate_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            activation="gelu" if use_swiglu_activation else "silu",
            kernel_initializer=clone_initializer(self.kernel_initializer),
            dtype=dtype,
            **kwargs,
        )
        self.initializer_range = initializer_range
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.feedforward_expansion_factor = feedforward_expansion_factor
        self.use_swiglu_activation = use_swiglu_activation
        self.pad_head_dim_to_multiple_of = pad_head_dim_to_multiple_of
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias

        self.head_dim = hidden_dim // num_heads
        if pad_head_dim_to_multiple_of is not None:
            self.head_dim = (
                (self.head_dim + pad_head_dim_to_multiple_of - 1)
                // pad_head_dim_to_multiple_of
            ) * pad_head_dim_to_multiple_of

        self.norm1 = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            center=False,
            scale=True,
            dtype=self.dtype,
        )
        self.self_attention = MoonshineMultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.head_dim,
            use_bias=False,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            use_causal_mask=True,
            apply_rotary_embedding=True,
            cache_mode="autoregressive",
            dtype=self.dtype,
        )
        self.norm2 = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            center=False,
            scale=True,
            dtype=self.dtype,
        )
        self.cross_attention = MoonshineMultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.head_dim,
            use_bias=False,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            use_causal_mask=False,
            apply_rotary_embedding=False,
            cache_mode="precomputed",
            dtype=self.dtype,
        )
        self.norm3 = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            center=False,
            scale=True,
            dtype=self.dtype,
        )
        self.ff = MoonshineMLP(
            hidden_dim=hidden_dim,
            feedforward_expansion_factor=feedforward_expansion_factor,
            use_swiglu_activation=use_swiglu_activation,
            initializer_range=initializer_range,
            dtype=self.dtype,
        )

    def build(self, input_shape):
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) < 2:
            raise ValueError(
                "Expected input_shape to be a list of at least two shapes."
            )
        decoder_sequence_shape = (
            input_shape[0]["decoder_token_ids"]  # Shape of x
            if isinstance(input_shape[0], dict)
            else input_shape[0]
        )
        context_shape = (
            input_shape[1]["input_values"]  # Shape of context
            if isinstance(input_shape[1], dict)
            else input_shape[1]
        )

        # Build sublayers.
        self.norm1.build(decoder_sequence_shape)
        self.norm2.build(decoder_sequence_shape)
        self.norm3.build(decoder_sequence_shape)

        self.self_attention.build(
            query_shape=decoder_sequence_shape,
            key_shape=decoder_sequence_shape,
            value_shape=decoder_sequence_shape,
        )

        self.cross_attention.build(
            query_shape=decoder_sequence_shape,
            key_shape=context_shape,
            value_shape=context_shape,
        )

        self.ff.build(decoder_sequence_shape)
        self.built = True

    def compute_output_spec(
        self,
        inputs,
        training=None,
        use_cache=False,
        decoder_attention_mask=None,
        encoder_attention_mask=None,
    ):
        if use_cache:
            # Cached case: expect 7 inputs.
            if len(inputs) != 7:
                raise ValueError(
                    "When use_cache=True, expected 7 inputs: "
                    "[x, context, cache_k, cache_v, x_attn_cache_k, "
                    "x_attn_cache_v, rotary_embedding]"
                )
            (
                x,
                context,
                cache_k,
                cache_v,
                x_attn_cache_k,
                x_attn_cache_v,
                rotary_embedding,
            ) = inputs
            # Output shape for x is the same as input x_shape but with
            # hidden_dim.
            x_shape = x.shape if hasattr(x, "shape") else x
            output_shape = x_shape[:-1] + (self.hidden_dim,)
            # New cache shapes are the same as input cache_k_shape and
            # cache_v_shape.
            # Note: In practice, sequence length may increase due to
            # concatenation, but symbolically, it remains None.
            new_cache_shape = (
                cache_k.shape if hasattr(cache_k, "shape") else cache_k
            )
            return (
                keras.KerasTensor(shape=output_shape, dtype=self.dtype),  # x
                keras.KerasTensor(
                    shape=new_cache_shape, dtype=self.dtype
                ),  # new_cache_k
                keras.KerasTensor(
                    shape=new_cache_shape, dtype=self.dtype
                ),  # new_cache_v
            )
        else:
            # Uncached case: expect 3 inputs.
            if len(inputs) != 3:
                raise ValueError(
                    "When use_cache=False, expected 3 inputs: [x, context, "
                    "rotary_embedding]"
                )
            x, context, rotary_embedding = inputs
            x_shape = x.shape if hasattr(x, "shape") else x
            context_shape = (
                context.shape if hasattr(context, "shape") else context
            )
            batch_size = x_shape[0]  # None (symbolic)
            seq_len = x_shape[1]  # None (symbolic)
            context_len = context_shape[1]  # None (symbolic)
            hidden_dim = self.hidden_dim
            num_heads = self.num_heads
            head_dim = self.head_dim

            # Define output shapes.
            output_shape = (batch_size, seq_len, hidden_dim)  # x
            cache_shape_self = (
                batch_size,
                seq_len,
                num_heads,
                head_dim,
            )  # Self-attention caches
            cache_shape_cross = (
                batch_size,
                context_len,
                num_heads,
                head_dim,
            )  # Cross-attention caches

            return (
                keras.KerasTensor(shape=output_shape, dtype=self.dtype),  # x
                keras.KerasTensor(
                    shape=cache_shape_self, dtype=self.dtype
                ),  # cache_k
                keras.KerasTensor(
                    shape=cache_shape_self, dtype=self.dtype
                ),  # cache_v
                keras.KerasTensor(
                    shape=cache_shape_cross, dtype=self.dtype
                ),  # x_attn_cache_k
                keras.KerasTensor(
                    shape=cache_shape_cross, dtype=self.dtype
                ),  # x_attn_cache_v
            )

    def call(
        self,
        inputs,
        training=None,
        use_cache=False,
        decoder_attention_mask=None,
        encoder_attention_mask=None,
        self_attention_cache=None,
        self_attention_cache_update_index=None,
    ):
        if use_cache:
            # <<< START ADDITION >>>
            if not isinstance(inputs, (list, tuple)) or len(inputs) != 7:
                 raise ValueError(
                     "When use_cache=True, expected 7 inputs: "
                     "[x, context, cache_k, cache_v, x_attn_cache_k, "
                     "x_attn_cache_v, rotary_embedding]. "
                     f"Received {len(inputs)} inputs."
                 )
            (
                x,
                context,
                cache_k, # Self-attn key cache
                cache_v, # Self-attn value cache
                x_attn_cache_k, # Cross-attn key cache (precomputed)
                x_attn_cache_v, # Cross-attn value cache (precomputed)
                rotary_embedding,
            ) = inputs
            # <<< END ADDITION >>>
        else:
            if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
                 raise ValueError(
                     "When use_cache=False, expected 3 inputs: [x, context, "
                     f"rotary_embedding]. Received {len(inputs)} inputs."
                 )
            x, context, rotary_embedding = inputs
            cache_k, cache_v, x_attn_cache_k, x_attn_cache_v = None, None, None, None

        # <<< START ADDITION >>>
        # <<< START MODIFICATION >>>
        should_print_details = False
        if use_cache and ops.shape(x)[1] == 1 and MoonshineDecoderBlock._debug_print_count < MoonshineDecoderBlock._max_debug_prints:
            should_print_details = True
            MoonshineDecoderBlock._debug_print_count += 1
        # <<< END MODIFICATION >>>

        if should_print_details:
            print_index = self_attention_cache_update_index if self_attention_cache_update_index is not None else "N/A"
            print(f"--- [DecoderBlock {self.name}] ENTERED (use_cache=True, seq_len=1) Index: {print_index} ---")
            try:
                print(f"[DecoderBlock {self.name}] Input x shape: {ops.shape(x)}")
                print(f"[DecoderBlock {self.name}] Input x mean: {ops.mean(x):.4f}, max: {ops.max(x):.4f}, min: {ops.min(x):.4f}")
                print(f"[DecoderBlock {self.name}] Input context shape: {ops.shape(context)}")
                if isinstance(rotary_embedding, tuple):
                     print(f"[DecoderBlock {self.name}] Input rotary_embedding[0] (cos) shape: {ops.shape(rotary_embedding[0])}")
                else:
                     print(f"[DecoderBlock {self.name}] Input rotary_embedding shape: {ops.shape(rotary_embedding)}")
                if decoder_attention_mask is not None: print(f"[DecoderBlock {self.name}] Input decoder_attention_mask shape: {ops.shape(decoder_attention_mask)}")
                if encoder_attention_mask is not None: print(f"[DecoderBlock {self.name}] Input encoder_attention_mask shape: {ops.shape(encoder_attention_mask)}")
                if cache_k is not None: print(f"[DecoderBlock {self.name}] Input cache_k shape: {ops.shape(cache_k)}")
                if cache_v is not None: print(f"[DecoderBlock {self.name}] Input cache_v shape: {ops.shape(cache_v)}")
                if x_attn_cache_k is not None: print(f"[DecoderBlock {self.name}] Input x_attn_cache_k shape: {ops.shape(x_attn_cache_k)}")
                if x_attn_cache_v is not None: print(f"[DecoderBlock {self.name}] Input x_attn_cache_v shape: {ops.shape(x_attn_cache_v)}")
            except Exception as e:
                print(f"[DecoderBlock {self.name}] Error printing inputs: {e}")
        # <<< END ADDITION >>>

        residual = x
        x_norm1 = self.norm1(x) # <<< NOTE: Renamed from 'x' to 'x_norm1' for clarity

        # <<< START ADDITION >>>
        if should_print_details:
            try:
                print(f"--- [DecoderBlock {self.name}] Before Self-Attention ---")
                print(f"[SelfAttn] Input x_norm1 shape: {ops.shape(x_norm1)}")
                print(f"[SelfAttn] Input x_norm1 mean: {ops.mean(x_norm1):.4f}, max: {ops.max(x_norm1):.4f}, min: {ops.min(x_norm1):.4f}")
            except Exception as e: print(f"[SelfAttn] Error printing before self-attn: {e}")
        # <<< END ADDITION >>>

        if use_cache:
            x_self_attn, new_cache_k, new_cache_v = self.self_attention(
                query=x_norm1,
                key=x_norm1,
                value=x_norm1,
                rotary_embedding=rotary_embedding,
                key_cache=cache_k,
                value_cache=cache_v,
                attention_mask=decoder_attention_mask,
                training=training,
            )
        else:
            x_self_attn, cache_k, cache_v = self.self_attention(
                query=x_norm1,
                key=x_norm1,
                value=x_norm1,
                rotary_embedding=rotary_embedding,
                attention_mask=decoder_attention_mask,
                training=training,
            )
        x = x_self_attn + residual

        # <<< START ADDITION >>>
        if should_print_details:
            try:
                print(f"--- [DecoderBlock {self.name}] After Self-Attention ---")
                print(f"[SelfAttn] Output x shape: {ops.shape(x)}")
                print(f"[SelfAttn] Output x mean: {ops.mean(x):.4f}, max: {ops.max(x):.4f}, min: {ops.min(x):.4f}")
                if use_cache:
                    if new_cache_k is not None: print(f"[SelfAttn] Output new_cache_k shape: {ops.shape(new_cache_k)}")
                    if new_cache_v is not None: print(f"[SelfAttn] Output new_cache_v shape: {ops.shape(new_cache_v)}")
                else:
                    if cache_k is not None: print(f"[SelfAttn] Output cache_k shape: {ops.shape(cache_k)}")
                    if cache_v is not None: print(f"[SelfAttn] Output cache_v shape: {ops.shape(cache_v)}")
            except Exception as e: print(f"[SelfAttn] Error printing after self-attn: {e}")
        # <<< END ADDITION >>>

        residual = x
        x_norm2 = self.norm2(x) # <<< NOTE: Renamed from 'x' to 'x_norm2'

        # <<< START ADDITION >>>
        if should_print_details:
            try:
                print(f"--- [DecoderBlock {self.name}] Before Cross-Attention ---")
                print(f"[CrossAttn] Input x_norm2 shape: {ops.shape(x_norm2)}")
                print(f"[CrossAttn] Input x_norm2 mean: {ops.mean(x_norm2):.4f}, max: {ops.max(x_norm2):.4f}, min: {ops.min(x_norm2):.4f}")
                print(f"[CrossAttn] Input context shape: {ops.shape(context)}")
            except Exception as e: print(f"[CrossAttn] Error printing before cross-attn: {e}")
        # <<< END ADDITION >>>

        if use_cache:
            x_cross_attn = self.cross_attention(
                query=x_norm2,
                key=context,
                value=context,
                key_cache=x_attn_cache_k,
                value_cache=x_attn_cache_v,
                attention_mask=encoder_attention_mask,
                training=training,
            )
        else:
            x_cross_attn, x_attn_cache_k, x_attn_cache_v = self.cross_attention( # <<< Renamed output var
                query=x_norm2,
                key=context,
                value=context,
                attention_mask=encoder_attention_mask,
                training=training,
            )
        x = x_cross_attn + residual

        # <<< START ADDITION >>>
        if should_print_details:
            try:
                print(f"--- [DecoderBlock {self.name}] After Cross-Attention ---")
                print(f"[CrossAttn] Output x shape: {ops.shape(x)}")
                print(f"[CrossAttn] Output x mean: {ops.mean(x):.4f}, max: {ops.max(x):.4f}, min: {ops.min(x):.4f}")
                if not use_cache:
                    if x_attn_cache_k is not None: print(f"[CrossAttn] Output x_attn_cache_k shape: {ops.shape(x_attn_cache_k)}")
                    if x_attn_cache_v is not None: print(f"[CrossAttn] Output x_attn_cache_v shape: {ops.shape(x_attn_cache_v)}")
            except Exception as e: print(f"[CrossAttn] Error printing after cross-attn: {e}")
        # <<< END ADDITION >>>

        residual = x
        x_norm3 = self.norm3(x) # <<< NOTE: Renamed from 'x' to 'x_norm3'

        # <<< START ADDITION >>>
        if should_print_details:
            try:
                print(f"--- [DecoderBlock {self.name}] Before Feedforward ---")
                print(f"[FF] Input x_norm3 shape: {ops.shape(x_norm3)}")
                print(f"[FF] Input x_norm3 mean: {ops.mean(x_norm3):.4f}, max: {ops.max(x_norm3):.4f}, min: {ops.min(x_norm3):.4f}")
            except Exception as e: print(f"[FF] Error printing before ff: {e}")
        # <<< END ADDITION >>>

        x_ff = self.ff(x_norm3)
        x = x_ff + residual

        # <<< START ADDITION >>>
        if should_print_details:
            try:
                print(f"--- [DecoderBlock {self.name}] After Feedforward ---")
                print(f"[FF] Output x shape: {ops.shape(x)}")
                print(f"[FF] Output x mean: {ops.mean(x):.4f}, max: {ops.max(x):.4f}, min: {ops.min(x):.4f}")
                print_index = self_attention_cache_update_index if self_attention_cache_update_index is not None else "N/A"
                # <<< START MODIFICATION >>>
                print(f"--- [DecoderBlock {self.name}] EXITED (Print {MoonshineDecoderBlock._debug_print_count}/{MoonshineDecoderBlock._max_debug_prints}, use_cache=True, seq_len=1) Index: {print_index} ---")
                # <<< END MODIFICATION >>>
            except Exception as e: print(f"[FF] Error printing after ff: {e}")
        # <<< END ADDITION >>>
        if use_cache:
            return x, new_cache_k, new_cache_v
        return x, cache_k, cache_v, x_attn_cache_k, x_attn_cache_v

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "feedforward_expansion_factor": self.feedforward_expansion_factor,  # noqa: E501
                "use_swiglu_activation": self.use_swiglu_activation,
                "pad_head_dim_to_multiple_of": self.pad_head_dim_to_multiple_of,  # noqa: E501
                "initializer_range": self.initializer_range,
                "attention_bias": self.attention_bias,
                "attention_dropout": self.attention_dropout,
                "dtype": self.dtype,
            }
        )
        return config
