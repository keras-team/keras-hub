import keras

from keras_hub.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_hub.src.layers.modeling.transformer_decoder import TransformerDecoder
from keras_hub.src.models.moonshine.moonshine_layers import MoonshineArange
from keras_hub.src.models.moonshine.moonshine_layers import MoonshineLinearGeLU
from keras_hub.src.models.moonshine.moonshine_layers import (
    MoonshineRotaryEmbedding,
)
from keras_hub.src.models.moonshine.moonshine_layers import MoonshineSwiGLU
from keras_hub.src.models.moonshine.moonshine_multi_head_attention import (
    MoonshineCausalMultiHeadAttention,
)
from keras_hub.src.models.moonshine.moonshine_multi_head_attention import (
    MoonshinePrecomputedKVMultiHeadAttention,
)


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

    Returns:
        MoonshineDecoderBlock: An instance of `MoonshineDecoderBlock` that can
        be used in a transformer decoder architecture, supporting both cached
        and uncached operations.

    ## References
    Defined and formulated based on the
    [UsefulSensors implementation of the DecoderLayer](https://github.com/usefulsensors/moonshine/blob/4a000427bd36a1c2c6d20a86c672dbd850b44c88/moonshine/model.py#L348)
    class.
    """

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
        super().__init__(
            intermediate_dim=intermediate_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            activation="gelu" if use_swiglu_activation else "relu",
            layer_norm_epsilon=1e-5,
            kernel_initializer=keras.initializers.RandomNormal(
                stddev=initializer_range
            ),
            bias_initializer="zeros",
            normalize_first=True,
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
        self.self_attention = MoonshineCausalMultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.head_dim,
            use_bias=False,
            kernel_initializer=keras.initializers.RandomNormal(
                stddev=initializer_range
            ),
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            dtype=self.dtype,
        )
        self.norm2 = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            center=False,
            scale=True,
            dtype=self.dtype,
        )
        self.cross_attention = MoonshinePrecomputedKVMultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.head_dim,
            use_bias=False,
            kernel_initializer=keras.initializers.RandomNormal(
                stddev=initializer_range
            ),
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            dtype=self.dtype,
        )
        self.norm3 = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            center=False,
            scale=True,
            dtype=self.dtype,
        )
        self.ff = (
            MoonshineSwiGLU(
                hidden_dim,
                feedforward_expansion_factor,
                kernel_initializer=keras.initializers.RandomNormal(
                    stddev=initializer_range
                ),
                dtype=self.dtype,
            )
            if use_swiglu_activation
            else MoonshineLinearGeLU(
                hidden_dim,
                feedforward_expansion_factor,
                kernel_initializer=keras.initializers.RandomNormal(
                    stddev=initializer_range
                ),
                dtype=self.dtype,
            )
        )

    def build(self, input_shape):
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) < 2:
            raise ValueError(
                "Expected input_shape to be a list of at least two shapes."
            )
        decoder_sequence_shape = input_shape[0]  # Shape of x.
        context_shape = input_shape[1]  # Shape of context.

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

    def compute_output_spec(self, inputs, training=None, use_cache=False):
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
            x_shape = x.shape
            output_shape = x_shape[:-1] + (self.hidden_dim,)
            # New cache shapes are the same as input cache_k_shape and
            # cache_v_shape.
            # Note: In practice, sequence length may increase due to
            # concatenation, but symbolically, it remains None.
            new_cache_shape = cache_k.shape
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
            x_shape = x.shape
            context_shape = context.shape
            batch_size = x_shape[0]  # None (symbolic).
            seq_len = x_shape[1]  # None (symbolic).
            context_len = context_shape[1]  # None (symbolic).
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
            )  # Self-attention caches.
            cache_shape_cross = (
                batch_size,
                context_len,
                num_heads,
                head_dim,
            )  # Cross-attention caches.

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

    def call(self, inputs, training=None, use_cache=False):
        if use_cache:
            (
                x,
                context,
                cache_k,
                cache_v,
                x_attn_cache_k,
                x_attn_cache_v,
                rotary_embedding,
            ) = inputs
        else:
            x, context, rotary_embedding = inputs

        residual = x
        x = self.norm1(x)
        if use_cache:
            x, new_cache_k, new_cache_v = self.self_attention(
                query=x,
                key=x,
                value=x,
                rotary_embedding=rotary_embedding,
                key_cache=cache_k,
                value_cache=cache_v,
                training=training,
            )
        else:
            x, cache_k, cache_v = self.self_attention(
                query=x,
                key=x,
                value=x,
                rotary_embedding=rotary_embedding,
                key_cache=None,
                value_cache=None,
                training=training,
            )
        x = x + residual

        residual = x
        x = self.norm2(x)
        if use_cache:
            x = self.cross_attention(
                query=x,
                key=context,
                value=context,
                key_cache=x_attn_cache_k,
                value_cache=x_attn_cache_v,
                training=training,
            )
        else:
            x, x_attn_cache_k, x_attn_cache_v = self.cross_attention(
                query=x,
                key=context,
                value=context,
                key_cache=None,
                value_cache=None,
                training=training,
            )
        x = x + residual

        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        x = x + residual

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


@keras.saving.register_keras_serializable(package="keras_hub")
class MoonshineDecoder(keras.layers.Layer):
    """Moonshine decoder block for sequence processing.

    A transformer decoder model that stacks multiple `MoonshineDecoderBlock`
    layers, an embedding layer with reversible projection, rotary positional
    embeddings, and a final normalization to produce output logits. This model
    supports both cached and uncached operation modes for efficient
    autoregressive generation.

    Args:
        num_layers: int. The number of decoder layers in the transformer stack.
        hidden_dim: int. The dimensionality of the model's hidden
            representations and embeddings.
        intermediate_dim: int. The dimensionality of the intermediate
            representations in the feedforward networks.
        num_heads: int. The number of attention heads for multi-head attention
            mechanisms.
        vocabulary_size: int. The size of the vocabulary for the reversible
            embedding layer.
        feedforward_expansion_factor: int, optional. A multiplicative factor
            for scaling the feedforward network dimension. Defaults to 4.
        use_swiglu_activation: bool, optional. Whether to use the SwiGLU
            activation in the feedforward networks for improved performance.
            Defaults to True.
        max_position_embeddings: int, optional. The maximum sequence length that
            can be processed, determining the range of positional embeddings.
            Defaults to 2048.
        pad_head_dim_to_multiple_of: int, optional. If specified, pads the head
            dimension to be a multiple of this value for performance
            optimization. Defaults to None.
        partial_rotary_factor: float, optional. The fraction of dimensions to
            apply rotary position embeddings to. Defaults to 0.62.
        initializer_range: float. The standard deviation of the truncated normal
            distribution used to initialize model weights.
        attention_bias: bool. Whether to add a bias term to the attention
            computations.
        attention_dropout: float. The dropout rate applied to attention weights
            during training.
        rope_theta: float. The base value for the rotary position embeddings.
        rope_scaling: dict, optional. A scaling mechanism for rotary
            position embeddings. Defaults to None.
        dtype: str, optional. The data type to use for model computations and
            weights. Defaults to None.
        **kwargs: Additional keyword arguments passed to the base layer.

    Returns:
        MoonshineDecoder: An instance of `MoonshineDecoder` that can be used
        for text generation and language modeling tasks, supporting both cached
        and uncached operations.

    ## References
    Defined and formulated based on the
    [UsefulSensors implementation of the Decoder](https://github.com/usefulsensors/moonshine/blob/4a000427bd36a1c2c6d20a86c672dbd850b44c88/moonshine/model.py#L487)
    class of Moonshine.
    """

    def __init__(
        self,
        num_layers,
        hidden_dim,
        intermediate_dim,
        num_heads,
        vocabulary_size,
        feedforward_expansion_factor=4,
        use_swiglu_activation=True,
        max_position_embeddings=2048,
        pad_head_dim_to_multiple_of=None,
        partial_rotary_factor=0.62,
        initializer_range=0.02,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        dtype=None,
        rope_scaling=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.initializer_range = initializer_range
        self.rope_theta = rope_theta
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.vocabulary_size = vocabulary_size
        self.feedforward_expansion_factor = feedforward_expansion_factor
        self.use_swiglu_activation = use_swiglu_activation
        self.max_position_embeddings = max_position_embeddings
        self.pad_head_dim_to_multiple_of = pad_head_dim_to_multiple_of
        self.partial_rotary_factor = partial_rotary_factor
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling

        self.head_dim = hidden_dim // num_heads
        if pad_head_dim_to_multiple_of is not None:
            self.head_dim = (
                (self.head_dim + pad_head_dim_to_multiple_of - 1)
                // pad_head_dim_to_multiple_of
            ) * pad_head_dim_to_multiple_of

        self.embedding_layer = ReversibleEmbedding(
            vocabulary_size,
            hidden_dim,
            embeddings_initializer=keras.initializers.RandomNormal(
                stddev=initializer_range
            ),
            dtype=self.dtype,
        )

        self.decoder_layers = [
            MoonshineDecoderBlock(
                hidden_dim=hidden_dim,
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                feedforward_expansion_factor=feedforward_expansion_factor,
                use_swiglu_activation=use_swiglu_activation,
                pad_head_dim_to_multiple_of=pad_head_dim_to_multiple_of,
                initializer_range=initializer_range,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                dtype=self.dtype,
            )
            for _ in range(num_layers)
        ]

        self.post_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            center=False,
            scale=True,
            dtype=self.dtype,
        )

        self.arange = MoonshineArange(dtype=self.dtype)
        self.rotary_embedding = MoonshineRotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base_value=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            name="rotary_embedding",
            rope_scaling=rope_scaling,
            dtype=self.dtype,
        )

        self.uncached_call = self._build_uncached_call()
        self.cached_call = self._build_cached_call()

    def _build_uncached_call(self):
        # ==== Functional Model ====
        inputs = keras.layers.Input(shape=[None], dtype="int32")
        seq_len = keras.layers.Input(shape=[], dtype="int32")
        context = keras.layers.Input(
            shape=[None, self.hidden_dim], dtype="float32"
        )
        pos_indices = keras.ops.arange(
            self.max_position_embeddings, dtype="int32"
        )

        x = self.embedding_layer(inputs)
        rotary_embedding = self.rotary_embedding(pos_indices)

        # Process through decoder blocks.
        outputs = []
        for layer in self.decoder_layers:
            x, cache_k, cache_v, cross_k, cross_v = layer(
                [x, context, rotary_embedding], use_cache=False
            )
            outputs.extend([cache_k, cache_v, cross_k, cross_v])

        x = self.post_norm(x)

        return keras.Model(
            inputs=[inputs, context, seq_len],
            outputs=x,
            name="uncached_decoder",
        )

    def call(self, inputs, training=None, use_cache=False):
        if use_cache:
            if not isinstance(inputs, (list, tuple)) or len(inputs) < 3:
                raise ValueError(
                    "When use_cache=True, inputs should be a list of "
                    "[token_ids, context, seq_len] + cache_inputs"
                )
            return self.cached_call(inputs)
        else:
            if not isinstance(inputs, (list, tuple)) or len(inputs) != 3:
                raise ValueError(
                    "When use_cache=False, inputs should be a list of "
                    "[token_ids, context, seq_len]"
                )
            return self.uncached_call(inputs)

    def compute_output_spec(self, inputs):
        if len(inputs) == 3:  # Uncached call.
            token_ids, context, seq_len = inputs
            batch_size = token_ids.shape[0]
            seq_len_shape = token_ids.shape[1]
            return keras.KerasTensor(
                shape=(batch_size, seq_len_shape, self.hidden_dim),
                dtype=self.dtype,
            )
        elif len(inputs) == 3 + 4 * self.num_layers:  # Cached call.
            # Unpack inputs: token_ids, context, seq_len, followed by cache
            # tensors.
            token_ids, context, seq_len, *cache_inputs = inputs
            batch_size = token_ids.shape[0]
            seq_len_shape = token_ids.shape[1]

            # Decoder output spec.
            x_spec = keras.KerasTensor(
                shape=(batch_size, seq_len_shape, self.hidden_dim),
                dtype=self.dtype,
            )

            # Cache output specs.
            cache_specs = []
            for i in range(self.num_layers):
                self_k_in = cache_inputs[4 * i]  # Self-attention key cache
                # input.
                self_v_in = cache_inputs[4 * i + 1]  # Self-attention value
                # cache input.
                cross_k_in = cache_inputs[4 * i + 2]  # Cross-attention key
                # cache input.
                cross_v_in = cache_inputs[4 * i + 3]  # Cross-attention value
                # cache input.

                # Output cache specs: shapes match input caches symbolically.
                self_k_out_spec = keras.KerasTensor(
                    shape=self_k_in.shape,
                    dtype=self.dtype,
                )
                self_v_out_spec = keras.KerasTensor(
                    shape=self_v_in.shape,
                    dtype=self.dtype,
                )
                cross_k_out_spec = keras.KerasTensor(
                    shape=cross_k_in.shape,
                    dtype=self.dtype,
                )
                cross_v_out_spec = keras.KerasTensor(
                    shape=cross_v_in.shape,
                    dtype=self.dtype,
                )

                cache_specs.extend(
                    [
                        self_k_out_spec,
                        self_v_out_spec,
                        cross_k_out_spec,
                        cross_v_out_spec,
                    ]
                )

            # Return list of [decoder output] + cache outputs.
            return [x_spec] + cache_specs
        else:
            raise ValueError(
                f"Expected 3 inputs for uncached call or {3 + 4 * self.num_layers} "  # noqa: E501
                f"inputs for cached call, got {len(inputs)} inputs."
            )

    def _build_cached_call(self):
        # ==== Functional Model ====
        key_dim = self.head_dim

        inputs = keras.layers.Input(shape=[None], dtype="int32")
        seq_len = keras.layers.Input(shape=[], batch_size=1, dtype="int32")
        context = keras.layers.Input(
            shape=[None, self.hidden_dim], dtype="float32"
        )

        # Cache inputs: [self_k, self_v, cross_k, cross_v] for each layer.
        cache_inputs = [
            [
                keras.layers.Input(
                    shape=[None, self.num_heads, key_dim], dtype="float32"
                ),
                keras.layers.Input(
                    shape=[None, self.num_heads, key_dim], dtype="float32"
                ),
                keras.layers.Input(
                    shape=[None, self.num_heads, key_dim], dtype="float32"
                ),
                keras.layers.Input(
                    shape=[None, self.num_heads, key_dim], dtype="float32"
                ),
            ]
            for _ in range(self.num_layers)
        ]
        cache_inputs = sum(cache_inputs, [])

        x = self.embedding_layer(inputs)
        rotary_embedding = self.rotary_embedding(self.arange(seq_len))

        new_caches = []
        for i, layer in enumerate(self.decoder_layers):
            # Retrieve layer's cache inputs.
            self_k_in = cache_inputs[4 * i + 0]
            self_v_in = cache_inputs[4 * i + 1]
            cross_k_in = cache_inputs[4 * i + 2]
            cross_v_in = cache_inputs[4 * i + 3]

            x, self_k_out, self_v_out = layer(
                [
                    x,
                    context,
                    self_k_in,
                    self_v_in,
                    cross_k_in,
                    cross_v_in,
                    rotary_embedding,
                ],
                use_cache=True,
            )
            new_caches.extend([self_k_out, self_v_out, cross_k_in, cross_v_in])

        x = self.post_norm(x)

        return keras.Model(
            inputs=[inputs, context, seq_len] + cache_inputs,
            outputs=[x] + new_caches,
            name="cached_decoder",
        )

    def get_config(self):
        # ==== Config ====
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "vocabulary_size": self.vocabulary_size,
                "feedforward_expansion_factor": self.feedforward_expansion_factor,  # noqa: E501
                "use_swiglu_activation": self.use_swiglu_activation,
                "max_position_embeddings": self.max_position_embeddings,
                "pad_head_dim_to_multiple_of": self.pad_head_dim_to_multiple_of,  # noqa: E501
                "partial_rotary_factor": self.partial_rotary_factor,
                "initializer_range": self.initializer_range,
                "attention_bias": self.attention_bias,
                "attention_dropout": self.attention_dropout,
                "rope_theta": self.rope_theta,
                "rope_scaling": self.rope_scaling,
                "dtype": self.dtype,
            }
        )
        return config
