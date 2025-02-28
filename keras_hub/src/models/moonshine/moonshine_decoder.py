import keras

from keras_hub.src.models.moonshine.moonshine_layers import MoonshineArange
from keras_hub.src.models.moonshine.moonshine_layers import MoonshineLinearGeLU
from keras_hub.src.models.moonshine.moonshine_layers import (
    MoonshineReversibleEmbedding,
)
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
class MoonshineDecoderBlock(keras.layers.Layer):
    """
    Moonshine decoder block.

    A decoder block that includes self-attention with causal masking and
    cross-attention with precomputed key/value pairs, and a feedforward network.
    Includes support for both cached and uncached operation modes.

    Args:
        hidden_dim: int, Dimensionality of the model's hidden representations.
        intermediate_dim: int, Dimensionality of the intermediate
        representations in the feedforward network.
        num_heads: int, Number of attention heads for multi-head attention
        mechanisms.
        feedforward_expansion_factor: int, Multiplicative factor for scaling the
        feedforward network dimension.
        use_swiglu_activation: bool, Whether to use SwiGLU activation in the
        feedforward network for improved performance.
        pad_head_dim_to_multiple_of: int, optional, If specified, pads the head
        dimension to be a multiple of this value for performance optimization.
        dtype: string or `keras.mixed_precision.DTypePolicy`, optional, The
        dtype to use for model computations and weights. Defaults to None.
        **kwargs, Additional keyword arguments passed to the base layer.
    """

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_heads,
        feedforward_expansion_factor=4,
        use_swiglu_activation=True,
        pad_head_dim_to_multiple_of=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.feedforward_expansion_factor = feedforward_expansion_factor
        self.use_swiglu_activation = use_swiglu_activation
        self.pad_head_dim_to_multiple_of = pad_head_dim_to_multiple_of

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
                dtype=self.dtype,
            )
            if use_swiglu_activation
            else MoonshineLinearGeLU(
                hidden_dim,
                feedforward_expansion_factor,
                dtype=self.dtype,
            )
        )

    def call(
        self,
        inputs,
        training=None,
        use_cache=False,
    ):
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

    def get_uncached_call(self, hidden_dim):
        inputs = keras.layers.Input(shape=[None, hidden_dim])
        context = keras.layers.Input(shape=[None, hidden_dim])
        rotary_embedding = keras.layers.Input(shape=[None, None], batch_size=1)
        rotary_embedding = keras.ops.squeeze(rotary_embedding)

        outputs = self([inputs, context, rotary_embedding], use_cache=False)

        return keras.Model(
            inputs=[inputs, context, rotary_embedding],
            outputs=outputs,
        )

    def get_cached_call(self, hidden_dim, key_dim, num_heads):
        inputs = keras.layers.Input(shape=[None, hidden_dim])
        context = keras.layers.Input(shape=[None, hidden_dim])
        cache_k = keras.layers.Input(shape=[None, num_heads, key_dim])
        cache_v = keras.layers.Input(shape=[None, num_heads, key_dim])
        x_attn_cache_k = keras.layers.Input(shape=[None, num_heads, key_dim])
        x_attn_cache_v = keras.layers.Input(shape=[None, num_heads, key_dim])
        rotary_embedding = keras.layers.Input(shape=[None, None], batch_size=1)
        rotary_embedding = keras.ops.squeeze(rotary_embedding)

        outputs = self(
            [
                inputs,
                context,
                cache_k,
                cache_v,
                x_attn_cache_k,
                x_attn_cache_v,
                rotary_embedding,
            ],
            use_cache=True,
        )

        return keras.Model(
            inputs=[
                inputs,
                context,
                cache_k,
                cache_v,
                x_attn_cache_k,
                x_attn_cache_v,
                rotary_embedding,
            ],
            outputs=outputs,
        )

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
                "dtype": self.dtype,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class MoonshineDecoder(keras.Model):
    """
    Moonshine decoder.

    A transformer decoder model that stacks multiple MoonshineDecoderBlock
    layers, an embedding layer with reversible projection, rotary positional
    embeddings, and a final normalization to produce output logits. This model
    supports both cached and uncached operation modes for efficient
    autoregressive generation.

    Args:
        num_layers: int, Number of decoder layers in the transformer stack.
        hidden_dim: int, Dimensionality of the model's hidden representations
        and embeddings.
        intermediate_dim: int, Dimensionality of the intermediate
        representations in the feedforward networks.
        num_heads: int, Number of attention heads for multi-head attention
        mechanisms.
        vocabulary_size: int, Size of the vocabulary for the reversible
        embedding layer.
        feedforward_expansion_factor: int, optional, Multiplicative factor for
        scaling the feedforward network dimension. Defaults to 4.
        use_swiglu_activation: bool, optional, Whether to use SwiGLU activation
        in the feedforward networks for improved performance. Defaults to True.
        max_position_embeddings: int, optional, Maximum sequence length that can
        be processed, determining the range of positional embeddings. Defaults
        to 2048.
        pad_head_dim_to_multiple_of: int, optional, If specified, pads the head
        dimension to be a multiple of this value for performance optimization.
        Defaults to None.
        partial_rotary_factor: float, optional, Fraction of dimensions to apply
        rotary position embeddings to. Defaults to 0.62.
        dtype: string or `keras.mixed_precision.DTypePolicy`, optional, The
        dtype to use for model computations and weights. Defaults to None.
        **kwargs, Additional keyword arguments passed to the base keras.Model.

    Examples:

    ```python
    import numpy as np
    from keras_hub.src.models.moonshine.moonshine_decoder import (
        MoonshineDecoder
    )

    token_ids = np.random.randint(0, 10000, size=(1, 20)).astype("int32")
    context = np.random.rand(1, 30, 256).astype("float32")
    seq_len = np.array([20], dtype="int32")

    decoder = MoonshineDecoder(
        num_layers=4,
        hidden_dim=256,
        intermediate_dim=512,
        num_heads=8,
        vocabulary_size=10000,
        feedforward_expansion_factor=4,
        use_swiglu_activation=True,
    )

    outputs = decoder([token_ids, context, seq_len])
    logits = outputs[0]
    print(logits.shape)
    ```
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
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
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

        self.head_dim = hidden_dim // num_heads
        if pad_head_dim_to_multiple_of is not None:
            self.head_dim = (
                (self.head_dim + pad_head_dim_to_multiple_of - 1)
                // pad_head_dim_to_multiple_of
            ) * pad_head_dim_to_multiple_of

        self.embedding_layer = MoonshineReversibleEmbedding(
            vocabulary_size,
            hidden_dim,
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
            dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            partial_rotary_factor=partial_rotary_factor,
            name="rotary_embedding",
            dtype=self.dtype,
        )

        self.uncached_call = self._build_uncached_call()
        self.cached_call = self._build_cached_call()

    def _build_uncached_call(self):
        # ==== Functional Model ====
        inputs = keras.layers.Input(shape=[None], dtype="int32")
        seq_len = keras.layers.Input(shape=[], batch_size=1, dtype="int32")
        context = keras.layers.Input(
            shape=[None, self.hidden_dim], dtype="float32"
        )

        x = self.embedding_layer(inputs)
        rotary_embedding = self.rotary_embedding(self.arange(seq_len))

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
            outputs=[x] + outputs,
            name="uncached_decoder",
        )

    def call(self, inputs, training=None, use_cache=False):
        """
        Forward pass of the model.

        Args:
        inputs: List[Tensor]
            List containing:
            - token_ids: Tensor[int32]
                Integer tensor of shape [batch_size, seq_len] containing token
                IDs.
            - context: Tensor[float32]
                Float tensor of shape [batch_size, context_len, hidden_dim]
                containing context vectors.
            - seq_len: Tensor[int32]
                Integer tensor of shape [batch_size] specifying valid sequence
                lengths.
            - [Optional] cache inputs if use_cache=True.
        training: bool, optional
            Flag indicating whether the model is in training mode.
        use_cache: bool, optional
            Flag indicating whether to use cached computation for efficient
            autoregressive generation.

        Returns:
        List[Tensor]
            List containing:
            - logits: Tensor[float32]
                Float tensor of shape [batch_size, seq_len, vocabulary_size]
                containing output logits.
            - cache outputs: List[Tensor], optional
                Cache tensors for subsequent calls if use_cache=True.
        """
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
                "dtype": self.dtype,
            }
        )
        return config
