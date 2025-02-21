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

    A transformer decoder block that includes self-attention with causal masking
    cross-attention with precomputed key/value pairs, and a feedforward network.
    Includes support for both cached and uncached operation modes.

    Args:
        hidden_dim (int): The dimensionality of the model.
        inner_dim (int): The inner dimensionality for feedforward layers.
        num_heads (int): The number of attention heads.
        ff_mult (int): Multiplicative factor for the feedforward dimension.
        ff_swiglu (bool): Whether to use SwiGLU in the feedforward branch.
        **kwargs: Additional keyword arguments passed to the base layer.
    """

    def __init__(
        self,
        hidden_dim,
        inner_dim,
        num_heads,
        ff_mult=4,
        ff_swiglu=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm1 = keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, center=False, scale=True
        )
        self.self_attention = MoonshineCausalMultiHeadAttention(
            num_heads=num_heads,
            key_dim=inner_dim // num_heads,
            use_bias=False,
        )
        self.norm2 = keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, center=False, scale=True
        )
        self.cross_attention = MoonshinePrecomputedKVMultiHeadAttention(
            num_heads=num_heads,
            key_dim=inner_dim // num_heads,
            use_bias=False,
        )
        self.norm3 = keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, center=False, scale=True
        )
        self.ff = (
            MoonshineSwiGLU(hidden_dim, ff_mult)
            if ff_swiglu
            else MoonshineLinearGeLU(hidden_dim, ff_mult)
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
                rot_pos_emb,
            ) = inputs
        else:
            x, context, rot_pos_emb = inputs

        residual = x
        x = self.norm1(x)
        if use_cache:
            x, new_cache_k, new_cache_v = self.self_attention(
                query=x,
                key=x,
                value=x,
                rot_pos_emb=rot_pos_emb,
                key_cache=cache_k,
                value_cache=cache_v,
                training=training,
            )
        else:
            x, cache_k, cache_v = self.self_attention(
                query=x,
                key=x,
                value=x,
                rot_pos_emb=rot_pos_emb,
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
        rot_pos_emb = keras.layers.Input(shape=[None, None], batch_size=1)
        rot_pos_emb = keras.ops.squeeze(rot_pos_emb)

        outputs = self([inputs, context, rot_pos_emb], use_cache=False)

        return keras.Model(
            inputs=[inputs, context, rot_pos_emb],
            outputs=outputs,
        )

    def get_cached_call(self, hidden_dim, key_dim, num_heads):
        inputs = keras.layers.Input(shape=[None, hidden_dim])
        context = keras.layers.Input(shape=[None, hidden_dim])
        cache_k = keras.layers.Input(shape=[None, num_heads, key_dim])
        cache_v = keras.layers.Input(shape=[None, num_heads, key_dim])
        x_attn_cache_k = keras.layers.Input(shape=[None, num_heads, key_dim])
        x_attn_cache_v = keras.layers.Input(shape=[None, num_heads, key_dim])
        rot_pos_emb = keras.layers.Input(shape=[None, None], batch_size=1)
        rot_pos_emb = keras.ops.squeeze(rot_pos_emb)

        outputs = self(
            [
                inputs,
                context,
                cache_k,
                cache_v,
                x_attn_cache_k,
                x_attn_cache_v,
                rot_pos_emb,
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
                rot_pos_emb,
            ],
            outputs=outputs,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "inner_dim": self.inner_dim,
                "num_heads": self.num_heads,
                "ff_mult": self.ff_mult,
                "ff_swiglu": self.ff_swiglu,
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
        num_layers (int): The number of decoder layers.
        hidden_dim (int): The dimensionality of the model.
        inner_dim (int): The inner dimensionality for the feedforward layers in
        each decoder block.
        num_heads (int): The number of attention heads.
        vocab_size (int): The size of the vocabulary for reversible embeddings.
        ff_mult (int, optional): Multiplicative factor for the feedforward
        dimension. Defaults to 4.
        ff_swiglu (bool, optional): Whether to use SwiGLU in the feedforward
        branch. Defaults to True.
        **kwargs: Additional keyword arguments passed to the base model.

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
        inner_dim=512,
        num_heads=8,
        vocab_size=10000,
        ff_mult=4,
        ff_swiglu=True,
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
        inner_dim,
        num_heads,
        vocab_size,
        ff_mult=4,
        ff_swiglu=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_layer = MoonshineReversibleEmbedding(
            vocab_size, hidden_dim
        )
        self.decoder_layers = [
            MoonshineDecoderBlock(
                hidden_dim, inner_dim, num_heads, ff_mult, ff_swiglu
            )
            for _ in range(num_layers)
        ]
        self.post_norm = keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, center=False, scale=True
        )

        rot_embed_dim = max(inner_dim // num_heads // 2, 32)
        self.rot_pos_emb = MoonshineRotaryEmbedding(rot_embed_dim)
        self.arange = MoonshineArange()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.ff_mult = ff_mult
        self.ff_swiglu = ff_swiglu
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
        rot_pos_emb = self.rot_pos_emb(self.arange(seq_len))

        # Process through decoder blocks.
        outputs = []
        for layer in self.decoder_layers:
            x, cache_k, cache_v, cross_k, cross_v = layer(
                [x, context, rot_pos_emb], use_cache=False
            )
            outputs.extend([cache_k, cache_v, cross_k, cross_v])

        x = self.post_norm(x)
        logits = self.embedding_layer(x, reverse=True)

        return keras.Model(
            inputs=[inputs, context, seq_len],
            outputs=[logits] + outputs,
            name="uncached_decoder",
        )

    def call(self, inputs, training=None, use_cache=False):
        """
        Forward pass of the model.

        Args:
            inputs: List containing:
                - token_ids: Int tensor of shape [batch_size, seq_len].
                - context: Float tensor of shape [batch_size, context_len,
                hidden_dim].
                - seq_len: Int tensor of shape [batch_size].
                - [Optional] cache inputs if use_cache=True.
            training: Boolean indicating training mode.
            use_cache: Boolean indicating whether to use cached computation.

        Returns:
            List containing:
                - logits: Float tensor of shape [batch_size, seq_len,
                vocab_size].
                - cache outputs if use_cache=True.
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
        key_dim = self.inner_dim // self.num_heads

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
        rot_pos_emb = self.rot_pos_emb(self.arange(seq_len))

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
                    rot_pos_emb,
                ],
                use_cache=True,
            )
            new_caches.extend([self_k_out, self_v_out, cross_k_in, cross_v_in])

        x = self.post_norm(x)
        logits = self.embedding_layer(x, reverse=True)

        return keras.Model(
            inputs=[inputs, context, seq_len] + cache_inputs,
            outputs=[logits] + new_caches,
            name="cached_decoder",
        )

    def get_config(self):
        # ==== Config ====
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "hidden_dim": self.hidden_dim,
                "inner_dim": self.inner_dim,
                "num_heads": self.num_heads,
                "vocab_size": self.vocab_size,
                "ff_mult": self.ff_mult,
                "ff_swiglu": self.ff_swiglu,
            }
        )
        return config
