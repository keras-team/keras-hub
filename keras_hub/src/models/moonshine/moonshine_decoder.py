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
        ff_swiglu=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.ff_mult = ff_mult
        self.ff_swiglu = ff_swiglu

        # Self-attention sublayers.
        self.self_attention_layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            center=False,
            scale=True,
            name="self_attention_layer_norm",
        )
        self.self_attention_layer = MoonshineCausalMultiHeadAttention(
            num_heads=num_heads,
            key_dim=inner_dim // num_heads,
            use_bias=False,
            name="self_attention_layer",
        )

        # Cross-attention sublayers.
        self.cross_attention_layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            center=False,
            scale=True,
            name="cross_attention_layer_norm",
        )
        self.cross_attention_layer = MoonshinePrecomputedKVMultiHeadAttention(
            num_heads=num_heads,
            key_dim=inner_dim // num_heads,
            use_bias=False,
            name="cross_attention_layer",
        )

        # Feedforward sublayers.
        self.feedforward_layer_norm = keras.layers.LayerNormalization(
            axis=-1,
            epsilon=1e-5,
            center=False,
            scale=True,
            name="feedforward_layer_norm",
        )
        if ff_swiglu:
            self.feedforward = MoonshineSwiGLU(
                hidden_dim, ff_mult, name="feedforward"
            )
        else:
            self.feedforward = MoonshineLinearGeLU(
                hidden_dim, ff_mult, name="feedforward"
            )

    def build(self, input_shape):
        super().build(input_shape)
        # Build self-attention branch.
        self.self_attention_layer_norm.build(input_shape)
        self.self_attention_layer.build(input_shape, input_shape)
        # Build cross-attention branch.
        self.cross_attention_layer_norm.build(input_shape)
        self.cross_attention_layer.build(input_shape, input_shape)
        # Build feedforward branch.
        self.feedforward_layer_norm.build(input_shape)
        feed_forward_input_shape = list(input_shape)
        feed_forward_input_shape[-1] = self.hidden_dim
        self.feedforward.build(tuple(feed_forward_input_shape))

    def call(
        self,
        inputs,
        context,
        rot_pos_emb,
        key_cache=None,
        value_cache=None,
        cross_key_cache=None,
        cross_value_cache=None,
        training=None,
        **kwargs,
    ):
        x = inputs

        # Self-attention.
        attention_residual = x
        x = self.self_attention_layer_norm(x)

        if key_cache is None and value_cache is None:
            x, cache_k, cache_v = self.self_attention_layer(
                query=x,
                key=x,
                value=x,
                rot_pos_emb=rot_pos_emb,
                training=training,
            )
        else:
            x, cache_k, cache_v = self.self_attention_layer(
                query=x,
                key=x,
                value=x,
                rot_pos_emb=rot_pos_emb,
                key_cache=key_cache,
                value_cache=value_cache,
                training=training,
            )
        x = x + attention_residual

        # Cross-attention.
        cross_attention_residual = x
        x = self.cross_attention_layer_norm(x)

        if cross_key_cache is None and cross_value_cache is None:
            x, cross_cache_k, cross_cache_v = self.cross_attention_layer(
                query=x,
                key=context,
                value=context,
                training=training,
            )
            x = x + cross_attention_residual

            # FF with residual.
            ff_residual = x
            x = self.feedforward_layer_norm(x)
            x = self.feedforward(x)
            x = x + ff_residual

            return x, cache_k, cache_v, cross_cache_k, cross_cache_v
        else:
            x = self.cross_attention_layer(
                query=x,
                key=context,
                value=context,
                key_cache=cross_key_cache,
                value_cache=cross_value_cache,
                training=training,
            )
            x = x + cross_attention_residual

            # FF with residual.
            ff_residual = x
            x = self.feedforward_layer_norm(x)
            x = self.feedforward(x)
            x = x + ff_residual

            return x, cache_k, cache_v

    def compute_output_shape(self, input_shape):
        return input_shape

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
    from keras_hub.models.moonshine import MoonshineDecoder

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
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size

        self.embedding = MoonshineReversibleEmbedding(vocab_size, hidden_dim)
        self.decoder_blocks = []
        for i in range(num_layers):
            block = MoonshineDecoderBlock(
                hidden_dim=hidden_dim,
                inner_dim=inner_dim,
                num_heads=num_heads,
                ff_mult=ff_mult,
                ff_swiglu=ff_swiglu,
                name=f"decoder_block_{i}",
            )
            self.decoder_blocks.append(block)

        self.post_norm = keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, center=False, scale=True
        )

        self.rot_embed_dim = max(inner_dim // num_heads // 2, 32)
        self.rot_pos_emb = MoonshineRotaryEmbedding(self.rot_embed_dim)
        self.arange = MoonshineArange()

    def call(self, inputs, training=None):
        if len(inputs) > 3:
            return self._cached_forward(inputs, training)
        return self._uncached_forward(inputs, training)

    def _uncached_forward(self, inputs, training=None):
        x, context, seq_len = inputs
        x = self.embedding(x)
        rot_pos_emb = self.rot_pos_emb(self.arange(seq_len))

        # Process through decoder blocks.
        outputs = []
        for block in self.decoder_blocks:
            x, k, v, cross_k, cross_v = block(
                x, context, rot_pos_emb, training=training
            )
            outputs.extend([k, v, cross_k, cross_v])

        # Final norm and logits.
        x = self.post_norm(x)
        logits = self.embedding(x, reverse=True)

        return [logits] + outputs

    def _cached_forward(self, inputs, training=None):
        x, context, seq_len = inputs[:3]
        cache = inputs[3:]
        x = self.embedding(x)
        rot_pos_emb = self.rot_pos_emb(self.arange(seq_len))

        # Process through decoder blocks with cache.
        outputs = []
        for i, block in enumerate(self.decoder_blocks):
            cache_idx = i * 4
            x, new_k, new_v = block(
                x,
                context,
                rot_pos_emb,
                key_cache=cache[cache_idx],
                value_cache=cache[cache_idx + 1],
                cross_key_cache=cache[cache_idx + 2],
                cross_value_cache=cache[cache_idx + 3],
                training=training,
            )
            outputs.extend(
                [new_k, new_v, cache[cache_idx + 2], cache[cache_idx + 3]]
            )

        # Final norm and logits.
        x = self.post_norm(x)
        logits = self.embedding(x, reverse=True)

        return [logits] + outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_layers": self.num_layers,
                "hidden_dim": self.hidden_dim,
                "inner_dim": self.inner_dim,
                "num_heads": self.num_heads,
                "vocab_size": self.vocab_size,
                "ff_mult": self.decoder_blocks[0].ff_mult
                if self.decoder_blocks
                else None,
                "ff_swiglu": self.decoder_blocks[0].ff_swiglu
                if self.decoder_blocks
                else None,
            }
        )
        return config
