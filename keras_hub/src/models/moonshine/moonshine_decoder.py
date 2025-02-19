import keras

from keras_hub.src.models.moonshine.moonshine_layers import MoonshineLinearGeLU
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
