from keras import layers
from keras import models
from keras import ops

from keras_hub.src.models.moonshine.moonshine_custom_attention import (
    MHACausalWithRope,
)
from keras_hub.src.models.moonshine.moonshine_custom_attention import (
    MHAPrecomputedKV,
)
from keras_hub.src.models.moonshine.moonshine_custom_feedforward import (
    FFLinearGelu,
)
from keras_hub.src.models.moonshine.moonshine_custom_feedforward import FFSwiGLU


class MoonshineDecoder(layers.Layer):
    def __init__(self, dim, inner_dim, n_head, ff_mult, ff_swiglu):
        self.norm1 = layers.LayerNormalization(
            axis=-1, epsilon=1e-5, center=False, scale=True
        )
        self.self_attention = MHACausalWithRope(
            num_heads=n_head, key_dim=inner_dim // n_head, use_bias=False
        )
        self.norm2 = layers.LayerNormalization(
            axis=-1, epsilon=1e-5, center=False, scale=True
        )
        self.cross_attention = MHAPrecomputedKV(
            num_heads=n_head, key_dim=inner_dim // n_head, use_bias=False
        )
        self.norm3 = layers.LayerNormalization(
            axis=-1, epsilon=1e-5, center=False, scale=True
        )
        self.ff = (
            FFSwiGLU(dim, ff_mult) if ff_swiglu else FFLinearGelu(dim, ff_mult)
        )
        self.uncached_call = self.get_uncached_call(dim)
        self.cached_call = self.get_cached_call(
            dim, inner_dim // n_head, n_head
        )

    def get_uncached_call(self, dim):
        inputs = layers.Input(shape=[None, dim])
        context = layers.Input(shape=[None, dim])
        rot_pos_emb = layers.Input(shape=[None, None], batch_size=1)
        rot_pos_emb = ops.squeeze(rot_pos_emb)
        x = inputs
        shortcut = x
        x = self.norm1(x)
        x, cache_k, cache_v = self.self_attention(
            query=x,
            key=x,
            value=x,
            rot_pos_emb=rot_pos_emb,
            key_cache=None,
            value_cache=None,
        )
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x, x_attn_cache_k, x_attn_cache_v = self.cross_attention(
            query=x,
            key=context,
            value=context,
            key_cache=None,
            value_cache=None,
        )
        x = x + shortcut

        shortcut = x
        x = self.norm3(x)
        x = self.ff(x)
        outputs = x + shortcut

        return models.Model(
            inputs=[inputs, context, rot_pos_emb],
            outputs=[outputs, cache_k, cache_v, x_attn_cache_k, x_attn_cache_v],
        )

    def get_cached_call(self, dim, key_dim, n_head):
        inputs = layers.Input(shape=[None, dim])
        context = layers.Input(shape=[None, dim])
        cache_k = layers.Input(shape=[None, n_head, key_dim])
        cache_v = layers.Input(shape=[None, n_head, key_dim])
        x_attn_cache_k = layers.Input(shape=[None, n_head, key_dim])
        x_attn_cache_v = layers.Input(shape=[None, n_head, key_dim])
        rot_pos_emb = layers.Input(shape=[None, None], batch_size=1)
        rot_pos_emb = ops.squeeze(rot_pos_emb)

        x = inputs
        shortcut = x
        x = self.norm1(x)
        x, new_cache_k, new_cache_v = self.self_attention(
            query=x,
            key=x,
            value=x,
            rot_pos_emb=rot_pos_emb,
            key_cache=cache_k,
            value_cache=cache_v,
        )
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.cross_attention(
            query=x,
            key=context,
            value=context,
            key_cache=x_attn_cache_k,
            value_cache=x_attn_cache_v,
        )
        x = x + shortcut

        shortcut = x
        x = self.norm3(x)
        x = self.ff(x)
        x = x + shortcut

        return models.Model(
            inputs=[
                inputs,
                context,
                cache_k,
                cache_v,
                x_attn_cache_k,
                x_attn_cache_v,
                rot_pos_emb,
            ],
            outputs=[x, new_cache_k, new_cache_v],
        )
