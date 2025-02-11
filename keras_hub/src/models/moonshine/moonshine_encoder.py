from keras import layers
from keras import models
from keras import ops
from moonshine_custom_attention import MHAWithRope
from moonshine_custom_feedforward import FFLinearGelu
from moonshine_custom_feedforward import FFSwiGLU


class MoonshineEncoder(layers.Layer):
    def __init__(self, dim, inner_dim, n_head, ff_mult, ff_swiglu, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(
            axis=-1, epsilon=1e-5, center=False, scale=True
        )
        self.attention = MHAWithRope(
            num_heads=n_head, key_dim=inner_dim // n_head, use_bias=False
        )
        self.norm2 = layers.LayerNormalization(
            axis=-1, epsilon=1e-5, center=False, scale=True
        )
        self.ff = (
            FFSwiGLU(dim, ff_mult) if ff_swiglu else FFLinearGelu(dim, ff_mult)
        )
        self.encoder_layer = self._build_layer(dim)

    def _build_layer(self, dim):
        inputs = layers.Input(shape=[None, dim])
        rot_pos_emb = layers.Input(shape=[None, None], batch_size=1)
        rot_pos_emb = ops.squeeze(rot_pos_emb)
        x = inputs
        shortcut = x
        x = self.norm1(x)
        x = self.attention(query=x, key=x, value=x, rot_pos_emb=rot_pos_emb)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        outputs = x + shortcut
        return models.Model(inputs=[inputs, rot_pos_emb], outputs=outputs)

    def call(self, x, rot_pos_emb):
        residual = x
        x = self.norm1(x)
        x = (
            self.attention(query=x, key=x, value=x, rot_pos_emb=rot_pos_emb)
            + residual
        )
        residual = x
        x = self.norm2(x)
        x = self.ff(x) + residual
        return x
