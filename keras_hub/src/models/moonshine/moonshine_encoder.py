from keras import layers
from keras import models
from keras import ops

from keras_hub.src.models.moonshine.moonshine_custom_attention import (
    MHAWithRope,
)
from keras_hub.src.models.moonshine.moonshine_custom_feedforward import (
    FFLinearGelu,
)
from keras_hub.src.models.moonshine.moonshine_custom_feedforward import FFSwiGLU
from keras_hub.src.models.moonshine.moonshine_utils import Arange
from keras_hub.src.models.moonshine.moonshine_utils import RotaryEmbedding


class MoonshineEncoderLayer(layers.Layer):
    def __init__(self, dim, inner_dim, n_head, ff_mult, ff_swiglu):
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim
        self.n_head = n_head
        self.ff_mult = ff_mult
        self.ff_swiglu = ff_swiglu

        self.norm1 = layers.LayerNormalization(
            axis=-1, epsilon=1e-5, center=False, scale=True
        )
        self.attention = MHAWithRope(
            num_heads=n_head,
            key_dim=inner_dim // n_head,
            use_bias=False,
        )
        self.norm2 = layers.LayerNormalization(
            axis=-1, epsilon=1e-5, center=False, scale=True
        )
        self.ff = (
            FFSwiGLU(dim, ff_mult) if ff_swiglu else FFLinearGelu(dim, ff_mult)
        )

        inputs = layers.Input(shape=[None, dim])
        rot_pos_emb = layers.Input(shape=[None, None], batch_size=1)
        rot_pos_emb = ops.squeeze(rot_pos_emb)

        x = inputs
        _x = x
        x = self.norm1(x)
        x = self.attention(query=x, key=x, value=x, rot_pos_emb=rot_pos_emb)
        x = x + _x
        _x = x
        x = self.norm2(x)
        x = self.ff(x)
        outputs = x + _x
        self.encoder_layer = models.Model(
            inputs=[inputs, rot_pos_emb], outputs=outputs
        )

    def call(self, x, rot_pos_emb):
        return self.encoder_layer([x, rot_pos_emb])

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "inner_dim": self.inner_dim,
                "n_head": self.n_head,
                "ff_mult": self.ff_mult,
                "ff_swiglu": self.ff_swiglu,
            }
        )
        return config

    def set_weights(self, weights):
        self.encoder_layer.set_weights(weights)

    def load_weights(self, filepath, **kwargs):
        return self.encoder_layer.load_weights(filepath)


class MoonshineEncoder(models.Model):
    def __init__(
        self, n_layers, dim, inner_dim, n_head, ff_mult=4, ff_swiglu=False
    ):
        super().__init__()
        self.n_layers = n_layers
        self.dim = dim
        self.inner_dim = inner_dim
        self.n_head = n_head
        self.ff_mult = ff_mult
        self.ff_swiglu = ff_swiglu

        rot_embed_dim = max(inner_dim // n_head // 2, 32)
        self.rot_pos_emb = RotaryEmbedding(rot_embed_dim)

        self.encoder_layers = [
            MoonshineEncoderLayer(dim, inner_dim, n_head, ff_mult, ff_swiglu)
            for _ in range(n_layers)
        ]

        self.final_norm = layers.LayerNormalization(
            axis=-1, epsilon=1e-5, center=False, scale=True
        )
        inputs = layers.Input(shape=[None, dim])
        seq_len = layers.Input(shape=[], batch_size=1, dtype="int32")
        pos_emb = self.rot_pos_emb(Arange()(inputs=seq_len))
        x = inputs
        for layer in self.encoder_layers:
            x = layer(x, pos_emb)
        outputs = self.final_norm(x)
        self.encoder = models.Model(inputs=[inputs, seq_len], outputs=outputs)

    def call(self, x, seq_len=None):
        # Allow seq_len to be optional. If not provided, compute it from x.
        if seq_len is None:
            seq_len = ops.convert_to_tensor([ops.shape(x)[1]], dtype="int32")
        return self.encoder([x, seq_len])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_layers": self.n_layers,
                "dim": self.dim,
                "inner_dim": self.inner_dim,
                "n_head": self.n_head,
                "ff_mult": self.ff_mult,
                "ff_swiglu": self.ff_swiglu,
            }
        )
        return config

    def set_weights(self, weights):
        self.encoder.set_weights(weights)

    def load_weights(self, filepath, **kwargs):
        return self.encoder.load_weights(filepath, **kwargs)
