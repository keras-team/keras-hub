import keras
from keras import layers
from keras import ops

from keras_hub.src.models.moonshine.moonshine_custom_attention import (
    MHAWithRope,
)
from keras_hub.src.models.moonshine.moonshine_custom_feedforward import (
    FFLinearGelu,
)
from keras_hub.src.models.moonshine.moonshine_custom_feedforward import FFSwiGLU


class MoonshineEncoderLayer(layers.Layer):
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

    def call(self, x, rot_pos_emb):
        # Self-attention block.
        shortcut = x
        x = self.norm1(x)
        x = self.attention(query=x, key=x, value=x, rot_pos_emb=rot_pos_emb)
        x = x + shortcut

        # Feed-forward block.
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class MoonshineEncoder(layers.Layer):
    def __init__(
        self,
        n_layers,
        dim,
        inner_dim,
        n_head,
        ff_mult=4,
        ff_swiglu=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_layers = n_layers
        self.dim = dim
        self.inner_dim = inner_dim
        self.n_head = n_head
        self.ff_mult = ff_mult
        self.ff_swiglu = ff_swiglu

        # Initialize layers.
        self.encoder_layers = []
        for i in range(n_layers):
            layer = MoonshineEncoderLayer(
                dim=dim,
                inner_dim=inner_dim,
                n_head=n_head,
                ff_mult=ff_mult,
                ff_swiglu=ff_swiglu,
                name=f"encoder_layer_{i}",
            )
            self.encoder_layers.append(layer)

        self.final_norm = layers.LayerNormalization(
            axis=-1, epsilon=1e-5, center=False, scale=True, name="final_norm"
        )

    def call(self, x, rot_pos_emb):
        if not isinstance(x, keras.KerasTensor):
            x = ops.convert_to_tensor(x, dtype="float32")
        if not isinstance(rot_pos_emb, keras.KerasTensor):
            rot_pos_emb = ops.convert_to_tensor(rot_pos_emb, dtype="float32")

        for layer in self.encoder_layers:
            x = layer(x, rot_pos_emb)

        return self.final_norm(x)

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
