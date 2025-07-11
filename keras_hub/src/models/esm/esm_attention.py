import keras
from keras import ops
from packaging import version

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.roformer_v2.roformer_v2_attention import (
    RoformerAttention,
)


class ESMRotaryEmbedding(RotaryEmbedding):
    def _compute_cos_sin_embedding(self, x, position=1):
        dim = x.shape[-1]
        inv_freq = self.scaling_factor / (
            self.max_wavelength ** (ops.arange(0, dim, 2, dtype=x.dtype) / dim)
        )
        t = ops.arange(x.shape[position], dtype=x.dtype)
        freqs = ops.outer(t, inv_freq)
        emb = ops.concatenate((freqs, freqs), axis=-1)

        cos_emb = ops.cos(emb)[None, :, None, :]
        sin_emb = ops.sin(emb)[None, :, None, :]
        return cos_emb, sin_emb

    def call(self, q, k, position=1):
        cos_emb, sin_emb = self._compute_cos_sin_embedding(q, position)

        return (
            self.apply_rotary_pos_emb(q, cos_emb, sin_emb),
            self.apply_rotary_pos_emb(k, cos_emb, sin_emb),
        )

    def rotate_half(self, x):
        x1, x2 = ops.split(x, 2, -1)
        return ops.concatenate((-x2, x1), axis=-1)

    def apply_rotary_pos_emb(self, x, cos, sin):
        cos = cos[:, : x.shape[1], :, :]
        sin = sin[:, : x.shape[1], :, :]

        return (x * cos) + (self.rotate_half(x) * sin)


class EsmSelfAttention(RoformerAttention):
    """MultiHeadAttention by ESM2

    Referred to the implementation of HuggingFace.
    In fact, this part of the calculation is exactly the same as RoFormer.
    Only the calculation of the rotary part is different.
    """

    def __init__(self, use_rotary=True, **kwargs):
        super().__init__(**kwargs)
        self.use_rotary = use_rotary

    def build(self, input_shape):
        super().build(input_shape)
        if self.use_rotary:
            self.rotary_embedding_layer = ESMRotaryEmbedding(
                max_wavelength=self.max_wavelength, dtype=self.dtype_policy
            )
            self.rotary_embedding_layer.build([])

    def call(self, x, attention_mask=None):
        qw = self.q_dense(x)
        kw = self.k_dense(x)
        vw = self.v_dense(x)

        b, s = ops.shape(qw)[:2]
        qw = ops.reshape(qw, (b, s, self.heads, self.head_size))
        kw = ops.reshape(kw, (b, s, self.heads, self.head_size))
        vw = ops.reshape(vw, (b, s, self.heads, self.head_size))

        if self.use_rotary:
            qw, kw = self.rotary_embedding_layer(qw, kw)
        if version.parse(keras.__version__) < version.parse("3.6"):
            raise ValueError("Please make sure your Keras version is >=3.6.")
        flash_attention = keras.config.is_flash_attention_enabled()
        attention_mask = ops.reshape(attention_mask, [b, 1, s, 1])
        if keras.config.backend() == "torch":
            attention_mask = ops.repeat(attention_mask, s, -1)
            attention_mask = ops.transpose(attention_mask, [0, 1, 3, 2])
        o = ops.dot_product_attention(
            qw, kw, vw, mask=attention_mask, flash_attention=flash_attention
        )
        return self.o_dense(ops.reshape(o, [b, s, -1]))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "use_rotary": self.use_rotary,
            }
        )
        return config
