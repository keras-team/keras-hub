import keras
import numpy as np
from keras import ops


@keras.saving.register_keras_serializable(package="keras_hub")
class Qwen2_5_VLMRoPE3D(keras.layers.Layer):
    """
    3D Multimodal Rotary Position Embedding for Qwen2.5-VL vision attention.

    Encodes temporal, height, and width positions independently using
    mrope_section to split head_dim into three axis-specific regions.
    Remaining dims beyond sum(mrope_section) receive identity rotation.

    For production models (head_dim=80): mrope_section=[16, 24, 24].
    For smaller models: sections are auto-derived as even thirds of head_dim.

    Parameters
    ----------
    head_dim : int
        Dimensionality of each attention head.
    theta : float
        RoPE base frequency.
    mrope_section : list of int or None
        [temporal, height, width] rotation dims. Auto-derived if None.
    """

    def __init__(self, head_dim, theta=10000.0, mrope_section=None, **kwargs):
        super().__init__(**kwargs)

        self.head_dim = head_dim
        self.theta    = theta

        if mrope_section is not None:
            self.mrope_section = list(mrope_section)
        else:
            d = (head_dim // 3) & ~1
            d = max(d, 2)
            self.mrope_section = [d, d, d]
            while sum(self.mrope_section) > head_dim:
                d -= 2
                self.mrope_section = [d, d, d]

        assert len(self.mrope_section) == 3
        assert sum(self.mrope_section) <= head_dim
        assert all(s % 2 == 0 for s in self.mrope_section)

        self.dim_t = self.mrope_section[0]
        self.dim_h = self.mrope_section[1]
        self.dim_w = self.mrope_section[2]

    def build(self, input_shape=None):
        def _inv_freq(dim):
            return (
                1.0 / (self.theta ** (
                    np.arange(0, dim, 2, dtype=np.float32) / dim
                ))
            ).astype(np.float32)

        self.inv_freq_t = self.add_weight(
            name="inv_freq_t",
            shape=(self.dim_t // 2,),
            initializer=keras.initializers.Constant(_inv_freq(self.dim_t)),
            trainable=False,
        )
        self.inv_freq_h = self.add_weight(
            name="inv_freq_h",
            shape=(self.dim_h // 2,),
            initializer=keras.initializers.Constant(_inv_freq(self.dim_h)),
            trainable=False,
        )
        self.inv_freq_w = self.add_weight(
            name="inv_freq_w",
            shape=(self.dim_w // 2,),
            initializer=keras.initializers.Constant(_inv_freq(self.dim_w)),
            trainable=False,
        )
        super().build(input_shape)

    def _axis_embedding(self, positions, inv_freq):
        freqs = ops.einsum("i,j->ij", positions, inv_freq)
        return ops.concatenate([freqs, freqs], axis=-1)

    def call(self, *, T, H, W):
        t_pos = ops.cast(ops.arange(T), "float32")
        h_pos = ops.cast(ops.arange(H), "float32")
        w_pos = ops.cast(ops.arange(W), "float32")

        t_emb = self._axis_embedding(t_pos, self.inv_freq_t)
        h_emb = self._axis_embedding(h_pos, self.inv_freq_h)
        w_emb = self._axis_embedding(w_pos, self.inv_freq_w)

        t_emb = ops.broadcast_to(
            ops.reshape(t_emb, (T, 1, 1, self.dim_t)), (T, H, W, self.dim_t)
        )
        h_emb = ops.broadcast_to(
            ops.reshape(h_emb, (1, H, 1, self.dim_h)), (T, H, W, self.dim_h)
        )
        w_emb = ops.broadcast_to(
            ops.reshape(w_emb, (1, 1, W, self.dim_w)), (T, H, W, self.dim_w)
        )

        emb = ops.concatenate([t_emb, h_emb, w_emb], axis=-1)

        pad_dim = self.head_dim - (self.dim_t + self.dim_h + self.dim_w)
        if pad_dim > 0:
            emb = ops.concatenate(
                [emb, ops.zeros((T, H, W, pad_dim), dtype=emb.dtype)],
                axis=-1,
            )

        emb = ops.reshape(emb, (T * H * W, self.head_dim))
        return ops.cos(emb), ops.sin(emb)

    def get_config(self):
        config = super().get_config()
        config.update({
            "head_dim":      self.head_dim,
            "theta":         self.theta,
            "mrope_section": self.mrope_section,
        })
        return config