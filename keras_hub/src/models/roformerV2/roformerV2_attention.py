import keras
from keras import initializers
from keras import ops


class RoformrPositionalEmbedding(keras.layers.Layer):
    """
    native rotary implement by jianlin su
    https://github.com/bojone/bert4keras
    """

    def __init__(self, output_dim, max_wavelength=10000, **kwargs):
        super().__init__(**kwargs)
        self.max_wavelength = max_wavelength
        self.output_dim = output_dim

    def call(self, tensors: list):
        input_shape = ops.shape(tensors[0])
        seq_len = input_shape[1]
        position_ids = ops.arange(0, seq_len, dtype=tensors[0].dtype)[None]
        embeddings = self.sinusoidal_embeddings(
            position_ids, self.output_dim, self.max_wavelength
        )
        embeddings = ops.cast(embeddings, self.compute_dtype)

        ndim = ops.ndim(tensors[0])
        sinusoidal = self.align(embeddings, [0, 1, -1], ndim)
        cos_pos = ops.repeat(sinusoidal[..., 1::2], 2, -1)
        sin_pos = ops.repeat(sinusoidal[..., ::2], 2, -1)
        outputs = []
        for tensor in tensors:
            tensor2 = ops.stack([-tensor[..., 1::2], tensor[..., ::2]], ndim)
            tensor2 = ops.reshape(tensor2, ops.shape(tensor))
            outputs.append(tensor * cos_pos + tensor2 * sin_pos)
        return outputs[0] if len(outputs) == 1 else outputs

    def align(self, tensor, axes, ndim=None):
        ndim = ndim or max(axes) + 1
        indices = [None] * ndim
        for i in axes:
            indices[i] = slice(None)
        return tensor[indices]

    def sinusoidal_embeddings(self, pos, dim, base=10000):
        assert dim % 2 == 0
        indices = ops.arange(0, dim // 2, dtype="float32")
        indices = ops.power(ops.cast(base, dtype="float32"), -2 * indices / dim)
        embeddings = ops.einsum("...,d->...d", pos, indices)
        embeddings = ops.stack(
            [ops.sin(embeddings), ops.cos(embeddings)], axis=-1
        )
        shape = list(ops.shape(embeddings))
        embeddings = ops.reshape(embeddings, shape[:-2] + [-1])
        return embeddings

    def get_config(self):
        config = {
            "out_dim": self.out_dim,
            "max_wavelength": self.max_wavelength,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras.saving.register_keras_serializable(package="keras_hub")
class RoformerAttention(keras.layers.Layer):
    """
    MultiHeadAttention by roformerV2
    modifity from native implement
    https://github.com/bojone/bert4keras
    """

    def __init__(
        self,
        heads,
        head_size,
        out_dim=None,
        use_bias=False,
        max_wavelength=10000,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.heads = heads
        self.head_size = head_size
        self.out_dim = out_dim or heads * head_size
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.max_wavelength = max_wavelength

    def build(self, input_shape):
        super().build(input_shape)
        self.q_dense = keras.layers.Dense(
            units=self.head_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name="q_dense_layer",
        )
        self.k_dense = keras.layers.Dense(
            units=self.head_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name="k_dense_layer",
        )
        self.v_dense = keras.layers.Dense(
            units=self.head_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name="v_dense_layer",
        )
        self.o_dense = keras.layers.Dense(
            units=self.out_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name="o_dense_layer",
        )

        self.rotary_embedding_layer = RoformrPositionalEmbedding(
            self.head_size, self.max_wavelength
        )

    def call(self, x, attention_mask=None):
        qw = self.q_dense(x)
        kw = self.k_dense(x)
        vw = self.v_dense(x)

        b, s = ops.shape(qw)[:2]
        qw = ops.reshape(qw, (b, s, self.heads, self.head_size))
        kw = ops.reshape(kw, (b, s, self.heads, self.head_size))
        vw = ops.reshape(vw, (b, s, self.heads, self.head_size))

        qw, kw = self.rotary_embedding_layer([qw, kw])
        if hasattr(keras.config, "is_flash_attention_enabled"):
            flash_attention = keras.config.is_flash_attention_enabled()
        else:
            flash_attention = False
        attention_mask = ops.reshape(attention_mask, [b, 1, s, 1])
        o = ops.dot_product_attention(
            qw, kw, vw, mask=attention_mask, flash_attention=flash_attention
        )

        o = self.o_dense(ops.reshape(o, [b, s, -1]))

        return o

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "heads": self.heads,
            "head_size": self.head_size,
            "out_dim": self.out_dim,
            "use_bias": self.use_bias,
            "max_wavelength": self.max_wavelength,
            "kernel_initializer": initializers.serialize(
                self.kernel_initializer
            ),
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
