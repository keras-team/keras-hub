import keras
from keras import initializers
from keras import ops


class RoformerNorm(keras.layers.Layer):
    """A normalization layer for Roformer that implements RMS normalization."""

    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[-1]
        self.scale = self.add_weight(
            name="scale",
            trainable=True,
            shape=(dim,),
            initializer="ones",
            dtype=self.variable_dtype,
        )
        self.built = True

    def call(self, x):
        x = ops.cast(x, "float32")
        var = ops.mean(ops.power(x, 2), axis=-1, keepdims=True)
        x = x * ops.rsqrt(var + self.epsilon)
        return ops.cast(x * self.scale, self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


class RoformrPositionalEmbedding(keras.layers.Layer):
    """Native rotary implement by jianlin su
    from native implement
    https://github.com/bojone/bert4keras

    """

    def __init__(self, output_dim, max_wavelength=10000, **kwargs):
        super().__init__(**kwargs)
        self.max_wavelength = max_wavelength
        self.output_dim = output_dim

    def call(self, tensors):
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
        if keras.config.backend() == "jax":
            return tensor[tuple(indices)]
        return tensor[indices]

    def sinusoidal_embeddings(self, pos, dim, base=10000):
        if dim % 2 != 0:
            raise ("Dimension must be even")

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
        config = super().get_config()
        config.update(
            {
                "out_dim": self.out_dim,
                "max_wavelength": self.max_wavelength,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class RoformerAttention(keras.layers.Layer):
    """MultiHeadAttention by roformerV2

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
            dtype=self.dtype_policy,
        )
        self.q_dense.build(input_shape)

        self.k_dense = keras.layers.Dense(
            units=self.head_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name="k_dense_layer",
            dtype=self.dtype_policy,
        )
        self.k_dense.build(input_shape)

        self.v_dense = keras.layers.Dense(
            units=self.head_size * self.heads,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name="v_dense_layer",
            dtype=self.dtype_policy,
        )
        self.v_dense.build(input_shape)

        self.o_dense = keras.layers.Dense(
            units=self.out_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            name="o_dense_layer",
            dtype=self.dtype_policy,
        )
        self.o_dense.build([None, None, self.head_size * self.heads])

        self.rotary_embedding_layer = RoformrPositionalEmbedding(
            self.head_size, self.max_wavelength, dtype=self.dtype_policy
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

        qw, kw = self.rotary_embedding_layer([qw, kw])
        if keras.__version__ < "3.6":
            raise ("Please make sure your Keras version is >=3.6.")
        flash_attention = keras.config.is_flash_attention_enabled()
        attention_mask = ops.reshape(attention_mask, [b, 1, s, 1])
        if keras.config.backend() == "torch":
            attention_mask = ops.repeat(attention_mask, s, -1)
            attention_mask = ops.transpose(attention_mask, [0, 1, 3, 2])
        o = ops.dot_product_attention(
            qw, kw, vw, mask=attention_mask, flash_attention=flash_attention
        )

        return self.o_dense(ops.reshape(o, [b, s, -1]))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "heads": self.heads,
                "head_size": self.head_size,
                "out_dim": self.out_dim,
                "use_bias": self.use_bias,
                "max_wavelength": self.max_wavelength,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
            }
        )
        return config
