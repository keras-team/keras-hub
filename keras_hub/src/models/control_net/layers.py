import keras


class PaddedConv2D(keras.layers.Layer):
    def __init__(self, channels, kernel_size, padding=0, stride=1, name=None):
        super().__init__()
        self.padding2d = keras.layers.ZeroPadding2D((padding, padding), name=name)
        self.conv2d = keras.layers.Conv2D(
            channels, kernel_size, strides=(stride, stride), name=name
        )

    def call(self, x):
        x = self.padding2d(x)
        return self.conv2d(x)


class GEGLU(keras.layers.Layer):
    def __init__(self, dim_out, name=None):
        super().__init__()
        self.proj = keras.layers.Dense(dim_out * 2, name=name)
        self.dim_out = dim_out

    def call(self, x):
        xp = self.proj(x)
        x, gate = xp[..., : self.dim_out], xp[..., self.dim_out :]
        return x * gelu(gate)


def gelu(x):
    tanh_res = keras.activations.tanh(x * 0.7978845608 * (1 + 0.044715 * (x**2)))
    return 0.5 * x * (1 + tanh_res)


def quick_gelu(x):
    return x * keras.ops.sigmoid(x * 1.702)


"""def apply_seq(x, layers):
    for l in layers:
        x = l(x)
    return x"""


def apply_seq(x, seq_layer):
    if isinstance(seq_layer, keras.Sequential):
        x = seq_layer(x)
    else:
        for l in seq_layer:
            x = l(x)
    return x


def td_dot(a, b):
    aa = keras.ops.reshape(a, (-1, a.shape[2], a.shape[3]))
    bb = keras.ops.reshape(b, (-1, b.shape[2], b.shape[3]))
    cc = keras.backend.batch_dot(aa, bb)
    return keras.ops.reshape(cc, (-1, a.shape[1], cc.shape[1], cc.shape[2]))
