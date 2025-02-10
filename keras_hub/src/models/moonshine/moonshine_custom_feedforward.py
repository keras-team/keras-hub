from keras import layers
from keras import ops


class FFLinearGelu(layers.Layer):
    def __init__(self, dim, ff_mult, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(dim * ff_mult, use_bias=True)
        self.gelu = layers.Activation("gelu")
        self.dense2 = layers.Dense(dim, use_bias=True)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.gelu(x)
        x = self.dense2(x)
        return x


class FFSwiGLU(layers.Layer):
    def __init__(self, dim, ff_mult, **kwargs):
        super().__init__(**kwargs)
        self.ff_proj = layers.Dense(dim * ff_mult * 2, use_bias=True)
        self.ff_act = layers.Activation("silu")
        self.ff_out = layers.Dense(dim, use_bias=True)

    def call(self, inputs):
        x = self.ff_proj(inputs)
        x, gate = ops.split(x, num_or_size_splits=2, axis=-1)
        x = x * self.ff_act(gate)
        x = self.ff_out(x)
        return x
