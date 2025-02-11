from keras import layers
from keras import models
from keras import ops


class FFLinearGelu(layers.Layer):
    def __init__(self, dim, ff_mult, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.ff_mult = ff_mult
        self.dense1 = layers.Dense(dim * ff_mult, use_bias=True)
        self.gelu = layers.Activation("gelu")
        self.dense2 = layers.Dense(dim, use_bias=True)

    def build(self, input_shape):
        # This method is called when the layer is first used
        # input_shape will be (batch_size, seq_len, dim).
        super().build(input_shape)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.gelu(x)
        return self.dense2(x)

    def compute_output_shape(self, input_shape):
        # Preserve sequence length dimension while changing feature dimension.
        return (input_shape[0], input_shape[1], self.dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "ff_mult": self.ff_mult,
            }
        )
        return config


class FFSwiGLU(layers.Layer):
    def __init__(self, dim, ff_mult, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.ff_mult = ff_mult
        self.ff_proj = layers.Dense(dim * ff_mult * 2, use_bias=True)
        self.ff_act = layers.Activation("silu")
        self.ff_out = layers.Dense(dim, use_bias=True)

    def build(self, input_shape):
        # Change: Create a functional model in build() to avoid cycles.
        inputs = layers.Input(shape=input_shape[1:])
        x = self.ff_proj(inputs)
        x_parts = ops.split(x, 2, axis=-1)
        gate = x_parts[1]
        x = x_parts[0] * self.ff_act(gate)
        x = self.ff_out(x)
        self.ff_model = models.Model(inputs=inputs, outputs=x)
        super().build(input_shape)

    def call(self, inputs):
        return self.ff_model(inputs)
