import keras
from keras import ops


class RMSNormalization(keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            trainable=True,
            shape=(input_shape[-1],),
            initializer="zeros",
        )
        self.built = True

    def call(self, x):
        # Always compute normalization in float32.
        x = ops.cast(x, "float32")
        scale = ops.cast(self.scale, "float32")
        var = ops.mean(ops.square(x), axis=-1, keepdims=True)
        normed_inputs = x * ops.reciprocal(ops.sqrt(var + self.epsilon))
        normed_inputs = normed_inputs * (1 + scale)
        return ops.cast(normed_inputs, self.compute_dtype)
