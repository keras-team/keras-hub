import keras
from keras import ops


class T5LayerNorm(keras.layers.Layer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.weight = self.add_weight(
            name="weight",
            shape=(input_shape[-1],),
            initializer="ones",
        )
        self.built = True

    def call(self, hidden_states):
        variance = ops.mean(ops.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.epsilon)
        return self.weight * hidden_states
