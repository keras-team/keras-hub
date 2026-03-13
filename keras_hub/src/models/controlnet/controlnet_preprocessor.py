import keras
import tensorflow as tf


class ControlNetPreprocessor(keras.layers.Layer):
    def __init__(self, target_size=(512, 512), **kwargs):
        super().__init__(**kwargs)
        self.target_size = tuple(target_size)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)

        if x.shape.rank != 4:
            raise ValueError("Inputs must be a 4D tensor (batch, height, width, channels).")

        x = tf.image.resize(x, self.target_size)
        x = tf.cast(x, tf.float32)

        max_val = tf.reduce_max(x)
        x = tf.cond(
            max_val > 1.0,
            lambda: x / 255.0,
            lambda: x,
        )

        return x

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            self.target_size[0],
            self.target_size[1],
            input_shape[-1],
        )

    def get_config(self):
        config = super().get_config()
        config.update({"target_size": self.target_size})
        return config
