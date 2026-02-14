import keras
import tensorflow as tf


class ControlNetBackbone(keras.Model):
    """Lightweight conditioning encoder for ControlNet."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.down1 = keras.layers.Conv2D(
            64, kernel_size=3, padding="same", activation="relu"
        )
        self.down2 = keras.layers.Conv2D(
            128, kernel_size=3, padding="same", activation="relu"
        )
        self.down3 = keras.layers.Conv2D(
            256, kernel_size=3, padding="same", activation="relu"
        )

        self.pool = keras.layers.MaxPooling2D(pool_size=2)

    def build(self, input_shape):
        self.down1.build(input_shape)
        b, h, w, c = input_shape
        half_shape = (b, h // 2, w // 2, 64)
        self.down2.build(half_shape)
        quarter_shape = (b, h // 4, w // 4, 128)
        self.down3.build(quarter_shape)

        super().build(input_shape)

    def call(self, x):
        f1 = self.down1(x)
        p1 = self.pool(f1)

        f2 = self.down2(p1)
        p2 = self.pool(f2)

        f3 = self.down3(p2)

        return {
            "scale_1": f1,
            "scale_2": f2,
            "scale_3": f3,
        }
