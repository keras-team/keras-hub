from keras import Sequential
from keras import layers
from keras import models


class AudioPreprocessor(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        # Define inputs with variable time length and one channel.
        inputs = layers.Input(shape=[None, 1])
        conv1 = layers.Conv1D(
            filters=dim,
            kernel_size=127,
            strides=64,
            use_bias=False,
        )
        tanh = layers.Activation("tanh")
        group_norm = layers.GroupNormalization(groups=1, axis=-1, epsilon=1e-5)
        conv2 = layers.Conv1D(
            filters=2 * dim, kernel_size=7, strides=3, padding="valid"
        )
        gelu1 = layers.Activation("gelu")
        conv3 = layers.Conv1D(
            filters=dim, kernel_size=3, strides=2, padding="valid"
        )
        gelu2 = layers.Activation("gelu")
        preprocess = Sequential(
            [conv1, tanh, group_norm, conv2, gelu1, conv3, gelu2]
        )
        outputs = preprocess(inputs)
        self.preprocess_model = models.Model(inputs=inputs, outputs=outputs)
        self.dim = dim

    def call(self, inputs):
        return self.preprocess_model(inputs)

    def set_weights(self, weights):
        self.preprocess_model.set_weights(weights)
