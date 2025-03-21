import keras

from keras_hub.src.models.gemma.rms_normalization import RMSNormalization
from keras_hub.src.utils.keras_utils import clone_initializer


class Gemma3VisionOutputEncoder(keras.layers.Layer):
    def __init__(
        self,
        output_dim,
        layer_norm_epsilon=1e-6,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.layer_norm_epsilon = layer_norm_epsilon
        self.output_dim = output_dim

        self._kernel_initializer = keras.initializers.get(
            clone_initializer(kernel_initializer)
        )

    def build(self, input_shape):
        self.vision_soft_embedding_norm = RMSNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="vision_soft_embedding_norm",
        )
        self.vision_soft_embedding_norm.build(input_shape)

        self.vision_input_projection = keras.layers.Dense(
            units=self.output_dim,
            use_bias=False,
            kernel_initializer=self._kernel_initializer,
            dtype=self.dtype_policy,
            name="vision_input_projection",
        )
        self.vision_input_projection.build(input_shape)

    def call(self, inputs):
        x = self.vision_soft_embedding_norm(inputs)
        x = self.vision_input_projection(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self._kernel_initializer
                ),
            }
        )

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.output_dim,)
