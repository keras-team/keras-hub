import math

from keras import Layer
from keras import ops


class DetrFrozenBatchNormalization(Layer):
    """BatchNormalization with fixed affine + batch stats.
    Based on https://github.com/facebookresearch/detr/blob/master/models/backbone.py.
    """

    def __init__(self, num_features, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.epsilon = epsilon

    def build(self):
        self.weight = self.add_weight(
            shape=(self.num_features,),
            initializer="ones",
            trainable=False,
            name="weight",
        )
        self.bias = self.add_weight(
            shape=(self.num_features,),
            initializer="zeros",
            trainable=False,
            name="bias",
        )
        self.running_mean = self.add_weight(
            shape=(self.num_features,),
            initializer="zeros",
            trainable=False,
            name="running_mean",
        )
        self.running_var = self.add_weight(
            shape=(self.num_features,),
            initializer="ones",
            trainable=False,
            name="running_var",
        )

    def call(self, inputs):
        weight = ops.reshape(self.weight, (1, 1, 1, -1))
        bias = ops.reshape(self.bias, (1, 1, 1, -1))
        running_mean = ops.reshape(self.running_mean, (1, 1, 1, -1))
        running_var = ops.reshape(self.running_var, (1, 1, 1, -1))

        scale = weight * ops.rsqrt(running_var + self.epsilon)
        bias = bias - running_mean * scale
        return inputs * scale + bias

    def get_config(self):
        config = super().get_config()
        config.update(
            {"num_features": self.num_features, "epsilon": self.epsilon}
        )
        return config


class DetrSinePositionEmbedding(Layer):
    def __init__(
        self, embedding_dim=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def call(self, pixel_mask):
        if pixel_mask is None:
            raise ValueError("No pixel mask provided")
        y_embed = ops.cumsum(pixel_mask, axis=1)
        x_embed = ops.cumsum(pixel_mask, axis=2)
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * self.scale

        dim_t = ops.arange(self.embedding_dim)
        dim_t = self.temperature ** (
            2 * ops.floor(dim_t / 2) / self.embedding_dim
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = ops.stack(
            (ops.sin(pos_x[:, :, :, ::2]), ops.cos(pos_x[:, :, :, 1::2])),
            axis=4,
        )
        pos_y = ops.stack(
            (ops.sin(pos_y[:, :, :, ::2]), ops.cos(pos_y[:, :, :, 1::2])),
            axis=4,
        )

        pos_x = ops.flatten(pos_x, axis=3)
        pos_y = ops.flatten(pos_y, axis=3)

        pos = ops.cat((pos_y, pos_x), axis=3)
        pos = ops.transpose(pos, [0, 3, 1, 2])
        return pos
