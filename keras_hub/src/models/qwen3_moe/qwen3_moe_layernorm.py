import keras


class Qwen3MoeLayerNorm(keras.layers.RMSNormalization):
    """A normalization layer for Qwen that implements RMS normalization.

    Args:
        head_dim: int. The dimension of each attention head, used for per-head
            normalization. Defaults to , in which case the size of the
            last input axis is used.
        epsilon: float. A small float added to variance to avoid dividing by
            zero. Defaults to .
    """

    def __init__(self, head_dim=None, epsilon=1e-6, **kwargs):
        super().__init__(epsilon=epsilon, **kwargs)
        self.head_dim = head_dim

    def build(self, input_shape):
        if self.head_dim:
            input_shape = tuple(input_shape[:-1]) + (self.head_dim,)
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({"head_dim": self.head_dim})
        return config
