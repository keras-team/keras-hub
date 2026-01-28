import keras

from keras_hub.src.models.mobilenet.util import adjust_channels


def num_groups(group_size, channels):
    if not group_size:
        return 1
    else:
        if channels % group_size != 0:
            raise ValueError(
                f"Number of channels ({channels}) must be divisible by "
                "group size ({group_size})."
            )
        return channels // group_size


def parse_ksize(ss):
    if ss.isdigit():
        return int(ss)
    else:
        return [int(k) for k in ss.split(".")]


def round_channels(
    channels, multiplier=1.0, divisor=8, channel_min=None, round_limit=0.9
):
    if not multiplier:
        return channels
    return adjust_channels(channels * multiplier, divisor, channel_min)


def feature_take_indices(num_stages, indices):
    if not isinstance(indices, (tuple, list)):
        indices = (indices,)
    if any(i < 0 for i in indices):
        indices = [i if i >= 0 else num_stages + i for i in indices]
    return indices, max(indices)


class SelectAdaptivePool2d(keras.layers.Layer):
    """A layer that selects and applies a 2D adaptive pooling strategy.

    This layer supports various pooling types like average, max, or a
    combination of both. It can also flatten the output.

    Args:
        pool_type: str. The type of pooling to apply. One of `"avg"`, `"max"`,
            `"avgmax"`, `"catavgmax"`, or `""` (identity).
        flatten: bool. If `True`, the output is flattened after pooling.
        data_format: str. The format of the input data, either
            `"channels_last"` or `"channels_first"`.
        channel_axis: int. The axis representing the channels in the input
            tensor.
    """

    def __init__(
        self,
        pool_type="avg",
        flatten=False,
        data_format=None,
        channel_axis=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.pool_type = pool_type.lower()
        self.flatten = flatten
        self.data_format = data_format
        self.channels_axis = channel_axis
        self.pool = None
        self.pool_avg = None
        self.pool_max = None
        self.pool_cat = None
        self.flatten_layer = None
        if self.pool_type not in ("avg", "max", "avgmax", "catavgmax", ""):
            raise ValueError(f"Invalid pool type: {self.pool_type}")

    def build(self, input_shape):
        if self.pool_type == "avg":
            self.pool = keras.layers.GlobalAveragePooling2D(
                data_format=self.data_format,
                keepdims=not self.flatten,
                dtype=self.dtype_policy,
            )
        elif self.pool_type == "max":
            self.pool = keras.layers.GlobalMaxPooling2D(
                data_format=self.data_format,
                keepdims=not self.flatten,
                dtype=self.dtype_policy,
            )
        elif self.pool_type in ("avgmax", "catavgmax"):
            self.pool_avg = keras.layers.GlobalAveragePooling2D(
                data_format=self.data_format,
                keepdims=not self.flatten,
                dtype=self.dtype_policy,
            )
            self.pool_max = keras.layers.GlobalMaxPooling2D(
                data_format=self.data_format,
                keepdims=not self.flatten,
                dtype=self.dtype_policy,
            )
            if self.pool_type == "catavgmax":
                axis = 1 if self.data_format == "channels_first" else -1
                self.pool_cat = keras.layers.Concatenate(
                    axis=axis, dtype=self.dtype_policy
                )
        elif not self.pool_type:
            self.pool = keras.layers.Identity(dtype=self.dtype_policy)
            if self.flatten:
                self.flatten_layer = keras.layers.Flatten(
                    dtype=self.dtype_policy
                )
        super().build(input_shape)

    def call(self, x):
        if self.pool_type in ("avg", "max"):
            return self.pool(x)
        elif self.pool_type == "avgmax":
            x_avg = self.pool_avg(x)
            x_max = self.pool_max(x)
            return 0.5 * (x_avg + x_max)
        elif self.pool_type == "catavgmax":
            x_avg = self.pool_avg(x)
            x_max = self.pool_max(x)
            return self.pool_cat([x_avg, x_max])
        elif not self.pool_type:
            x = self.pool(x)
            if self.flatten_layer:
                x = self.flatten_layer(x)
            return x
        return x

    def feat_mult(self):
        return 2 if self.pool_type == "catavgmax" else 1

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pool_type": self.pool_type,
                "flatten": self.flatten,
                "data_format": self.data_format,
            }
        )
        return config
