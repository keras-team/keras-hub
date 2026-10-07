from keras import backend
from keras import ops

from keras_hub.src.utils.keras_utils import standardize_data_format


def interpolate(x, size, data_format=None):
    """Performs a backend-agnostic version of Torch's `F.interpolate`.

    Args:
        x: A 4D image tensor.
        size: A tuple of 2 integers, `(height, width)`.
        data_format: One of `channels_last` or `channels_first`.
    """
    data_format = standardize_data_format(data_format)
    if backend.backend() == "jax":
        import jax

        if data_format == "channels_first":
            x = ops.transpose(x, (0, 2, 3, 1))
        scale = ops.convert_to_tensor(
            [
                (size[0] - 1.0) / (x.shape[1] - 1.0),
                (size[1] - 1.0) / (x.shape[2] - 1.0),
            ]
        )
        translation = -(scale / 2.0 - 0.5)
        x = jax.image.scale_and_translate(
            x,
            (x.shape[0], *size, x.shape[-1]),
            method="bilinear",
            scale=scale,
            spatial_dims=(1, 2),
            translation=translation,
            antialias=False,
        )
        if data_format == "channels_first":
            x = ops.transpose(x, (0, 3, 1, 2))
    elif backend.backend() == "tensorflow":
        import tensorflow as tf

        if data_format == "channels_first":
            x = ops.transpose(x, (0, 2, 3, 1))
        x = tf.compat.v1.image.resize(
            x,
            size=size,
            method="bilinear",
            align_corners=True,
        )
        if data_format == "channels_first":
            x = ops.transpose(x, (0, 3, 1, 2))
    elif backend.backend() == "torch":
        import torch.nn.functional as F

        if data_format == "channels_last":
            x = ops.transpose(x, (0, 3, 1, 2))
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        if data_format == "channels_last":
            x = ops.transpose(x, (0, 2, 3, 1))
    else:
        raise NotImplementedError(f"Unsupported backend: {backend.backend()}")
    return x
