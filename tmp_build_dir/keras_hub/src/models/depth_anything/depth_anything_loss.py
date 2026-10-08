import keras
from keras import ops
from keras.src.losses.losses import LossFunctionWrapper


class DepthAnythingLoss(LossFunctionWrapper):
    """Computes the DepthAnything loss between `y_true` & `y_pred`.

    This loss is the Scale-Invariant Logarithmic (SiLog) loss, which is
    widely used for depth estimation tasks.

    See: [Depth Map Prediction from a Single Image using a Multi-Scale Deep Network](https://arxiv.org/abs/1406.2283)

    Args:
        lambd: The weighting factor in the scale-invariant log loss formula.
            Defaults to `0.5`.
        min_depth: Minimum depth value used to filter `y_pred` and `y_true`.
            Defaults to `keras.config.epsilon()`.
        max_depth: Optional maximum depth value used to filter `y_pred` and
            `y_true`. If not specified, there will be no upper bound.
        reduction: Type of reduction to apply to the loss. In almost all cases
            this should be `"sum_over_batch_size"`. Supported options are
            `"sum"`, `"sum_over_batch_size"`, `"mean"`,
            `"mean_with_sample_weight"` or `None`. `"sum"` sums the loss,
            `"sum_over_batch_size"` and `"mean"` sum the loss and divide by the
            sample size, and `"mean_with_sample_weight"` sums the loss and
            divides by the sum of the sample weights. `"none"` and `None`
            perform no aggregation. Defaults to `"sum_over_batch_size"`.
        name: Optional name for the instance.
        dtype: The dtype of the loss's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.
    """

    def __init__(
        self,
        lambd=0.5,
        min_depth=keras.config.epsilon(),
        max_depth=None,
        reduction="sum_over_batch_size",
        name="depth_anything_loss",
        dtype=None,
    ):
        super().__init__(
            silog,
            name=name,
            reduction=reduction,
            dtype=dtype,
            lambd=lambd,
            min_depth=min_depth,
            max_depth=max_depth,
        )


def silog(
    y_true, y_pred, lambd=0.5, min_depth=keras.config.epsilon(), max_depth=None
):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)

    # Apply the valid mask.
    if max_depth is None:
        valid_mask = ops.greater_equal(y_true, min_depth)
    else:
        valid_mask = ops.logical_and(
            ops.greater_equal(y_true, min_depth),
            ops.less_equal(y_true, max_depth),
        )
    y_true = ops.multiply(y_true, valid_mask)
    y_pred = ops.multiply(y_pred, valid_mask)

    diff_log = ops.where(
        valid_mask,
        ops.subtract(ops.log(y_true), ops.log(y_pred)),
        ops.zeros_like(y_true),
    )

    divisor = ops.sum(ops.cast(valid_mask, y_true.dtype), axis=(1, 2, 3))
    mean_power2_diff_log = ops.divide_no_nan(
        ops.sum(ops.power(diff_log, 2), axis=(1, 2, 3)), divisor
    )
    power2_mean_diff_log = ops.power(
        ops.divide_no_nan(ops.sum(diff_log, axis=(1, 2, 3)), divisor), 2
    )
    return ops.sqrt(
        mean_power2_diff_log - ops.multiply(lambd, power2_mean_diff_log)
    )
