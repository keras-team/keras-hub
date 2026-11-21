from keras import backend
from keras import ops


def _bilinear_interpolate(
    feature_maps, roi_batch_ind, y, x, ymask, xmask, height, width, hidden_dim
):
    feature_maps_dtype = backend.standardize_dtype(feature_maps.dtype)
    y = ops.maximum(y, 0.0)
    x = ops.maximum(x, 0.0)
    y_low = ops.cast(y, "int32")
    x_low = ops.cast(x, "int32")
    y_high = ops.where(
        ops.greater_equal(y_low, height - 1), height - 1, y_low + 1
    )
    y_low = ops.where(ops.greater_equal(y_low, height - 1), height - 1, y_low)
    y = ops.where(
        ops.greater_equal(y_low, height - 1),
        ops.cast(y, dtype=feature_maps_dtype),
        y,
    )

    x_high = ops.where(
        ops.greater_equal(x_low, width - 1), width - 1, x_low + 1
    )
    x_low = ops.where(ops.greater_equal(x_low, width - 1), width - 1, x_low)
    x = ops.where(
        ops.greater_equal(x_low, width - 1),
        ops.cast(x, dtype=feature_maps_dtype),
        x,
    )

    ly = ops.subtract(y, y_low)
    lx = ops.subtract(x, x_low)
    hy = ops.subtract(1.0, ly)
    hx = ops.subtract(1.0, lx)

    def masked_index(y, x):
        y = ops.where(ymask[:, None, :], y, 0)
        x = ops.where(xmask[:, None, :], x, 0)
        batch_idx = roi_batch_ind[:, None, None, None, None, None]
        channel_idx = ops.arange(hidden_dim)[None, None, None, None, None, :]
        y_idx = y[:, :, None, :, None, None]
        x_idx = x[:, None, :, None, :, None]

        if backend.backend() == "tensorflow":
            import tensorflow as tf

            # Explicitly broadcast indices to the same shape for XLA
            # compatibility
            common_zero = ops.zeros_like(
                batch_idx + y_idx + x_idx + channel_idx
            )
            batch_idx = batch_idx + common_zero
            y_idx = y_idx + common_zero
            x_idx = ops.transpose(
                ops.transpose(x_idx, (0, 2, 1, 4, 3, 5)) + common_zero,
                (0, 2, 1, 4, 3, 5),
            )
            channel_idx = channel_idx + common_zero
            indices = ops.stack([batch_idx, y_idx, x_idx, channel_idx], axis=-1)
            indices = ops.cast(indices, "int32")
            return tf.gather_nd(feature_maps, indices)
        else:
            return feature_maps[
                batch_idx,
                y_idx,
                x_idx,
                channel_idx,
            ]

    v1 = masked_index(y_low, x_low)
    v2 = masked_index(y_low, x_high)
    v3 = masked_index(y_high, x_low)
    v4 = masked_index(y_high, x_high)

    def outer_prod(y, x):
        return ops.multiply(
            y[:, :, None, :, None, None], x[:, None, :, None, :, None]
        )

    w1 = outer_prod(hy, hx)
    w2 = outer_prod(hy, lx)
    w3 = outer_prod(ly, hx)
    w4 = outer_prod(ly, lx)

    val = ops.add(
        ops.add(ops.multiply(w1, v1), ops.multiply(w2, v2)),
        ops.add(ops.multiply(w3, v3), ops.multiply(w4, v4)),
    )
    return val


def roi_align_torch(
    feature_maps,
    rois,
    output_size,
    spatial_scale=1.0,
    aligned=False,
):
    import torchvision

    dtype = backend.standardize_dtype(feature_maps.dtype)
    need_cast = False
    if dtype == "bfloat16":
        # torchvision.ops.roi_align does not support bfloat16.
        feature_maps = ops.cast(feature_maps, "float32")
        rois = ops.cast(rois, "float32")
        need_cast = True

    output = ops.transpose(
        torchvision.ops.roi_align(
            ops.transpose(feature_maps, (0, 3, 1, 2)),
            rois,
            output_size,
            spatial_scale=spatial_scale,
            aligned=aligned,
        ),
        (0, 2, 3, 1),
    )
    if need_cast:
        output = ops.cast(output, dtype)
    return output


def roi_align(
    feature_maps,
    rois,
    output_size,
    height,
    width,
    hidden_dim,
    spatial_scale=1.0,
    aligned=False,
):
    # Use torchvision's optimized roi_align implementation.
    if backend.backend() == "torch":
        return roi_align_torch(
            feature_maps,
            rois,
            output_size,
            spatial_scale=spatial_scale,
            aligned=aligned,
        )

    original_dtype = backend.standardize_dtype(feature_maps.dtype)
    out_h, out_w = output_size[0], output_size[1]

    feature_maps = ops.cast(feature_maps, "float32")
    rois = ops.cast(rois, "float32")

    ph = ops.arange(out_h, dtype="float32")
    pw = ops.arange(out_w, dtype="float32")

    # input: [N, C, H, W]
    # rois: [K, 5]

    roi_batch_ind = ops.cast(rois[:, 0], "int32")
    offset = 0.5 if aligned else 0.0
    roi_start_w = ops.subtract(ops.multiply(rois[:, 1], spatial_scale), offset)
    roi_start_h = ops.subtract(ops.multiply(rois[:, 2], spatial_scale), offset)
    roi_end_w = ops.subtract(ops.multiply(rois[:, 3], spatial_scale), offset)
    roi_end_h = ops.subtract(ops.multiply(rois[:, 4], spatial_scale), offset)

    roi_width = ops.subtract(roi_end_w, roi_start_w)
    roi_height = ops.subtract(roi_end_h, roi_start_h)
    if not aligned:
        roi_width = ops.maximum(roi_width, 1.0)
        roi_height = ops.maximum(roi_height, 1.0)

    bin_size_h = ops.divide(roi_height, out_h)
    bin_size_w = ops.divide(roi_width, out_w)

    roi_bin_grid_h = ops.ceil(ops.divide(roi_height, out_h))
    roi_bin_grid_w = ops.ceil(ops.divide(roi_width, out_w))

    count = ops.maximum(ops.multiply(roi_bin_grid_h, roi_bin_grid_w), 1.0)
    iy = ops.arange(height, dtype="float32")
    ix = ops.arange(width, dtype="float32")
    ymask = ops.less(iy[None, :], roi_bin_grid_h[:, None])
    xmask = ops.less(ix[None, :], roi_bin_grid_w[:, None])

    def from_k(t):
        return t[:, None, None]

    y = ops.add(
        ops.add(
            from_k(roi_start_h),
            ops.multiply(ph[None, :, None], from_k(bin_size_h)),
        ),
        ops.multiply(
            ops.cast(ops.add(iy[None, None, :], 0.5), dtype="float32"),
            from_k(ops.divide(bin_size_h, roi_bin_grid_h)),
        ),
    )
    x = ops.add(
        ops.add(
            from_k(roi_start_w),
            ops.multiply(pw[None, :, None], from_k(bin_size_w)),
        ),
        ops.multiply(
            ops.cast(ops.add(ix[None, None, :], 0.5), dtype="float32"),
            from_k(ops.divide(bin_size_w, roi_bin_grid_w)),
        ),
    )
    val = _bilinear_interpolate(
        feature_maps,
        roi_batch_ind,
        y,
        x,
        ymask,
        xmask,
        height,
        width,
        hidden_dim,
    )
    val = ops.where(ymask[:, None, None, :, None, None], val, 0.0)
    val = ops.where(xmask[:, None, None, None, :, None], val, 0.0)

    output = ops.sum(val, axis=(3, 4))
    output = ops.divide(output, count[:, None, None, None])
    return ops.cast(output, original_dtype)
