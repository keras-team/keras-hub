import keras


def inverse_sigmoid(x, eps=1e-5):
    """Computes the inverse sigmoid (logit) function.

    This function computes the inverse of the sigmoid function, also known as
    the logit function. It is used in D-FINE to transform bounding box
    coordinates from the `[0, 1]` range back to logits, for example in
    `DFineContrastiveDenoisingGroupGenerator` and `DFineDecoder`.

    Args:
        x: Tensor, Input tensor with values in `[0, 1]`.
        eps: float, Small epsilon value to prevent numerical instability
            at the boundaries. Default is `1e-5`.

    Returns:
        Tensor: The inverse sigmoid of the input tensor.
    """
    x = keras.ops.clip(x, 0, 1)
    x1 = keras.ops.clip(x, eps, 1.0 - eps)
    x2 = 1 - x
    x2 = keras.ops.clip(x2, eps, 1.0 - eps)
    return keras.ops.log(x1 / x2)


def grid_sample(data, grid, align_corners=False, height=None, width=None):
    """Samples data at specified grid locations using bilinear interpolation.

    This function performs bilinear interpolation to sample data at arbitrary
    grid locations. It is a core component of the deformable attention
    mechanism, used within `multi_scale_deformable_attention_v2`.

    Args:
        data: Tensor, Input data tensor of shape `[batch, channels, height,
            width]`.
        grid: Tensor, Grid coordinates of shape `[batch, out_height, out_width,
            2]`. The last dimension contains `(x, y)` coordinates normalized to
            `[-1, 1]`.
        align_corners: bool, If `True`, align corners for coordinate mapping.
            Default is `False`.
        height: int, optional, Override height for coordinate normalization.
        width: int, optional, Override width for coordinate normalization.

    Returns:
        Tensor: Sampled data of shape `[batch, channels, out_height,
            out_width]`.
    """
    num_batch, _, data_height, data_width = keras.ops.shape(data)
    _, out_height, out_width, _ = keras.ops.shape(grid)
    dtype = data.dtype
    grid_x_norm = grid[..., 0]
    grid_y_norm = grid[..., 1]
    h_in = height if height is not None else data_height
    w_in = width if width is not None else data_width
    height_f = keras.ops.cast(h_in, dtype=dtype)
    width_f = keras.ops.cast(w_in, dtype=dtype)
    if align_corners:
        x_unnorm = (grid_x_norm + 1) / 2 * (width_f - 1)
        y_unnorm = (grid_y_norm + 1) / 2 * (height_f - 1)
    else:
        x_unnorm = ((grid_x_norm + 1) / 2 * width_f) - 0.5
        y_unnorm = ((grid_y_norm + 1) / 2 * height_f) - 0.5
    x0 = keras.ops.floor(x_unnorm)
    y0 = keras.ops.floor(y_unnorm)
    x1 = x0 + 1
    y1 = y0 + 1
    w_y0_val = y1 - y_unnorm
    w_y1_val = y_unnorm - y0
    w_x0_val = x1 - x_unnorm
    w_x1_val = x_unnorm - x0
    data_permuted = keras.ops.transpose(data, (0, 2, 3, 1))

    def gather_padded(
        data_p,
        y_coords,
        x_coords,
        actual_data_height,
        actual_data_width,
        override_height=None,
        override_width=None,
    ):
        y_coords_int = keras.ops.cast(y_coords, "int32")
        x_coords_int = keras.ops.cast(x_coords, "int32")

        y_oob = keras.ops.logical_or(
            y_coords_int < 0, y_coords_int >= actual_data_height
        )
        x_oob = keras.ops.logical_or(
            x_coords_int < 0, x_coords_int >= actual_data_width
        )
        oob_mask = keras.ops.logical_or(y_oob, x_oob)

        y_coords_clipped = keras.ops.clip(
            y_coords_int, 0, actual_data_height - 1
        )
        x_coords_clipped = keras.ops.clip(
            x_coords_int, 0, actual_data_width - 1
        )

        _width_for_indexing = (
            override_width if override_width is not None else actual_data_width
        )

        if override_height is not None and override_width is not None:
            data_flat = keras.ops.reshape(
                data_p,
                (
                    num_batch,
                    override_height * override_width,
                    keras.ops.shape(data_p)[-1],
                ),
            )
        else:
            data_flat = keras.ops.reshape(
                data_p, (num_batch, -1, keras.ops.shape(data_p)[-1])
            )
        y_coords_flat = keras.ops.reshape(
            y_coords_clipped, (num_batch, out_height * out_width)
        )
        x_coords_flat = keras.ops.reshape(
            x_coords_clipped, (num_batch, out_height * out_width)
        )
        indices = y_coords_flat * _width_for_indexing + x_coords_flat

        num_elements_per_batch = keras.ops.shape(data_flat)[1]
        batch_offsets = (
            keras.ops.arange(num_batch, dtype=indices.dtype)
            * num_elements_per_batch
        )
        batch_offsets = keras.ops.reshape(batch_offsets, (num_batch, 1))
        absolute_indices = indices + batch_offsets
        data_reshaped_for_gather = keras.ops.reshape(
            data_flat, (-1, keras.ops.shape(data_flat)[-1])
        )
        gathered = keras.ops.take(
            data_reshaped_for_gather, absolute_indices, axis=0
        )
        gathered = keras.ops.reshape(
            gathered, (num_batch, out_height, out_width, -1)
        )
        oob_mask_expanded = keras.ops.expand_dims(oob_mask, axis=-1)
        gathered_values = gathered * keras.ops.cast(
            keras.ops.logical_not(oob_mask_expanded), dtype=gathered.dtype
        )
        return gathered_values

    batch_indices = keras.ops.arange(0, num_batch, dtype="int32")
    batch_indices = keras.ops.reshape(batch_indices, (num_batch, 1, 1))
    batch_indices = keras.ops.tile(batch_indices, (1, out_height, out_width))
    val_y0_x0 = gather_padded(data_permuted, y0, x0, h_in, w_in, height, width)
    val_y0_x1 = gather_padded(data_permuted, y0, x1, h_in, w_in, height, width)
    val_y1_x0 = gather_padded(data_permuted, y1, x0, h_in, w_in, height, width)
    val_y1_x1 = gather_padded(data_permuted, y1, x1, h_in, w_in, height, width)
    interp_val = (
        val_y0_x0 * keras.ops.expand_dims(w_y0_val * w_x0_val, -1)
        + val_y0_x1 * keras.ops.expand_dims(w_y0_val * w_x1_val, -1)
        + val_y1_x0 * keras.ops.expand_dims(w_y1_val * w_x0_val, -1)
        + val_y1_x1 * keras.ops.expand_dims(w_y1_val * w_x1_val, -1)
    )

    return keras.ops.transpose(interp_val, (0, 3, 1, 2))


def multi_scale_deformable_attention_v2(
    value,
    dynamic_spatial_shapes,
    sampling_locations,
    attention_weights,
    num_points_list,
    slice_sizes,
    spatial_shapes_list,
    num_levels,
    num_queries,
    method="default",
):
    """Computes multi-scale deformable attention mechanism.

    This function implements the core of the multi-scale deformable attention
    mechanism used in `DFineMultiScaleDeformableAttention`. It samples features
    at multiple scales and locations based on learned attention weights and
    sampling locations.

    Args:
        value: Tensor, Feature values of shape `[batch, seq_len, num_heads,
            hidden_dim]`.
        dynamic_spatial_shapes: Tensor, Spatial shapes for each level.
        sampling_locations: Tensor, Sampling locations of shape
            `[batch, num_queries, num_heads, num_levels, num_points, 2]`.
        attention_weights: Tensor, Attention weights of shape `[batch,
            num_queries, num_heads, total_points]`.
        num_points_list: list, Number of sampling points for each level.
        slice_sizes: list, Sizes for slicing the value tensor.
        spatial_shapes_list: list, Spatial shapes for each level.
        num_levels: int, Number of feature levels.
        num_queries: int, Number of queries.
        method: str, Sampling method, either `"default"` or `"discrete"`.
            Default is `"default"`.

    Returns:
        Tensor: Output features of shape `[batch, num_queries, num_heads *
            hidden_dim]`.
    """
    value_shape = keras.ops.shape(value)
    batch_size = value_shape[0]
    num_heads = value_shape[2]
    hidden_dim = value_shape[3]
    sampling_shape = keras.ops.shape(sampling_locations)
    num_levels_from_shape = sampling_shape[3]
    num_points_from_shape = sampling_shape[4]
    permuted_value = keras.ops.transpose(value, axes=(0, 2, 3, 1))
    seq_len = value_shape[1]
    flattened_value = keras.ops.reshape(
        permuted_value, (-1, hidden_dim, seq_len)
    )
    value_chunk_sizes = keras.ops.array(slice_sizes, dtype="int32")
    cum_sizes = keras.ops.concatenate(
        [
            keras.ops.zeros((1,), dtype="int32"),
            keras.ops.cumsum(value_chunk_sizes),
        ]
    )
    value_list = []
    for i in range(len(spatial_shapes_list)):
        start = cum_sizes[i]
        current_slice_size = slice_sizes[i]
        dynamic_slice_start_indices = (0, 0, start)
        dynamic_slice_shape = (
            keras.ops.shape(flattened_value)[0],
            keras.ops.shape(flattened_value)[1],
            current_slice_size,
        )
        sliced_value = keras.ops.slice(
            flattened_value, dynamic_slice_start_indices, dynamic_slice_shape
        )
        value_list.append(sliced_value)
    if method == "default":
        sampling_grids = 2 * sampling_locations - 1
    elif method == "discrete":
        sampling_grids = sampling_locations
    else:
        sampling_grids = 2 * sampling_locations - 1
    permuted_sampling_grids = keras.ops.transpose(
        sampling_grids, axes=(0, 2, 1, 3, 4)
    )
    flattened_sampling_grids = keras.ops.reshape(
        permuted_sampling_grids,
        (
            batch_size * num_heads,
            num_queries,
            num_levels_from_shape,
            num_points_from_shape,
        ),
    )
    cum_points = keras.ops.concatenate(
        [
            keras.ops.zeros((1,), dtype="int32"),
            keras.ops.cumsum(keras.ops.array(num_points_list, dtype="int32")),
        ]
    )
    sampling_grids_list = []
    for i in range(num_levels):
        start = cum_points[i]
        current_level_num_points = num_points_list[i]
        slice_start_indices = (0, 0, start, 0)
        slice_shape = (
            keras.ops.shape(flattened_sampling_grids)[0],
            keras.ops.shape(flattened_sampling_grids)[1],
            current_level_num_points,
            keras.ops.shape(flattened_sampling_grids)[3],
        )
        sliced_grid = keras.ops.slice(
            flattened_sampling_grids, slice_start_indices, slice_shape
        )
        sampling_grids_list.append(sliced_grid)
    sampling_value_list = []
    for level_id in range(num_levels):
        # batch_size, height*width, num_heads, hidden_dim
        # -> batch_size, height*width, num_heads*hidden_dim
        # -> batch_size, num_heads*hidden_dim, height*width
        # -> batch_size*num_heads, hidden_dim, height, width
        if (
            spatial_shapes_list is not None
            and len(spatial_shapes_list) == num_levels
        ):
            height, width = spatial_shapes_list[level_id]
        else:
            height = dynamic_spatial_shapes[level_id, 0]
            width = dynamic_spatial_shapes[level_id, 1]
        value_l_ = keras.ops.reshape(
            value_list[level_id],
            (batch_size * num_heads, hidden_dim, height, width),
        )
        # batch_size, num_queries, num_heads, num_points, 2
        # -> batch_size, num_heads, num_queries, num_points, 2
        # -> batch_size*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids_list[level_id]
        # batch_size*num_heads, hidden_dim, num_queries, num_points
        if method == "default":
            sampling_value_l_ = grid_sample(
                data=value_l_,
                grid=sampling_grid_l_,
                align_corners=False,
                height=height,
                width=width,
            )
        elif method == "discrete":
            scale_factors = keras.ops.cast(
                keras.ops.array([width, height]),
                dtype=sampling_grid_l_.dtype,
            )
            sampling_coord_float = sampling_grid_l_ * scale_factors
            _sampling_coord_x_int = keras.ops.cast(
                keras.ops.floor(sampling_coord_float[..., 0] + 0.5), "int32"
            )
            _sampling_coord_y_int = keras.ops.cast(
                keras.ops.floor(sampling_coord_float[..., 1] + 0.5), "int32"
            )
            clamped_coord_x = keras.ops.clip(
                _sampling_coord_x_int, 0, width - 1
            )
            clamped_coord_y = keras.ops.clip(
                _sampling_coord_y_int, 0, height - 1
            )
            sampling_coord_stacked = keras.ops.stack(
                [clamped_coord_x, clamped_coord_y], axis=-1
            )
            B_prime = batch_size * num_heads
            Q_dim = num_queries
            P_level = num_points_list[level_id]
            sampling_coord = keras.ops.reshape(
                sampling_coord_stacked, (B_prime, Q_dim * P_level, 2)
            )
            value_l_permuted = keras.ops.transpose(value_l_, (0, 2, 3, 1))
            y_coords_for_gather = sampling_coord[
                ..., 1
            ]  # (B_prime, Q_dim * P_level)
            x_coords_for_gather = sampling_coord[
                ..., 0
            ]  # (B_prime, Q_dim * P_level)
            indices = y_coords_for_gather * width + x_coords_for_gather
            indices = keras.ops.expand_dims(indices, axis=-1)
            value_l_flat = keras.ops.reshape(
                value_l_permuted, (B_prime, height * width, hidden_dim)
            )
            gathered_values = keras.ops.take_along_axis(
                value_l_flat, indices, axis=1
            )
            permuted_gathered_values = keras.ops.transpose(
                gathered_values, axes=(0, 2, 1)
            )
            sampling_value_l_ = keras.ops.reshape(
                permuted_gathered_values, (B_prime, hidden_dim, Q_dim, P_level)
            )
        else:
            sampling_value_l_ = grid_sample(
                data=value_l_,
                grid=sampling_grid_l_,
                align_corners=False,
                height=height,
                width=width,
            )
        sampling_value_list.append(sampling_value_l_)
    # (batch_size, num_queries, num_heads, num_levels, num_points)
    # -> (batch_size, num_heads, num_queries, num_levels, num_points)
    # -> (batch_size, num_heads, 1, num_queries, num_levels*num_points)
    _attention_weights = keras.ops.transpose(
        attention_weights, axes=(0, 2, 1, 3)
    )
    _attention_weights = keras.ops.reshape(
        _attention_weights,
        (batch_size * num_heads, 1, num_queries, sum(num_points_list)),
    )
    concatenated_sampling_values = keras.ops.concatenate(
        sampling_value_list, axis=-1
    )
    weighted_values = concatenated_sampling_values * _attention_weights
    summed_values = keras.ops.sum(weighted_values, axis=-1)
    output = keras.ops.reshape(
        summed_values, (batch_size, num_heads * hidden_dim, num_queries)
    )
    return keras.ops.transpose(output, axes=(0, 2, 1))


def weighting_function(max_num_bins, up, reg_scale):
    """Generates weighting values for binning operations.

    This function creates a set of weighting values used for integral-based
    bounding box regression. It is used in `DFineDecoder` to create a
    projection matrix for converting corner predictions into distances. The
    weights follow an exponential distribution around zero.

    Args:
        max_num_bins: int, Maximum number of bins to generate.
        up: Tensor, Upper bound reference value.
        reg_scale: float, Regularization scale factor.

    Returns:
        Tensor: Weighting values of shape `[max_num_bins]`.
    """
    upper_bound1 = abs(up[0]) * abs(reg_scale)
    upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
    step = (upper_bound1 + 1) ** (2 / (max_num_bins - 2))
    left_values = [
        -((step) ** i) + 1 for i in range(max_num_bins // 2 - 1, 0, -1)
    ]
    right_values = [(step) ** i - 1 for i in range(1, max_num_bins // 2)]
    values = (
        [-upper_bound2]
        + left_values
        + [keras.ops.zeros_like(keras.ops.expand_dims(up[0], axis=0))]
        + right_values
        + [upper_bound2]
    )
    values = keras.ops.concatenate(values, 0)
    return values


def corners_to_center_format(bboxes_corners):
    """Converts bounding boxes from corner format to center format.

    This function converts bounding boxes from the corner format
    `(top-left, bottom-right)` to the center format `(center_x, center_y,
    width, height)`. It is used in `DFineContrastiveDenoisingGroupGenerator`
    for box noise augmentation and in `distance2bbox` to return the final
    bounding box format.

    Args:
        bboxes_corners: Tensor, Bounding boxes in corner format of shape
            `[..., 4]` where the last dimension contains
            `[top_left_x, top_left_y, bottom_right_x, bottom_right_y]`.

    Returns:
        Tensor: Bounding boxes in center format of shape `[..., 4]` where
            the last dimension contains `[center_x, center_y, width, height]`.
    """
    top_left_x = bboxes_corners[..., 0]
    top_left_y = bboxes_corners[..., 1]
    bottom_right_x = bboxes_corners[..., 2]
    bottom_right_y = bboxes_corners[..., 3]
    center_x = (top_left_x + bottom_right_x) / 2
    center_y = (top_left_y + bottom_right_y) / 2
    width = bottom_right_x - top_left_x
    height = bottom_right_y - top_left_y
    return keras.ops.stack([center_x, center_y, width, height], axis=-1)


def center_to_corners_format(bboxes_center):
    """Converts bounding boxes from center format to corner format.

    This function converts bounding boxes from the center format
    `(center_x, center_y, width, height)` to the corner format
    `(top-left, bottom-right)`. It is used extensively in
    `DFineObjectDetector` for loss calculations (e.g., `hungarian_matcher`,
    `compute_box_losses`) that require corner representations for IoU
    computation.

    Args:
        bboxes_center: Tensor, Bounding boxes in center format of shape
            `[..., 4]` where the last dimension contains
            `[center_x, center_y, width, height]`.

    Returns:
        Tensor: Bounding boxes in corner format of shape `[..., 4]` where
            the last dimension contains `[top_left_x, top_left_y,
            bottom_right_x, bottom_right_y]`.
    """
    center_x = bboxes_center[..., 0]
    center_y = bboxes_center[..., 1]
    width = bboxes_center[..., 2]
    height = bboxes_center[..., 3]

    top_left_x = center_x - 0.5 * width
    top_left_y = center_y - 0.5 * height
    bottom_right_x = center_x + 0.5 * width
    bottom_right_y = center_y + 0.5 * height

    return keras.ops.stack(
        [top_left_x, top_left_y, bottom_right_x, bottom_right_y], axis=-1
    )


def distance2bbox(points, distance, reg_scale):
    """Converts distance predictions to bounding boxes.

    This function converts distance predictions from anchor points to
    bounding boxes. It is a key part of the regression head in `DFineDecoder`,
    transforming the output of the integral-based prediction into final
    bounding box coordinates.

    Args:
        points: Tensor, Anchor points of shape `[..., 4]` where the last
            dimension contains `[x, y, width, height]`.
        distance: Tensor, Distance predictions of shape `[..., 4]` where
            the last dimension contains `[left, top, right, bottom]` distances.
        reg_scale: float, Regularization scale factor.

    Returns:
        Tensor: Bounding boxes in center format of shape `[..., 4]` where
            the last dimension contains `[center_x, center_y, width, height]`.
    """
    reg_scale = abs(reg_scale)
    top_left_x = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (
        points[..., 2] / reg_scale
    )
    top_left_y = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (
        points[..., 3] / reg_scale
    )
    bottom_right_x = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (
        points[..., 2] / reg_scale
    )
    bottom_right_y = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (
        points[..., 3] / reg_scale
    )
    bboxes = keras.ops.stack(
        [top_left_x, top_left_y, bottom_right_x, bottom_right_y], axis=-1
    )
    return corners_to_center_format(bboxes)
