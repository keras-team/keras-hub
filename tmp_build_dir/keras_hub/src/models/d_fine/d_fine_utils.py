import keras


def d_fine_kernel_initializer(initializer_range=0.01, name="random_normal"):
    if name == "random_normal":
        return keras.initializers.RandomNormal(
            mean=0.0, stddev=initializer_range
        )
    elif name == "glorot_uniform":
        return keras.initializers.GlorotUniform()
    elif name == "zeros":
        return keras.initializers.Zeros()


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
    x1 = keras.ops.maximum(x, eps)
    x2 = keras.ops.maximum(1 - x, eps)
    return keras.ops.log(x1 / x2)


def grid_sample(data, grid, align_corners=False, height=None, width=None):
    """Samples data at specified grid locations using bilinear interpolation.

    This function performs bilinear interpolation to sample data at arbitrary
    grid locations. It is a core component of the deformable attention
    mechanism, used within `multi_scale_deformable_attention_v2`.
    This is a Keras-native implementation (polyfill) for
    `torch.nn.functional.grid_sample`.

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

        width_for_indexing = (
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
        indices = y_coords_flat * width_for_indexing + x_coords_flat

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
    num_points,
    slice_sizes,
    spatial_shapes,
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
        num_points: list, Number of sampling points for each level.
        slice_sizes: list, Sizes for slicing the value tensor.
        spatial_shapes: list, Spatial shapes for each level.
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
    values = []
    for i in range(len(spatial_shapes)):
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
        values.append(sliced_value)
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
            keras.ops.cumsum(keras.ops.array(num_points, dtype="int32")),
        ]
    )
    sampling_grids = []
    for i in range(num_levels):
        start = cum_points[i]
        current_level_num_points = num_points[i]
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
        sampling_grids.append(sliced_grid)
    sampling_values = []
    for level_id in range(num_levels):
        if spatial_shapes is not None and len(spatial_shapes) == num_levels:
            height, width = spatial_shapes[level_id]
        else:
            height = dynamic_spatial_shapes[level_id, 0]
            width = dynamic_spatial_shapes[level_id, 1]
        value_l_ = keras.ops.reshape(
            values[level_id],
            (batch_size * num_heads, hidden_dim, height, width),
        )
        sampling_grid_l_ = sampling_grids[level_id]
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
            sampling_coord_x_int = keras.ops.cast(
                keras.ops.floor(sampling_coord_float[..., 0] + 0.5), "int32"
            )
            sampling_coord_y_int = keras.ops.cast(
                keras.ops.floor(sampling_coord_float[..., 1] + 0.5), "int32"
            )
            clamped_coord_x = keras.ops.clip(sampling_coord_x_int, 0, width - 1)
            clamped_coord_y = keras.ops.clip(
                sampling_coord_y_int, 0, height - 1
            )
            sampling_coord_stacked = keras.ops.stack(
                [clamped_coord_x, clamped_coord_y], axis=-1
            )
            B_prime = batch_size * num_heads
            Q_dim = num_queries
            P_level = num_points[level_id]
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
        sampling_values.append(sampling_value_l_)
    attention_weights = keras.ops.transpose(
        attention_weights, axes=(0, 2, 1, 3)
    )
    attention_weights = keras.ops.reshape(
        attention_weights,
        (batch_size * num_heads, 1, num_queries, sum(num_points)),
    )
    concatenated_sampling_values = keras.ops.concatenate(
        sampling_values, axis=-1
    )
    weighted_values = concatenated_sampling_values * attention_weights
    summed_values = keras.ops.sum(weighted_values, axis=-1)
    output = keras.ops.reshape(
        summed_values, (batch_size, num_heads * hidden_dim, num_queries)
    )
    return keras.ops.transpose(output, axes=(0, 2, 1))


def weighting_function(max_num_bins, upsampling_factor, reg_scale):
    """Generates weighting values for binning operations.

    This function creates a set of weighting values used for integral-based
    bounding box regression. It is used in `DFineDecoder` to create a
    projection matrix for converting corner predictions into distances. The
    weights follow an exponential distribution around zero.

    Args:
        max_num_bins: int, Maximum number of bins to generate.
        upsampling_factor: Tensor, A scaling hyperparameter that controls the
            range of the bins used for integral-based bounding box regression.
        reg_scale: float, Regularization scale factor.

    Returns:
        Tensor: Weighting values of shape `[max_num_bins]`.
    """
    upper_bound1 = abs(upsampling_factor[0]) * abs(reg_scale)
    upper_bound2 = abs(upsampling_factor[0]) * abs(reg_scale) * 2
    step = (upper_bound1 + 1) ** (2 / (max_num_bins - 2))
    left_values = [
        -((step) ** i) + 1 for i in range(max_num_bins // 2 - 1, 0, -1)
    ]
    right_values = [(step) ** i - 1 for i in range(1, max_num_bins // 2)]
    values = (
        [-upper_bound2]
        + left_values
        + [
            keras.ops.zeros_like(
                keras.ops.expand_dims(upsampling_factor[0], axis=0)
            )
        ]
        + right_values
        + [upper_bound2]
    )
    values = keras.ops.concatenate(values, 0)
    return values


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
    return keras.utils.bounding_boxes.convert_format(
        bboxes,
        source="xyxy",
        target="center_xywh",
        dtype=points.dtype,
    )


def hungarian_assignment(cost_matrix, num_queries):
    """Solves the linear assignment problem using the Hungarian algorithm.

    This function provides a JIT-compatible implementation of the Hungarian
    (Munkres) algorithm using pure `keras.ops` operations. It is designed to
    replace Scipy's `optimize.linear_sum_assignment` for backend-agnostic
    end-to-end model compilation. The implementation uses a stateful loop
    with `keras.ops.while_loop`, a state machine pattern with
    `keras.ops.switch`, and tensor-only operations to ensure compatibility
    with static graphs and standard accelerators.

    Args:
        cost_matrix: Tensor, A 2D tensor of shape `(num_rows, num_cols)`
            representing the cost of each potential assignment. `num_rows`
            typically corresponds to the number of predictions (queries),
            and `num_cols` corresponds to number of ground-truth targets.
        num_queries: int, The fixed number of queries (predictions) from
            the model, used to establish static shapes for JAX compatibility.

    Returns:
        Tuple: A tuple `(row_ind, col_ind, valid_mask)` containing:
            - row_ind: Tensor with integer indices for the rows (predictions).
            - col_ind: Tensor with integer indices for the assigned columns
                (targets).
            - valid_mask: Boolean tensor where `True` indicates a valid
                assignment that falls within the original (unpadded) cost
                matrix dimensions.
    """
    # Reference: https://github.com/bmc/munkres/blob/master/munkres.py

    original_num_rows, original_num_cols = keras.ops.shape(cost_matrix)
    # Pad matrix to be square.
    padded_cost_matrix = keras.ops.full(
        (num_queries, num_queries), 1e9, dtype=cost_matrix.dtype
    )
    padded_cost_matrix = keras.ops.slice_update(
        padded_cost_matrix,
        (0, 0),
        cost_matrix,
    )
    # Step 1: Subtract row minima.
    cost = padded_cost_matrix - keras.ops.min(
        padded_cost_matrix, axis=1, keepdims=True
    )
    # Step 2: Subtract column minima.
    cost = cost - keras.ops.min(cost, axis=0, keepdims=True)

    def body(
        step,
        cost,
        starred_mask,
        row_covered,
        col_covered,
        primed_mask,
        path_start_row,
        path_start_col,
    ):
        zero_mask = keras.ops.abs(cost) < 1e-6

        def step_2():
            # Initial starring: Star zeros with no starred zero in their row or
            # column.
            s_mask = keras.ops.zeros_like(starred_mask, dtype="bool")

            def star_zeros(i, s_m):
                def star_zeros_in_row(j, s_m_inner):
                    is_zero = zero_mask[i, j]
                    # Check if no starred zero in this row.
                    no_star_in_row = keras.ops.logical_not(
                        keras.ops.any(s_m_inner[i])
                    )
                    # Check if no starred zero in this column.
                    no_star_in_col = keras.ops.logical_not(
                        keras.ops.any(s_m_inner[:, j])
                    )

                    def can_star():
                        return keras.ops.scatter_update(
                            s_m_inner,
                            [[i, j]],
                            [True],
                        )

                    def cannot_star():
                        return s_m_inner

                    should_star = keras.ops.logical_and(
                        keras.ops.logical_and(is_zero, no_star_in_row),
                        no_star_in_col,
                    )
                    return keras.ops.cond(should_star, can_star, cannot_star)

                return keras.ops.fori_loop(
                    0, num_queries, star_zeros_in_row, s_m
                )

            s_mask = keras.ops.fori_loop(0, num_queries, star_zeros, s_mask)
            return (
                3,
                cost,
                s_mask,
                keras.ops.zeros_like(row_covered),
                keras.ops.zeros_like(col_covered),
                keras.ops.zeros_like(primed_mask),
                -1,
                -1,
            )

        def step_3():
            # Step 3: Cover each column containing a starred zero.
            new_col_covered = keras.ops.any(starred_mask, axis=0)
            num_covered = keras.ops.sum(
                keras.ops.cast(new_col_covered, "int32")
            )
            return keras.ops.cond(
                num_covered >= num_queries,
                lambda: (
                    0,
                    cost,
                    starred_mask,
                    row_covered,
                    new_col_covered,
                    primed_mask,
                    -1,
                    -1,
                ),  # Done
                lambda: (
                    4,
                    cost,
                    starred_mask,
                    row_covered,
                    new_col_covered,
                    primed_mask,
                    -1,
                    -1,
                ),  # Continue to step 4
            )

        def step_4():
            # Step 4: Find a noncovered zero and prime it.
            uncovered_zeros = keras.ops.logical_and(
                keras.ops.logical_and(
                    zero_mask,
                    keras.ops.logical_not(
                        keras.ops.expand_dims(row_covered, 1)
                    ),
                ),
                keras.ops.logical_not(keras.ops.expand_dims(col_covered, 0)),
            )

            def has_uncovered_zero():
                uncovered_zeros_flat = keras.ops.reshape(uncovered_zeros, [-1])
                first_idx = keras.ops.argmax(
                    keras.ops.cast(uncovered_zeros_flat, "int32")
                )
                r = first_idx // num_queries
                c = first_idx % num_queries
                p_mask = keras.ops.scatter_update(primed_mask, [[r, c]], [True])
                starred_in_row = starred_mask[r]

                def has_starred_in_row():
                    star_col = keras.ops.argmax(
                        keras.ops.cast(starred_in_row, "int32")
                    )
                    r_cov = keras.ops.scatter_update(row_covered, [[r]], [True])
                    c_cov = keras.ops.scatter_update(
                        col_covered, [[star_col]], [False]
                    )
                    return 4, cost, starred_mask, r_cov, c_cov, p_mask, -1, -1

                def no_starred_in_row():
                    return (
                        5,
                        cost,
                        starred_mask,
                        row_covered,
                        col_covered,
                        p_mask,
                        r,
                        c,
                    )

                return keras.ops.cond(
                    keras.ops.any(starred_in_row),
                    has_starred_in_row,
                    no_starred_in_row,
                )

            def no_uncovered_zero():
                return (
                    6,
                    cost,
                    starred_mask,
                    row_covered,
                    col_covered,
                    primed_mask,
                    -1,
                    -1,
                )

            return keras.ops.cond(
                keras.ops.any(uncovered_zeros),
                has_uncovered_zero,
                no_uncovered_zero,
            )

        def step_5():
            # Step 5: Construct a series of alternating starred and primed
            # zeros.
            path = keras.ops.full((num_queries * 2, 2), -1, dtype="int32")
            path = keras.ops.scatter_update(
                path, [[0]], [[path_start_row, path_start_col]]
            )

            def build_path(count, path_state):
                def continue_building(cnt, p):
                    current_col = p[cnt - 1, 1]
                    starred_in_col = starred_mask[:, current_col]

                    def found_star():
                        star_row = keras.ops.argmax(
                            keras.ops.cast(starred_in_col, "int32")
                        )
                        p1 = keras.ops.scatter_update(
                            p, [[cnt]], [[star_row, current_col]]
                        )
                        primed_in_star_row = primed_mask[star_row]
                        prime_col = keras.ops.argmax(
                            keras.ops.cast(primed_in_star_row, "int32")
                        )
                        p2 = keras.ops.scatter_update(
                            p1, [[cnt + 1]], [[star_row, prime_col]]
                        )
                        return cnt + 2, p2

                    def no_star():
                        # Path complete.
                        return cnt, p

                    return keras.ops.cond(
                        keras.ops.any(starred_in_col), found_star, no_star
                    )

                def should_continue(cnt, p):
                    return keras.ops.logical_and(
                        cnt < num_queries * 2, p[cnt - 1, 1] >= 0
                    )

                return keras.ops.while_loop(
                    should_continue,
                    continue_building,
                    (count, path_state),
                    maximum_iterations=num_queries,
                )

            path_count, final_path = build_path(1, path)
            s_mask = starred_mask

            def update_star_mask(i, mask):
                def apply_update():
                    row_idx = final_path[i, 0]
                    col_idx = final_path[i, 1]
                    valid_row = keras.ops.logical_and(
                        row_idx >= 0, row_idx < num_queries
                    )
                    valid_col = keras.ops.logical_and(
                        col_idx >= 0, col_idx < num_queries
                    )
                    valid_indices = keras.ops.logical_and(valid_row, valid_col)

                    def do_update():
                        current_value = mask[row_idx, col_idx]
                        new_value = keras.ops.logical_not(current_value)
                        return keras.ops.scatter_update(
                            mask, [[row_idx, col_idx]], [new_value]
                        )

                    def skip_update():
                        return mask

                    return keras.ops.cond(valid_indices, do_update, skip_update)

                def skip_iteration():
                    return mask

                should_process = i < path_count
                return keras.ops.cond(
                    should_process, apply_update, skip_iteration
                )

            s_mask = keras.ops.fori_loop(
                0, num_queries * 2, update_star_mask, s_mask
            )
            return (
                3,
                cost,
                s_mask,
                keras.ops.zeros_like(row_covered),
                keras.ops.zeros_like(col_covered),
                keras.ops.zeros_like(primed_mask),
                -1,
                -1,
            )

        def step_6():
            # Step 6: Add/subtract minimum uncovered value.
            uncovered_mask = keras.ops.logical_and(
                keras.ops.logical_not(keras.ops.expand_dims(row_covered, 1)),
                keras.ops.logical_not(keras.ops.expand_dims(col_covered, 0)),
            )
            min_val = keras.ops.min(keras.ops.where(uncovered_mask, cost, 1e9))
            # Add to covered rows.
            row_adjustment = keras.ops.where(
                keras.ops.expand_dims(row_covered, 1), min_val, 0.0
            )
            # Subtract from uncovered columns.
            col_adjustment = keras.ops.where(
                keras.ops.expand_dims(col_covered, 0), 0.0, -min_val
            )
            new_cost = cost + row_adjustment + col_adjustment
            return (
                4,
                new_cost,
                starred_mask,
                row_covered,
                col_covered,
                primed_mask,
                -1,
                -1,
            )

        return keras.ops.switch(
            step - 2, [step_2, step_3, step_4, step_5, step_6]
        )

    # Main algorithm loop.
    init_state = (
        2,  # Start at step 2
        cost,
        keras.ops.zeros(
            (num_queries, num_queries), dtype="bool"
        ),  # starred_mask
        keras.ops.zeros((num_queries,), dtype="bool"),  # row_covered
        keras.ops.zeros((num_queries,), dtype="bool"),  # col_covered
        keras.ops.zeros(
            (num_queries, num_queries), dtype="bool"
        ),  # primed_mask
        -1,  # path_start_row
        -1,  # path_start_col
    )
    final_state = keras.ops.while_loop(
        lambda step, *_: step > 0,
        body,
        init_state,
        maximum_iterations=num_queries * num_queries,
    )
    final_starred_mask = final_state[2]
    row_ind = keras.ops.arange(num_queries, dtype="int32")
    col_ind = keras.ops.argmax(
        keras.ops.cast(final_starred_mask, "int32"), axis=1
    )
    valid_mask = keras.ops.logical_and(
        row_ind < original_num_rows, col_ind < original_num_cols
    )
    return row_ind, col_ind, valid_mask
