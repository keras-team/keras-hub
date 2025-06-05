import keras


def Polygon(coords):
    """Calculate the area of a polygon using the Shoelace formula.
    Args:
        coords: A tensor of shape (N, 2) representing the coordinates of
        the polygon vertices.
    Returns:
        The area of the polygon.
    """
    coords = keras.ops.convert_to_tensor(coords, dtype="float32")
    x = coords[:, 0]
    y = coords[:, 1]

    x_next = keras.ops.roll(x, shift=-1, axis=0)
    y_next = keras.ops.roll(y, shift=-1, axis=0)

    area = 0.5 * keras.ops.abs(keras.ops.sum(x * y_next - x_next * y))
    return area


def fill_poly_keras(vertices, image_shape):
    """Fill a polygon using the fillPoly function with keras.ops.
    Ray-casting algorithm to determine if a point is inside a polygon.
    Args:
        vertices: A tensor of shape (N, 2) representing the polygon vertices.
        image_shape: A tuple (height, width) representing the image dimensions.
    Returns:
        A binary mask of the same shape as the image, where the polygon area i
        s filled with 1s.
    """
    height, width = image_shape
    ys = keras.ops.arange(0, height, dtype="float32")
    xs = keras.ops.arange(0, width, dtype="float32")
    xx, yy = keras.ops.meshgrid(xs, ys)

    vertices = keras.ops.convert_to_tensor(vertices, dtype="float32")
    x = vertices[:, 0]
    y = vertices[:, 1]

    x2 = keras.ops.concatenate([x[1:], x[:1]], axis=0)
    y2 = keras.ops.concatenate([y[1:], y[:1]], axis=0)

    x1 = keras.ops.expand_dims(x, axis=0)
    y1 = keras.ops.expand_dims(y, axis=0)
    x2 = keras.ops.expand_dims(x2, axis=0)
    y2 = keras.ops.expand_dims(y2, axis=0)

    px = keras.ops.expand_dims(xx, axis=-1)
    py = keras.ops.expand_dims(yy, axis=-1)

    cond1 = ((y1 <= py) & (y2 > py)) | ((y1 > py) & (y2 <= py))
    slope = (x2 - x1) / (y2 - y1 + 1e-6)
    intersect_x = x1 + slope * (py - y1)
    cond2 = px < intersect_x

    mask = (
        keras.ops.sum(keras.ops.cast(cond1 & cond2, dtype="int32"), axis=-1) % 2
    )
    return keras.ops.cast(mask, "float32")


def get_mask(w, h, polys, ignores):
    """Generates a binary mask using fill_poly function:
    - Ignored regions are set to 0
    - Text regions are set to 1
    Args:
        w: Width of the image.
        h: Height of the image.
        polys: List of polygons, where each polygon is a list of vertices.
        ignores: List of booleans indicating whether to ignore the polygon.
    Returns:
        A binary mask of shape (height, width) with 1s for text regions and
        0s for ignored regions.
    """
    mask = keras.ops.ones((h, w), dtype="float32")
    for poly, ignore in zip(polys, ignores):
        poly = keras.ops.convert_to_tensor(poly, dtype="float32")
        if keras.ops.shape(poly)[0] < 3:
            continue
        poly_mask = fill_poly_keras(poly, (h, w))
        if ignore:
            mask = keras.ops.where(
                poly_mask == 1.0, keras.ops.zeros_like(mask), mask
            )
        else:
            mask = keras.ops.maximum(mask, poly_mask)
    return mask


def step_function(x, y, k=50.0):
    """
    Step function that returns 1 if x > y, else 0.
    Args:
        x: Input tensor.
        y: Threshold value.
        k: Steepness of the step function.
    Returns:
            A tensor with values between 0 and 1, representing the step function.
    """
    return 1.0 / (1.0 + keras.ops.exp(-k * (x - y)))


def project_point_to_line(x, u, v, axis=0):
    """Projects a point x onto the line defined by points u and v"""
    x = keras.ops.convert_to_tensor(x, dtype="float32")
    u = keras.ops.convert_to_tensor(u, dtype="float32")
    v = keras.ops.convert_to_tensor(v, dtype="float32")

    n = v - u
    n = n / (
        keras.ops.norm(n, axis=axis, keepdims=True) + keras.backend.epsilon()
    )
    p = u + n * keras.ops.sum((x - u) * n, axis=axis, keepdims=True)
    return p


def project_point_to_segment(x, u, v, axis=0):
    """Projects a point x onto the line segment defined by points u and v
    Args:
        x: The point to project.
        u: One endpoint of the line segment.
        v: The other endpoint of the line segment.
        axis: The axis along which to project.
    Returns:
        The projected point on the line segment.
    """
    p = project_point_to_line(x, u, v, axis=axis)
    outer = keras.ops.greater_equal(
        keras.ops.sum((u - p) * (v - p), axis=axis, keepdims=True), 0
    )
    near_u = keras.ops.less_equal(
        keras.ops.norm(u - p, axis=axis, keepdims=True),
        keras.ops.norm(v - p, axis=axis, keepdims=True),
    )
    o = keras.ops.where(outer, keras.ops.where(near_u, u, v), p)
    return o


def get_coords_poly_projection(coords, poly):
    """This projects set of points onto edges of a polygan and return closest
    projected points
    Args:
        coords: A tensor of shape (N, 2) representing the coordinates of the
        points to project.
        poly: A tensor of shape (M, 2) representing the polygon vertices.
    Returns:
        A tensor of shape (N, 2) representing the closest projected points on
        the polygon edges.
    """
    start_points = keras.ops.array(poly, dtype="float32")
    end_points = keras.ops.concatenate(
        [
            keras.ops.array(poly[1:], dtype="float32"),
            keras.ops.array(poly[:1], dtype="float32"),
        ],
        axis=0,
    )
    region_points = keras.ops.array(coords, dtype="float32")

    projected_points = project_point_to_segment(
        keras.ops.expand_dims(region_points, axis=1),
        keras.ops.expand_dims(start_points, axis=0),
        keras.ops.expand_dims(end_points, axis=0),
        axis=2,
    )

    projection_distances = keras.ops.norm(
        keras.ops.expand_dims(region_points, axis=1) - projected_points, axis=2
    )

    indices = keras.ops.expand_dims(
        keras.ops.argmin(projection_distances, axis=1), axis=-1
    )
    best_projected_points = keras.ops.take_along_axis(
        projected_points, indices[..., None], axis=1
    )[:, 0, :]

    return best_projected_points


def get_coords_poly_distance(coords, poly):
    """This function calculates distance between set of points and polygan
    Args:
        coords: A tensor of shape (N, 2) representing the coordinates of the
        points.
        poly: A tensor of shape (M, 2) representing the polygon vertices.
    Returns:
        A tensor of shape (N,) representing the distances between the points
        and the polygon edges.
    """
    projection = get_coords_poly_projection(coords, poly)
    return keras.ops.linalg.norm(projection - coords, axis=1)


def get_normalized_weight(heatmap, mask, background_weight=3.0):
    """This function calculates normalized weight of heatmap
    Args:
        heatmap: A tensor of shape (N, H, W) representing the heatmap.
        mask: A tensor of shape (N, H, W) representing the mask.
        background_weight: A float representing the background weight.
    Returns:
        A tensor of shape (N, H, W) representing the normalized weight.
    """
    pos = keras.ops.greater_equal(heatmap, 0.5)
    neg = keras.ops.ones_like(pos, dtype="float32") - keras.ops.cast(
        pos, dtype="float32"
    )
    pos = keras.ops.logical_and(pos, mask)
    neg = keras.ops.logical_and(neg, mask)
    npos = keras.ops.sum(pos)
    nneg = keras.ops.sum(neg)
    smooth = (
        keras.ops.cast(npos, dtype="float32")
        + keras.ops.cast(nneg, dtype="float32")
        + 1
    ) * 0.05
    wpos = (keras.ops.cast(nneg, dtype="float32") + smooth) / (
        keras.ops.cast(npos, dtype="float32") + smooth
    )
    weight = keras.ops.zeros_like(heatmap)
    neg = keras.ops.cast(neg, "bool")
    weight = keras.ops.where(neg, background_weight, weight)
    pos = keras.ops.cast(pos, "bool")
    weight = keras.ops.where(pos, wpos, weight)
    return weight
