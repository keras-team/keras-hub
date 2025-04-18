import keras


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __neg__(self):
        return Point(-self.x, -self.y)

    def cross(self, other):
        return self.x * other.y - self.y * other.x

    def to_tuple(self):
        return (self.x, self.y)


def shrink_polygan(polygon, offset):
    """
    Shrinks a polygon inward by moving each point toward the center.
    """
    if len(polygon) < 3:
        return polygon

    if not isinstance(polygon[0], Point):
        polygon = [Point(p[0], p[1]) for p in polygon]

    cx = sum(p.x for p in polygon) / len(polygon)
    cy = sum(p.y for p in polygon) / len(polygon)

    shrunk = []
    for p in polygon:
        dx = p.x - cx
        dy = p.y - cy
        norm = max((dx**2 + dy**2) ** 0.5, 1e-6)
        shrink_ratio = max(0, 1 - offset / norm)
        shrunk.append(Point(cx + dx * shrink_ratio, cy + dy * shrink_ratio))

    return shrunk


# Polygon Area
def Polygon(coords):
    """
    Calculate the area of a polygon using the Shoelace formula.
    """
    coords = keras.ops.convert_to_tensor(coords, dtype="float32")
    x = coords[:, 0]
    y = coords[:, 1]

    x_next = keras.ops.roll(x, shift=-1, axis=0)
    y_next = keras.ops.roll(y, shift=-1, axis=0)

    area = 0.5 * keras.ops.abs(keras.ops.sum(x * y_next - x_next * y))
    return area


# binary search smallest width
def binary_search_smallest_width(poly):
    """
    The function aims maximum amount by which polygan can be shrunk by
    taking polygan's smallest width
    """
    if len(poly) < 3:
        return 0

    low, high = 0, 1

    while high - low > 0.01:
        mid = (high + low) / 2
        mid_poly = shrink_polygan(poly, mid)
        mid_poly = keras.ops.cast(
            keras.ops.stack([[p.x, p.y] for p in mid_poly]), dtype="float32"
        )
        area = Polygon(mid_poly)

        if area > 0.1:
            low = mid
        else:
            high = mid

    height = (low + high) / 2
    height = (low + high) / 2
    return int(height) if height >= 0.1 else 0


# project point to line
def project_point_to_line(x, u, v, axis=0):
    """
    Projects a point x onto the line defined by points u and v
    """
    x = keras.ops.convert_to_tensor(x, dtype="float32")
    u = keras.ops.convert_to_tensor(u, dtype="float32")
    v = keras.ops.convert_to_tensor(v, dtype="float32")

    n = v - u
    n = n / (
        keras.ops.norm(n, axis=axis, keepdims=True) + keras.backend.epsilon()
    )
    p = u + n * keras.ops.sum((x - u) * n, axis=axis, keepdims=True)
    return p


# project_point_to_segment
def project_point_to_segment(x, u, v, axis=0):
    """
    Projects a point x onto the line segment defined by points u and v
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


# get line of height
def get_line_height(poly):
    return binary_search_smallest_width(poly)


# cv2.fillpoly function with keras.ops
def fill_poly_keras(vertices, image_shape):
    """
    Fill a polygon using the cv2.fillPoly function with keras.ops.
    Ray-casting algorithm to determine if a point is inside a polygon.
    """
    height, width = image_shape
    x = keras.ops.arange(width)
    y = keras.ops.arange(height)
    xx, yy = keras.ops.meshgrid(x, y)
    xx = keras.ops.cast(xx, "float32")
    yy = keras.ops.cast(yy, "float32")

    result = keras.ops.zeros((height, width), dtype="float32")

    vertices = keras.ops.convert_to_tensor(vertices, dtype="float32")
    num_vertices = vertices.shape[0]

    for i in range(num_vertices):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % num_vertices]

        # Modified conditions to potentially include more boundary pixels
        cond1 = (yy > keras.ops.minimum(y1, y2)) & (
            yy <= keras.ops.maximum(y1, y2)
        )
        cond2 = xx < (x1 + (yy - y1) * (x2 - x1) / (y2 - y1))

        result = keras.ops.where(
            cond1 & cond2 & ((y1 > yy) != (y2 > yy)), 1 - result, result
        )

    result = keras.ops.cast(result, "int32")
    return result


# get mask
def get_mask(w, h, polys, ignores):
    """
    Generates a binary mask where:
    - Ignored regions are set to 0
    - Text regions are set to 1
    """
    mask = keras.ops.ones((h, w), dtype="float32")

    for poly, ignore in zip(polys, ignores):
        poly = keras.ops.cast(keras.ops.convert_to_numpy(poly), dtype="int32")

        if poly.shape[0] < 3:
            print("Skipping invalid polygon:", poly)
            continue

        fill_value = 0.0 if ignore else 1.0
        poly_mask = fill_poly_keras(poly, (h, w))

        if ignore:
            mask = keras.ops.where(
                keras.ops.cast(poly_mask, "float32") == 1.0,
                keras.ops.zeros_like(mask),
                mask,
            )
        else:
            mask = keras.ops.maximum(mask, poly_mask)
    return mask


# get polygan coordinates projection
def get_coords_poly_projection(coords, poly):
    """
    This projects set of points onto edges of a polygan and return closest
    projected points
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


# get polygan coordinates distance
def get_coords_poly_distance(coords, poly):
    """
    This function calculates distance between set of points and polygan
    """
    projection = get_coords_poly_projection(coords, poly)
    return keras.ops.linalg.norm(projection - coords, axis=1)


# get normalized weight
def get_normalized_weight(heatmap, mask, background_weight=3.0):
    """
    This function calculates normalized weight of heatmap
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
    #   weight[keras.ops.cast(neg, dtype="bool")] = background_weight
    neg = keras.ops.cast(neg, "bool")
    weight = keras.ops.where(neg, background_weight, weight)
    pos = keras.ops.cast(pos, "bool")
    weight = keras.ops.where(pos, wpos, weight)
    return weight


# Getting region coordinates
def get_region_coordinate(w, h, poly, heights, shrink):
    """
    Extract coordinates of regions corresponding to text lines in an image.
    """
    label_map = keras.ops.zeros((h, w), dtype="float32")
    for line_id, (p, height) in enumerate(zip(poly, heights)):
        if height > 0:
            poly_points = [Point(row[0], row[1]) for row in p]
            shrinked_poly = shrink_polygan(poly_points, height * shrink)
            shrunk_poly_tuples = [point.to_tuple() for point in shrinked_poly]
            shrunk_poly_tensor = keras.ops.convert_to_tensor(
                shrunk_poly_tuples, dtype="float32"
            )
            filled_polygon = fill_poly_keras(shrunk_poly_tensor, (h, w))
            label_map = keras.ops.maximum(label_map, filled_polygon)

    label_map = keras.ops.convert_to_tensor(label_map)
    sorted_tensor = keras.ops.sort(keras.ops.reshape(label_map, (-1,)))
    diff = keras.ops.concatenate(
        [
            keras.ops.convert_to_tensor([True]),
            (sorted_tensor[1:] != sorted_tensor[:-1]),
        ]
    )
    diff = keras.ops.reshape(diff, (-1,))
    indices = keras.ops.convert_to_tensor(keras.ops.where(diff))
    indices = keras.ops.reshape(indices, (-1,))
    unique_labels = keras.ops.take(sorted_tensor, indices)
    unique_labels = unique_labels[unique_labels != 0]
    regions_coords = []
    for label in unique_labels:
        mask = keras.ops.equal(label_map, label)
        y, x = keras.ops.nonzero(mask)
        coords = keras.ops.stack([x, y], axis=-1)
        regions_coords.append(keras.ops.convert_to_numpy(coords))

    return regions_coords
