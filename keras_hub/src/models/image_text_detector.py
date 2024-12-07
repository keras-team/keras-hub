import keras
import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.task import Task


@keras_hub_export("keras_hub.models.ImageTextDetector")
class ImageTextDetector(Task):
    """Base class for all text detection tasks.

    `ImageTextDetector` tasks wrap a `keras_hub.models.Task` and
    a `keras_hub.models.Preprocessor` to create a model that can be used for
    image segmentation.

    All `ImageTextDetector` tasks include a `from_preset()` constructor which
    can be used to load a pre-trained config and weights.

    Args:
        detection_thresh: The value for thresholding predicted mask outputs.
            Defaults to 0.3.
        min_area: Minimum area for a polygon to be considered valid. Defaults
            to 10.0.
        unclip_ratio: Expansion ratio of for the detected polygons.
            Defaults to 3.0.
    """

    def __init__(
        self, detection_thresh=0.3, min_area=10.0, unclip_ratio=2.0, **kwargs
    ):
        self.detection_thresh = detection_thresh
        self.min_area = min_area
        self.unclip_ratio = unclip_ratio
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "detection_thresh": self.detection_thresh,
                "min_area": self.min_area,
                "unclip_ratio": self.unclip_ratio,
            }
        )
        return config

    def postprocess_to_polygons(self, masks, contour_finder="simple"):
        """Converts the mask output of a text detector to polygon coordinates.

        Args:
            masks: Segmentation masks (3D batch of masks).
            contour_finder: Determines the method for contour finding. Possible
                values are "simple", which detects connected regions by walking
                the image, and "opencv", which uses OpenCV's contour finder if
                available. Defaults to "simple".

        Returns:
            List-of-list-of-lists. A list of polygons for each batch element,
            where each polygon is represented as a list of (x, y) points.
        """

        if not isinstance(masks, np.ndarray):
            masks = keras.ops.convert_to_numpy(masks)
        masks = masks > self.detection_thresh
        polygons = []
        for mask in masks:
            mask_polygons = mask_to_polygons(
                mask, min_area=self.min_area, contour_finder=contour_finder
            )
            mask_polygons = [
                unclip_polygon(polygon, self.unclip_ratio)
                for polygon in mask_polygons
            ]
            polygons.append(mask_polygons)
        return polygons


def compute_polygon_area(polygon):
    """Calculates the area of a polygon."""
    x, y = zip(*polygon)
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def simplify_polygon(polygon, approximation_tol=2.0):
    """Simplifies a polygon using Ramer-Douglas-Peucker."""

    if len(polygon) < 3:
        return polygon

    def perpendicular_distance(point):
        if np.all(line_start == line_end):
            return np.linalg.norm(point - line_start)
        line_vec = line_end - line_start
        point_vec = point - line_start
        return np.linalg.norm(
            np.cross(line_vec, point_vec) / np.linalg.norm(line_vec)
        )

    # find the point with the maximum distance from the line segment
    line_start, line_end = polygon[0], polygon[-1]
    distances = np.array([perpendicular_distance(pt) for pt in polygon[1:-1]])
    max_idx = np.argmax(distances)
    max_dist = distances[max_idx]
    if max_dist > approximation_tol:
        # simplify recursively
        left = simplify_polygon(polygon[: max_idx + 2], approximation_tol)
        right = simplify_polygon(polygon[max_idx + 1 :], approximation_tol)
        return np.vstack((left[:-1], right))
    else:
        return np.array([line_start, line_end])


def compute_edge_normals(polygon):
    """Computes outward normals for each edge of a polygon."""
    normals = []
    n = len(polygon)
    for i in range(n):
        # get edge vector
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        edge = np.array([x2 - x1, y2 - y1])
        # rotate by 90 degrees (clockwise) to get the normal vector
        normal = np.array([edge[1], -edge[0]])
        normal = normal / np.linalg.norm(normal)  # normalize
        normals.append(normal)
    return normals


def convex_hull(points):
    """Graham scan algorithm for computing the convex hull of a set of 2D points."""
    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def walk_contour(x, y, visited, mask):
    """Depth-first search to extract a contour."""
    contour = []
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if not visited[cx, cy] and mask[cx, cy]:
            visited[cx, cy] = True
            # we typically store the y-coordinate in the first dimension
            contour.append((cy, cx))
            # Add neighbors (8-connectivity)
            neighbors = [
                (cx - 1, cy),
                (cx + 1, cy),
                (cx, cy - 1),
                (cx, cy + 1),
                (cx - 1, cy - 1),
                (cx + 1, cy + 1),
                (cx - 1, cy + 1),
                (cx + 1, cy - 1),
            ]
            for nx, ny in neighbors:
                if not visited[nx, ny]:
                    if 0 <= nx < mask.shape[0] and 0 <= ny < mask.shape[1]:
                        stack.append((nx, ny))
    return contour


def find_contours_simple(mask):
    """Simple DFS-based contour finding."""
    visited = np.zeros_like(mask, dtype=bool)
    contours = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if not mask[i, j] or visited[i, j]:
                continue
            contour = walk_contour(i, j, visited, mask)
            if contour:
                contour = convex_hull(np.array(contour))
                contours.append(contour)
    return contours


def mask_to_polygons(
    mask, min_area=10.0, approximation_tol=2.0, contour_finder="simple"
):
    """Converts a binary segmentation mask to polygon representations.

    Args:
        mask: Binary segmentation mask (2D numpy array where 1 indicates
            text regions and 0 is background).
        min_area: Minimum area for a polygon to be considered valid. Defaults
            to 10.0.
        approximation_tol: Approximation tolerance for simplifying polygons
             (higher for less detail). Defaults to 2.0.
        contour_finder: Determines the method for contour finding. Possible
            values are "simple", which detects connected regions by walking
            the image, and "opencv", which uses OpenCV's contour finder if
            available. Defaults to "simple".

    Returns:
        A list of polygons, where each polygon is represented as a list of
        (x, y) points.
    """

    if contour_finder == "opencv":
        import cv2

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = [
            contour.squeeze(axis=1)
            for contour in contours
            if contour.shape[0] > 2
        ]  # Squeeze to 2D
    elif contour_finder == "simple":
        contours = find_contours_simple(mask)
    else:
        raise ValueError(
            f"Invalid argument for contour_finder: {contour_finder}."
        )

    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        simplified = simplify_polygon(contour, approximation_tol)
        # Lower-bound the area of detected polygons to filter out noise
        area = compute_polygon_area(simplified)
        if area >= min_area:
            polygons.append(simplified.tolist())
    return polygons


def unclip_polygon(polygon, unclip_ratio):
    """Enlarges a polygon by a specified ratio.

    Args:
        polygon: List of (x, y) coordinates of the polygon vertices.
        unclip_ratio: Expansion ratio.

    Returns:
        List of expanded polygon coordinates.
    """

    # compute the expansion distance based on the polygon's area and perimeter
    area = compute_polygon_area(polygon)
    perimeter = sum(
        np.linalg.norm(
            np.array(polygon[i]) - np.array(polygon[(i + 1) % len(polygon)])
        )
        for i in range(len(polygon))
    )
    distance = area * unclip_ratio / perimeter
    # enlarge the polygon by moving vertices outwards based on
    # (outwards-pointing) normals of edge vectors
    normals = compute_edge_normals(polygon)
    expanded_polygon = []
    for i, (x, y) in enumerate(polygon):
        # average the normals of the two adjacent edges
        prev_normal = normals[i - 1]
        curr_normal = normals[i]
        avg_normal = (prev_normal + curr_normal) / 2
        avg_normal = avg_normal / np.linalg.norm(avg_normal)  # normalize
        # offset the vertex along the averaged normal
        offset_x, offset_y = avg_normal * distance
        expanded_polygon.append((x + offset_x, y + offset_y))
    return expanded_polygon
