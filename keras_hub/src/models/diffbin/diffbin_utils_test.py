import keras
import numpy as np

from keras_hub.src.models.diffbin.diffbin_utils import Polygon
from keras_hub.src.models.diffbin.diffbin_utils import fill_poly_keras
from keras_hub.src.models.diffbin.diffbin_utils import get_coords_poly_distance
from keras_hub.src.models.diffbin.diffbin_utils import (
    get_coords_poly_projection,
)
from keras_hub.src.models.diffbin.diffbin_utils import get_mask
from keras_hub.src.models.diffbin.diffbin_utils import get_normalized_weight
from keras_hub.src.models.diffbin.diffbin_utils import project_point_to_line
from keras_hub.src.models.diffbin.diffbin_utils import project_point_to_segment
from keras_hub.src.tests.test_case import TestCase


class TestDBUtils(TestCase):
    def test_polygon_area(self):
        coords = [[0, 0], [2, 0], [2, 2], [0, 2]]
        area = Polygon(coords)
        area_np = keras.ops.convert_to_numpy(area)
        assert np.allclose(area_np, 4.0)

        triangle = [[0, 0], [1, 0], [0, 1]]
        area_triangle = Polygon(triangle)
        area_triangle_np = keras.ops.convert_to_numpy(area_triangle)
        assert np.allclose(area_triangle_np, 0.5)

    def test_project_point_to_line(self):
        x = np.array([1, 1], dtype=np.float32)
        u = np.array([0, 0], dtype=np.float32)
        v = np.array([2, 0], dtype=np.float32)
        projection = project_point_to_line(x, u, v)
        projection_np = keras.ops.convert_to_numpy(projection)
        assert np.allclose(projection_np, [1, 0])

    def test_project_point_to_segment(self):
        x = np.array([1, 1], dtype=np.float32)
        u = np.array([0, 0], dtype=np.float32)
        v = np.array([2, 0], dtype=np.float32)
        projection = project_point_to_segment(x, u, v)
        projection_np = keras.ops.convert_to_numpy(projection)
        assert np.allclose(projection_np, [1, 0])

        x_off = [3, 1]
        projection_off = project_point_to_segment(x_off, u, v)
        projection_off_np = keras.ops.convert_to_numpy(projection_off)
        assert np.allclose(projection_off_np, [2, 0])

    def test_fill_poly_keras(self):
        vertices = [[0, 0], [2, 0], [2, 2], [0, 2]]
        image_shape = (3, 3)
        mask = fill_poly_keras(vertices, image_shape)
        assert np.allclose(mask.shape, image_shape)

    def test_get_mask(self):
        w, h = 3, 3
        polys = [[[0, 0], [2, 0], [2, 2], [0, 2]]]
        ignores = [False]
        mask = get_mask(w, h, polys, ignores)
        assert np.allclose(mask.shape, (h, w))
        assert keras.ops.any(mask >= 0) and keras.ops.any(mask <= 1)

    def test_get_coords_poly_projection(self):
        coords = [[1, 1], [3, 3]]
        poly = [[0, 0], [2, 0], [2, 2], [0, 2]]
        projection = get_coords_poly_projection(coords, poly)
        assert np.allclose(projection.shape, (len(coords), 2))

    def test_get_coords_poly_distance(self):
        coords = [[1, 1], [3, 3]]
        poly = [[0, 0], [2, 0], [2, 2], [0, 2]]
        distances = get_coords_poly_distance(coords, poly)
        assert np.allclose(distances.shape, (len(coords),))

    def test_get_normalized_weight(self):
        heatmap = np.array([[0.1, 0.6], [0.4, 0.8]], dtype=np.float32)
        mask = np.array([[1, 1], [1, 1]], dtype=np.float32)
        weight = get_normalized_weight(heatmap, mask)
        assert np.allclose(weight.shape, heatmap.shape)
        weight_np = keras.ops.convert_to_numpy(weight)
        assert np.all(weight_np >= 0)

        mask_partial = np.array([[1, 0], [1, 1]])
        weight_partial = get_normalized_weight(heatmap, mask_partial)
        assert np.allclose(weight_partial.shape, heatmap.shape)
        weight_partial_np = keras.ops.convert_to_numpy(weight_partial)
        assert np.all(weight_partial_np >= 0)
