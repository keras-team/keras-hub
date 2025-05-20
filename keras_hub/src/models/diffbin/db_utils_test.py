import keras
import numpy as np

from keras_hub.src.models.diffbin.db_utils import Polygon
from keras_hub.src.models.diffbin.db_utils import fill_poly_keras
from keras_hub.src.models.diffbin.db_utils import get_coords_poly_distance
from keras_hub.src.models.diffbin.db_utils import get_coords_poly_projection
from keras_hub.src.models.diffbin.db_utils import get_mask
from keras_hub.src.models.diffbin.db_utils import get_normalized_weight
from keras_hub.src.models.diffbin.db_utils import project_point_to_line
from keras_hub.src.models.diffbin.db_utils import project_point_to_segment
from keras_hub.src.tests.test_case import TestCase


class TestDBUtils(TestCase):
    def test_polygon_area(self):
        coords = [[0, 0], [2, 0], [2, 2], [0, 2]]
        area = Polygon(coords)
        assert area == 4.0

        coords_np = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        area_np = Polygon(coords_np)
        assert area_np == 4.0

        triangle = [[0, 0], [1, 0], [0, 1]]
        area_triangle = Polygon(triangle)
        assert area_triangle == 0.5

    def test_project_point_to_line(self):
        x = [1, 1]
        u = [0, 0]
        v = [2, 0]
        projection = project_point_to_line(x, u, v)
        assert np.allclose(projection, [1, 0])

        x_np = np.array([1, 1])
        u_np = np.array([0, 0])
        v_np = np.array([2, 0])
        projection_np = project_point_to_line(x_np, u_np, v_np)
        assert np.allclose(projection_np, [1, 0])

    def test_project_point_to_segment(self):
        x = [1, 1]
        u = [0, 0]
        v = [2, 0]
        projection = project_point_to_segment(x, u, v)
        assert np.allclose(projection, [1, 0])

        x_off = [3, 1]
        projection_off = project_point_to_segment(x_off, u, v)
        assert np.allclose(projection_off, [2, 0])

    def test_fill_poly_keras(self):
        vertices = [[0, 0], [2, 0], [2, 2], [0, 2]]
        image_shape = (3, 3)
        mask = fill_poly_keras(vertices, image_shape)
        assert mask.shape == image_shape
        assert keras.ops.any(mask >= 0) and keras.ops.any(mask <= 1)

    def test_get_mask(self):
        w, h = 3, 3
        polys = [[[0, 0], [2, 0], [2, 2], [0, 2]]]
        ignores = [False]
        mask = get_mask(w, h, polys, ignores)
        assert mask.shape == (h, w)
        assert keras.ops.any(mask >= 0) and keras.ops.any(mask <= 1)

    def test_get_coords_poly_projection(self):
        coords = [[1, 1], [3, 3]]
        poly = [[0, 0], [2, 0], [2, 2], [0, 2]]
        projection = get_coords_poly_projection(coords, poly)
        assert projection.shape == (len(coords), 2)

    def test_get_coords_poly_distance(self):
        coords = [[1, 1], [3, 3]]
        poly = [[0, 0], [2, 0], [2, 2], [0, 2]]
        distances = get_coords_poly_distance(coords, poly)
        assert distances.shape == (len(coords),)
        assert keras.ops.all(distances >= 0)

    def test_get_normalized_weight(self):
        heatmap = np.array([[0.1, 0.6], [0.4, 0.8]])
        mask = np.array([[1, 1], [1, 1]])
        weight = get_normalized_weight(heatmap, mask)
        assert weight.shape == heatmap.shape
        assert np.all(weight >= 0)

        mask_partial = np.array([[1, 0], [1, 1]])
        weight_partial = get_normalized_weight(heatmap, mask_partial)
        assert np.all(weight_partial >= 0)