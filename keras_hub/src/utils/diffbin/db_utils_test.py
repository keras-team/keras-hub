import os
import unittest
import numpy as np
import keras
import tensorflow as tf

from .db_utils import Polygon, shrink_polygan, binary_search_smallest_width, project_point_to_line, project_point_to_segment, get_line_height, line_segment_intersection, fill_poly, get_mask, get_region_coordinate, get_coords_poly_projection, get_coords_poly_distance_keras, get_normalized_weight


class TestdbUtils(unittest.TestCase):
    def test_Polygon(self):
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        area = Polygon(coords)
        self.assertAlmostEqual(area, 1.0)

        coords = np.array([[0, 0], [2, 0], [2, 2]])
        area = Polygon(coords)
        self.assertAlmostEqual(area, 2.0)

    def test_shrink_polygan(self):
        poly = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        factor = 0.4
        shrinked_poly = shrink_polygan(poly, factor).numpy()
        expected = np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [1.0, 1.0]])
        np.testing.assert_array_almost_equal(shrinked_poly, expected)
        

    def test_binary_search_smallest_width(self):
        poly = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        height = binary_search_smallest_width(poly)
        self.assertIsInstance(height, int)
        self.assertTrue(height >= 0)

    def test_project_point_to_line(self):
        x = np.array([1, 1])
        u = np.array([0, 0])
        v = np.array([2, 2])
        projected = project_point_to_line(x, u, v).numpy()
        expected = np.array([1, 1])
        np.testing.assert_array_almost_equal(projected, expected)

    def test_project_point_to_segment(self):
        x = np.array([1, 1])
        u = np.array([0, 0])
        v = np.array([2, 2])
        projected = project_point_to_segment(x, u, v).numpy()
        expected = np.array([1, 1])
        np.testing.assert_array_almost_equal(projected, expected)

        x = np.array([3, 3])
        projected = project_point_to_segment(x,u,v).numpy()
        expected = np.array([2,2])
        np.testing.assert_array_almost_equal(projected, expected)

    def test_get_line_height(self):
        poly = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        height = get_line_height(poly)
        self.assertIsInstance(height, int)
        self.assertTrue(height >= 0)

    def test_line_segment_intersection(self):
        polygon = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        self.assertTrue(line_segment_intersection(1, 1, polygon))
        self.assertFalse(line_segment_intersection(3, 3, polygon))

    def test_fill_poly(self):
        vertices = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        image_shape = (4, 4)
        mask = fill_poly(vertices, image_shape).numpy()
        self.assertEqual(mask.shape, image_shape)
        self.assertTrue(np.any(mask))

    def test_get_mask(self):
        w, h = 4, 4
        polys = [[[0, 0], [2, 0], [2, 2], [0, 2]]]
        ignores = [False]
        mask = get_mask(w, h, polys, ignores).numpy()
        self.assertEqual(mask.shape, (h, w))
        self.assertTrue(np.any(mask))

    def test_get_region_coordinate(self):
        w, h = 4, 4
        polys = [[[0, 0], [2, 0], [2, 2], [0, 2]]]
        heights = [1]
        shrink = 0.1
        regions = get_region_coordinate(w, h, polys, heights, shrink)
        self.assertTrue(isinstance(regions, list))

    def test_get_coords_poly_projection(self):
        coords = np.array([[1, 1], [3, 3]])
        poly = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        projected = get_coords_poly_projection(coords, poly).numpy()
        self.assertEqual(projected.shape, coords.shape)

    def test_get_coords_poly_distance_keras(self):
        coords = np.array([[1, 1], [3, 3]])
        poly = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
        distances = get_coords_poly_distance_keras(coords, poly).numpy()
        self.assertEqual(distances.shape, (2,))

    def test_get_normalized_weight(self):
        heatmap = np.array([[0.1, 0.6], [0.8, 0.2]])
        mask = np.array([[1, 1], [1, 1]])
        weight = get_normalized_weight(heatmap, mask)
        self.assertEqual(weight.shape, heatmap.shape)

if __name__ == '__main__':
    unittest.main()
