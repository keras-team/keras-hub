import numpy as np

from keras_hub.src.models.image_text_detector import mask_to_polygons
from keras_hub.src.models.image_text_detector import unclip_polygon
from keras_hub.src.tests.test_case import TestCase


class PolygonFunctionsTest(TestCase):
    def test_mask_to_polygons(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        # detect two square regions
        mask[5:15, 10:20] = 1
        mask[35:45, 30:40] = 1
        detected_polygons = mask_to_polygons(mask, contour_finder="simple")
        self.assertEqual(
            detected_polygons,
            [
                [[10, 5], [19, 5], [19, 14], [10, 14]],
                [[30, 35], [39, 35], [39, 44], [30, 44]],
            ],
        )

    def test_unclip(self):
        polygon = [(10, 10), (20, 10), (20, 20), (10, 20)]
        unclip_ratio = 1.5
        unclipped_box = unclip_polygon(polygon, unclip_ratio)
        self.assertAllClose(
            unclipped_box,
            [(7.348, 7.348), (22.65, 7.348), (22.65, 22.65), (7.348, 22.65)],
            rtol=1e-3,
        )
