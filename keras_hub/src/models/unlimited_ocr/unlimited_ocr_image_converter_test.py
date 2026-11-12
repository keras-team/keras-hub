import grain
import numpy as np
import pytest

from keras_hub.src.models.unlimited_ocr.unlimited_ocr_image_converter import (
    UnlimitedOCRImageConverter,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_image_converter import (
    _get_optimal_tiled_canvas_tf,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_image_converter import (
    get_optimal_tiled_canvas,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.tests.test_case import assert_grain_safe_types
from keras_hub.src.utils.tensor_utils import tf


class UnlimitedOCRImageConverterTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "global_size": 32,
            "tile_size": 16,
            "min_tiles": 2,
            "max_tiles": 6,
            "patch_size": 4,
            "downsample_ratio": 2,
        }
        self.converter = UnlimitedOCRImageConverter(**self.init_kwargs)

    def test_serialization(self):
        self.run_serialization_test(self.converter)

    def test_grid_selection(self):
        def grid(height, width):
            return get_optimal_tiled_canvas((height, width), 640, 2, 32)

        # A square image picks the smallest grid with a square shape.
        self.assertEqual(grid(1000, 1000), (2, 2))
        # A 2:1 image picks two columns and one row.
        self.assertEqual(grid(700, 1400), (2, 1))
        # A 1:2 image picks one column and two rows.
        self.assertEqual(grid(1400, 700), (1, 2))

    def test_small_image_has_no_tiles(self):
        image = np.full((16, 16, 3), 255.0, dtype="float32")
        outputs = self.converter([image])
        self.assertEqual(outputs["pixel_values"].shape, (1, 32, 32, 3))
        self.assertEqual(outputs["pixel_values_local"].shape, (0, 16, 16, 3))
        self.assertAllEqual(outputs["patches_grid"], [[0, 0]])
        # (255 - 127.5) / 127.5 is 1 for white pixels.
        self.assertAllClose(outputs["pixel_values"], np.ones((1, 32, 32, 3)))

    def test_global_view_pads_to_a_square(self):
        # A 20x40 image scales by 32 / 40. The view is 16x32. The pad adds
        # 8 rows above and 8 rows below, with the background color.
        image = np.full((20, 40, 3), 255.0, dtype="float32")
        converter = UnlimitedOCRImageConverter(
            **{
                **self.init_kwargs,
                "crop_to_patches": False,
                "maximum_pad_value": 16,
            }
        )
        outputs = converter([image])
        view = np.array(outputs["pixel_values"])[0]
        background = (127 - 127.5) / 127.5
        self.assertAllClose(view[:8], np.full((8, 32, 3), background))
        self.assertAllClose(view[24:], np.full((8, 32, 3), background))
        self.assertAllClose(view[8:24], np.ones((16, 32, 3)))
        # `crop_to_patches=False` gives no tiles.
        self.assertEqual(outputs["pixel_values_local"].shape, (0, 16, 16, 3))
        self.assertAllEqual(outputs["patches_grid"], [[0, 0]])

    def test_small_global_view_stretches_to_a_square(self):
        # `global_size` is at most `maximum_pad_value` and the converter
        # does not tile. The view fills the square with no pad.
        image = np.full((20, 40, 3), 255.0, dtype="float32")
        converter = UnlimitedOCRImageConverter(
            **{
                **self.init_kwargs,
                "crop_to_patches": False,
                "maximum_pad_value": 32,
            }
        )
        view = np.array(converter([image])["pixel_values"])[0]
        self.assertAllClose(view, np.ones((32, 32, 3)))

    def test_tiles_are_cut_in_row_major_order(self):
        # A 16x32 image has an aspect ratio of 2, so the grid is 2 columns
        # and 1 row. The long side is above the tile size.
        image = np.zeros((16, 32, 3), dtype="float32")
        image[:, :16] = 255.0
        outputs = self.converter([image])
        self.assertAllEqual(outputs["patches_grid"], [[2, 1]])
        tiles = np.array(outputs["pixel_values_local"])
        self.assertEqual(tiles.shape, (2, 16, 16, 3))
        # The left tile is white and the right tile is black.
        self.assertAllClose(tiles[0], np.ones((16, 16, 3)))
        self.assertAllClose(tiles[1], -np.ones((16, 16, 3)))

    def test_tiles_of_all_images_follow_each_other(self):
        wide = np.zeros((16, 32, 3), dtype="float32")
        small = np.zeros((8, 8, 3), dtype="float32")
        tall = np.zeros((32, 16, 3), dtype="float32")
        outputs = self.converter([wide, small, tall])
        self.assertEqual(outputs["pixel_values"].shape, (3, 32, 32, 3))
        self.assertAllEqual(outputs["patches_grid"], [[2, 1], [0, 0], [1, 2]])
        self.assertEqual(outputs["pixel_values_local"].shape, (4, 16, 16, 3))

    def test_number_of_image_tokens(self):
        converter = UnlimitedOCRImageConverter()
        # The global view has 16 rows of 16 tokens and a newline each, then
        # one separator.
        self.assertEqual(converter.get_num_image_tokens(0, 0), 16 * 17 + 1)
        # Two tile columns and one row add 10 rows of 2 * 10 + 1 tokens.
        self.assertEqual(
            converter.get_num_image_tokens(2, 1), 16 * 17 + 1 + 10 * 21
        )

    def test_grain_matches_direct_call(self):
        # The Python path must return NumPy arrays inside Grain.
        images = np.zeros((2, 16, 32, 3), dtype="float32")
        images[:, :, :16] = 255.0
        direct = self.converter(images)
        dataset = grain.MapDataset.source([images])
        (output,) = list(dataset.map(self.converter))
        assert_grain_safe_types(output)
        for key in direct:
            self.assertAllClose(direct[key], output[key])

    @pytest.mark.skipif(tf is None, reason="TensorFlow is not installed.")
    def test_tf_data_matches_direct_call(self):
        images = np.zeros((2, 16, 32, 3), dtype="float32")
        images[:, :, :16] = 255.0
        direct = self.converter(images)
        ds = tf.data.Dataset.from_tensor_slices(images).batch(2)
        mapped = next(iter(ds.map(self.converter)))
        for key in direct:
            self.assertAllClose(direct[key], mapped[key].numpy())

    @pytest.mark.skipif(tf is None, reason="TensorFlow is not installed.")
    def test_tf_grid_matches_python_grid(self):
        heights = (641, 700, 1000, 1280, 2000, 3001)
        widths = (641, 900, 1280, 1920, 4000)
        sizes = [(h, w) for h in heights for w in widths]
        sizes += [(1000, 1000), (1280, 640), (640, 1280), (2560, 640)]
        for height, width in sizes:
            expected = get_optimal_tiled_canvas((height, width), 640, 2, 32)
            columns, rows = _get_optimal_tiled_canvas_tf(
                tf.constant(height), tf.constant(width), 640, 2, 32
            )
            self.assertEqual((int(columns), int(rows)), expected)

    @pytest.mark.skipif(tf is None, reason="TensorFlow is not installed.")
    def test_tf_path_matches_python_path(self):
        rng = np.random.RandomState(0)
        images = [
            rng.randint(0, 256, (h, w, 3)).astype("float32")
            for h, w in [(16, 32), (8, 8), (32, 16), (20, 40), (45, 30)]
        ]
        for extra in (
            {},
            {"crop_to_patches": False, "maximum_pad_value": 16},
            {"crop_to_patches": False, "maximum_pad_value": 32},
        ):
            converter = UnlimitedOCRImageConverter(
                **{**self.init_kwargs, **extra}
            )
            for image in images:
                expected = converter._call_python([image])
                actual = converter._call_tf(tf.constant(image))
                for key in expected:
                    self.assertEqual(expected[key].shape, actual[key].shape)
                    self.assertAllClose(expected[key], np.array(actual[key]))

    @pytest.mark.skipif(tf is None, reason="TensorFlow is not installed.")
    def test_tf_path_matches_python_path_for_a_batch(self):
        images = np.random.RandomState(2).randint(0, 256, (3, 45, 30, 3))
        images = images.astype("float32")
        expected = self.converter._call_python(images)
        actual = self.converter._call_tf(tf.constant(images))
        for key in expected:
            self.assertEqual(expected[key].shape, actual[key].shape)
            self.assertAllClose(expected[key], np.array(actual[key]))

    @pytest.mark.skipif(tf is None, reason="TensorFlow is not installed.")
    def test_tf_data_unknown_size_matches_python_path(self):
        image = np.random.RandomState(1).randint(0, 256, (45, 30, 3))
        image = image.astype("float32")
        ds = tf.data.Dataset.from_generator(
            lambda: iter([image]),
            output_signature=tf.TensorSpec((None, None, 3), "float32"),
        )
        mapped = next(iter(ds.map(self.converter)))
        expected = self.converter._call_python([image])
        for key in expected:
            self.assertAllClose(expected[key], mapped[key].numpy())
