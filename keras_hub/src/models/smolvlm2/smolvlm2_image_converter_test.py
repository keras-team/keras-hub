import grain
import keras
import numpy as np
from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.smolvlm2.smolvlm2_image_converter import (
    SmolVLM2ImageConverter,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import running_on_gpu

# `tf.image.resize(method=<name>, antialias=True)` on the (1, 8, 12, 3)
# hard-edge image in `test_lanczos_resize_matches_reference`, resized to
# 16x16, clipped to [0, 255] and rounded. The converter resizes width then
# height with a clip + round after each pass (the PIL uint8 pipeline);
# every row of this image is identical, so the height pass is an identity
# and the result is the rounded single-pass row. Every row and channel is
# the same.
_LANCZOS3_REFERENCE_ROW = [
    0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 23.0,
    232.0, 255.0, 247.0, 255.0, 255.0, 255.0, 255.0, 255.0,
]  # fmt: skip
_LANCZOS5_REFERENCE_ROW = [
    0.0, 0.0, 1.0, 0.0, 0.0, 15.0, 0.0, 25.0,
    230.0, 255.0, 240.0, 255.0, 255.0, 254.0, 255.0, 255.0,
]  # fmt: skip


class SmolVLM2ImageConverterTest(TestCase):
    def test_single_crop_output(self):
        """Small image that fits in one crop produces 1 sub-image."""
        converter = SmolVLM2ImageConverter(
            max_image_size=32,
            size=32,
            do_image_splitting=True,
            scale=[1 / 255.0] * 3,
            offset=[0.0] * 3,
        )
        # 10x10 image — after resize to longest_edge=32 it stays <=32,
        # so it fits in a single 32×32 crop.
        img = np.random.randint(0, 256, size=(10, 10, 3)).astype("uint8")
        result = converter(img)

        self.assertIn("pixel_values", result)
        self.assertIn("rows", result)
        self.assertIn("cols", result)

        pixel_values = ops.convert_to_numpy(result["pixel_values"])
        # Single crop → rows=0, cols=0.
        self.assertEqual(int(result["rows"]), 0)
        self.assertEqual(int(result["cols"]), 0)
        # Shape: (1, max_image_size, max_image_size, 3)
        self.assertEqual(pixel_values.shape, (1, 32, 32, 3))

    def test_multi_crop_output(self):
        """Larger image is split into multiple crops + global view."""
        converter = SmolVLM2ImageConverter(
            max_image_size=16,
            size=64,
            do_image_splitting=True,
            scale=[1 / 255.0] * 3,
            offset=[0.0] * 3,
        )
        # 100x100 image — after resize to longest_edge=64 and snap to
        # multiples of 16, should produce multiple 16×16 crops.
        img = np.random.randint(0, 256, size=(100, 100, 3)).astype("uint8")
        result = converter(img)

        pixel_values = ops.convert_to_numpy(result["pixel_values"])
        num_rows = int(result["rows"])
        num_cols = int(result["cols"])

        # Should be split.
        self.assertGreater(num_rows, 0)
        self.assertGreater(num_cols, 0)

        # num_sub_images = rows * cols + 1 (global view).
        expected_sub_images = num_rows * num_cols + 1
        self.assertEqual(pixel_values.shape[0], expected_sub_images)
        self.assertEqual(pixel_values.shape[1], 16)
        self.assertEqual(pixel_values.shape[2], 16)
        self.assertEqual(pixel_values.shape[3], 3)

    def test_no_splitting(self):
        """do_image_splitting=False always produces 1 crop."""
        converter = SmolVLM2ImageConverter(
            max_image_size=16,
            size=64,
            do_image_splitting=False,
            scale=[1 / 255.0] * 3,
            offset=[0.0] * 3,
        )
        img = np.random.randint(0, 256, size=(100, 60, 3)).astype("uint8")
        result = converter(img)

        pixel_values = ops.convert_to_numpy(result["pixel_values"])
        self.assertEqual(int(result["rows"]), 0)
        self.assertEqual(int(result["cols"]), 0)
        self.assertEqual(pixel_values.shape, (1, 16, 16, 3))

    def test_normalization_range(self):
        """Verify pixels normalized to [-1, 1] with default HF params."""
        # HF uses mean=0.5, std=0.5 → scale=1/(255*0.5), offset=-1.0
        converter = SmolVLM2ImageConverter(
            max_image_size=32,
            size=64,
            do_image_splitting=False,
            scale=[1.0 / (0.5 * 255)] * 3,
            offset=[-0.5 / 0.5] * 3,
        )
        img = np.random.randint(0, 256, size=(20, 20, 3)).astype("uint8")
        result = converter(img)

        pixel_values = ops.convert_to_numpy(result["pixel_values"])
        # Should be in [-1, 1] range (approximately).
        self.assertAllInRange(pixel_values, -1.05, 1.05)

    def test_config_roundtrip(self):
        """Test serialization / deserialization."""
        converter = SmolVLM2ImageConverter(
            max_image_size=512,
            size=2048,
            do_image_splitting=True,
        )
        self.run_serialization_test(converter)
        config = converter.get_config()
        self.assertEqual(config["max_image_size"], 512)
        self.assertEqual(config["size"], 2048)
        self.assertTrue(config["do_image_splitting"])

    def test_square_image(self):
        """Square image is handled correctly."""
        converter = SmolVLM2ImageConverter(
            max_image_size=32,
            size=64,
            do_image_splitting=True,
            scale=[1 / 255.0] * 3,
            offset=[0.0] * 3,
        )
        img = np.random.randint(0, 256, size=(32, 32, 3)).astype("uint8")
        result = converter(img)

        pixel_values = ops.convert_to_numpy(result["pixel_values"])
        self.assertEqual(pixel_values.shape[1], 32)
        self.assertEqual(pixel_values.shape[2], 32)

    def test_batched_input_without_splitting(self):
        """A batch of images is resized without a Python loop."""
        converter = SmolVLM2ImageConverter(
            max_image_size=16,
            size=32,
            do_image_splitting=False,
            scale=[1 / 255.0] * 3,
            offset=[0.0] * 3,
        )
        images = np.random.randint(0, 256, size=(4, 20, 24, 3)).astype("uint8")
        result = converter(images)

        pixel_values = ops.convert_to_numpy(result["pixel_values"])
        self.assertEqual(pixel_values.shape, (4, 16, 16, 3))
        self.assertEqual(int(result["rows"]), 0)
        self.assertEqual(int(result["cols"]), 0)

    def test_batched_input_with_splitting_raises(self):
        """Splitting a batch is ambiguous and must raise."""
        converter = SmolVLM2ImageConverter(
            max_image_size=16,
            size=32,
            do_image_splitting=True,
            scale=[1 / 255.0] * 3,
            offset=[0.0] * 3,
        )
        images = np.random.randint(0, 256, size=(2, 20, 24, 3)).astype("uint8")
        with self.assertRaisesRegex(ValueError, "do_image_splitting"):
            converter(images)

    def test_grain_pipeline_returns_numpy(self):
        """Inside Grain every output, including `rows`/`cols`, is NumPy."""
        converter = SmolVLM2ImageConverter(
            max_image_size=16,
            size=32,
            do_image_splitting=True,
            scale=[1 / 255.0] * 3,
            offset=[0.0] * 3,
        )
        images = [
            np.random.randint(0, 256, size=(20, 24, 3)).astype("uint8")
            for _ in range(2)
        ]
        for output in grain.MapDataset.source(images).map(converter):
            self.assertIsInstance(output["pixel_values"], np.ndarray)
            # On TF, `.numpy()` of a 0-d tensor gives a NumPy scalar
            # (`np.int32`) rather than a 0-d array. Both pickle fine.
            for key in ("rows", "cols"):
                self.assertIsInstance(output[key], (np.ndarray, np.generic))

    def test_ragged_list_without_splitting_stacks(self):
        """Different-size images give one dict when each is one crop."""
        converter = SmolVLM2ImageConverter(
            max_image_size=16,
            size=32,
            do_image_splitting=False,
            scale=[1 / 255.0] * 3,
            offset=[0.0] * 3,
        )
        images = [
            np.random.randint(0, 256, size=(20, 24, 3)).astype("uint8"),
            np.random.randint(0, 256, size=(30, 12, 3)).astype("uint8"),
        ]
        result = converter(images)
        self.assertIsInstance(result, dict)
        expected = np.concatenate(
            [
                ops.convert_to_numpy(converter(image)["pixel_values"])
                for image in images
            ]
        )
        self.assertAllClose(
            ops.convert_to_numpy(result["pixel_values"]), expected
        )
        self.assertEqual(int(result["rows"]), 0)
        self.assertEqual(int(result["cols"]), 0)

    def test_ragged_list_with_splitting_stays_a_list(self):
        """With splitting, crop counts differ, so one dict per image."""
        converter = SmolVLM2ImageConverter(
            max_image_size=16,
            size=32,
            do_image_splitting=True,
            scale=[1 / 255.0] * 3,
            offset=[0.0] * 3,
        )
        images = [
            np.random.randint(0, 256, size=(20, 24, 3)).astype("uint8"),
            np.random.randint(0, 256, size=(30, 12, 3)).astype("uint8"),
        ]
        result = converter(images)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

    def test_same_size_list_with_splitting_stays_a_list(self):
        """Same-size images are not stacked into a batch when splitting."""
        converter = SmolVLM2ImageConverter(
            max_image_size=16,
            size=32,
            do_image_splitting=True,
            scale=[1 / 255.0] * 3,
            offset=[0.0] * 3,
        )
        images = [
            np.random.randint(0, 256, size=(20, 24, 3)).astype("uint8")
            for _ in range(2)
        ]
        result = converter(images)
        self.assertIsInstance(result, list)
        for output, image in zip(result, images):
            self.assertAllClose(
                ops.convert_to_numpy(output["pixel_values"]),
                ops.convert_to_numpy(converter(image)["pixel_values"]),
            )

    @parameterized.named_parameters(
        ("lanczos3", "lanczos3", _LANCZOS3_REFERENCE_ROW),
        ("lanczos5", "lanczos5", _LANCZOS5_REFERENCE_ROW),
    )
    def test_lanczos_resize_matches_reference(
        self, interpolation, expected_row
    ):
        """Both Lanczos filters run on every backend and match TF's."""
        converter = SmolVLM2ImageConverter(
            max_image_size=16,
            size=16,
            do_image_splitting=False,
            interpolation=interpolation,
            scale=[1.0] * 3,
            offset=[0.0] * 3,
        )
        # A hard vertical edge. Lanczos rings past [0, 255] here, so the
        # clip is exercised too.
        images = np.zeros((1, 8, 12, 3), dtype="float32")
        images[:, :, 6:, :] = 255.0
        pixel_values = ops.convert_to_numpy(converter(images)["pixel_values"])
        expected = np.broadcast_to(
            np.array(expected_row)[None, None, :, None], (1, 16, 16, 3)
        )
        atol = 0.1
        if (
            keras.config.backend() == "tensorflow"
            and running_on_gpu()
            and interpolation == "lanczos5"
        ):
            # The GPU kernel is up to 0.25 off before rounding, which can
            # flip a value by one u8 after it.
            atol = 1.0
        self.assertAllClose(pixel_values, expected, atol=atol)

    def test_empty_batch_lanczos_resize(self):
        """An empty batch resizes to an empty batch on every backend."""
        converter = SmolVLM2ImageConverter(
            max_image_size=16,
            size=32,
            do_image_splitting=False,
            scale=[1 / 255.0] * 3,
            offset=[0.0] * 3,
        )
        images = np.zeros((0, 20, 24, 3), dtype="uint8")
        pixel_values = ops.convert_to_numpy(converter(images)["pixel_values"])
        self.assertEqual(pixel_values.shape, (0, 16, 16, 3))

    def test_image_size_is_rejected(self):
        """The base resize to `image_size` would distort the crops."""
        with self.assertRaisesRegex(ValueError, "image_size"):
            SmolVLM2ImageConverter(image_size=(16, 16))

    def test_image_size_set_after_init_is_rejected(self):
        """Setting `image_size` later must not resize the crops."""
        converter = SmolVLM2ImageConverter(max_image_size=16, size=32)
        converter.image_size = (8, 8)
        image = np.random.randint(0, 256, size=(20, 24, 3)).astype("uint8")
        with self.assertRaisesRegex(ValueError, "image_size"):
            converter(image)

    def test_tf_data_map_jax_torch(self):
        """Map over tf.data works on all backends for training inputs."""
        import tensorflow as tf

        converter = SmolVLM2ImageConverter(
            max_image_size=16,
            size=32,
            do_image_splitting=False,
        )
        images = np.random.randint(0, 256, size=(4, 20, 24, 3)).astype("uint8")
        ds = tf.data.Dataset.from_tensor_slices(images).batch(2).map(converter)
        for batch in ds:
            self.assertEqual(batch["pixel_values"].shape, (2, 16, 16, 3))

    def test_dynamic_sizing_matches_static(self):
        """The `tf.data` dynamic-shape sizing equals the Python helper.

        Covers both orientations, odd sizes (the even bump), tiny inputs
        and the 4096 cap, which only engages when `size` exceeds it.
        """
        import tensorflow as tf

        cases = [
            (480, 640),
            (640, 480),
            (333, 517),
            (517, 333),
            (1, 1),
            (7, 5),
            (2500, 5000),
            (5000, 2500),
        ]
        # In production this helper only runs from `_call_tf`, which
        # `@preprocessing_function` already pins to CPU. Calling it directly
        # here would place it on the GPU when one is visible, where TF's
        # float64 `Floor` is a JIT kernel that needs libdevice and fails on
        # the GPU CI runners. Match the production placement.
        with tf.device("/CPU:0"):
            for size in (512, 2048, 8192):
                converter = SmolVLM2ImageConverter(
                    max_image_size=16, size=size, do_image_splitting=False
                )
                for h, w in cases:
                    expected = converter._sizes_for(h, w)
                    got = converter._sizes_for_dynamic(
                        tf.constant(h, "int32"), tf.constant(w, "int32")
                    )
                    got = (int(got[0].numpy()), int(got[1].numpy()))
                    self.assertEqual(
                        got, expected, msg=f"size={size} hw={h, w}"
                    )

    def _pillow_pipeline(self, image, converter):
        """HF's two Pillow resizes: longest edge to `size`, then square."""
        from PIL import Image

        h, w = image.shape[:2]
        new_h, new_w = converter._sizes_for(h, w)
        ms = converter.max_image_size
        out = Image.fromarray(image).resize((new_w, new_h), Image.LANCZOS)
        out = out.resize((ms, ms), Image.LANCZOS)
        return np.asarray(out).astype("float32")

    @parameterized.named_parameters(
        ("diagonal_edge", "edge"),
        ("noise", "noise"),
    )
    def test_lanczos3_matches_pillow_bit_exact(self, kind):
        """`lanczos3` reproduces Pillow's fixed-point resample exactly.

        HF resizes with `PIL.Image.LANCZOS`; Pillow quantizes the taps to
        22 bits, accumulates in int32 and clips to uint8 after each pass.
        A float pipeline lands within 1-2 u8 of that, which is enough to
        flip a greedy token on a near tie, so the check is equality, not a
        tolerance. The diagonal edge rings past [0, 255] on the upscale and
        only matches if the clip happens between the two passes too.
        """
        rows, cols = np.mgrid[0:24, 0:32]
        if kind == "edge":
            image = np.where(cols * 24 > rows * 32, 255, 0).astype("uint8")
            image = np.broadcast_to(image[:, :, None], (24, 32, 3))
        else:
            image = np.random.default_rng(0).integers(
                0, 256, (24, 32, 3), dtype="uint8"
            )
        converter = SmolVLM2ImageConverter(
            # 24x32 -> 72x96 (upscale), then -> 40x40 (downscale). These
            # sizes separate Pillow's result from a float two-pass Lanczos
            # (1 u8), a 20-bit tap quantization (1 u8) and a missing
            # inter-pass clip (15 u8); 4x/3x ratios do not.
            max_image_size=40,
            size=96,
            do_image_splitting=False,
            scale=[1.0] * 3,
            offset=[0.0] * 3,
            interpolation="lanczos3",
        )
        expected = self._pillow_pipeline(np.ascontiguousarray(image), converter)
        actual = ops.convert_to_numpy(converter(image)["pixel_values"])
        self.assertAllEqual(actual, expected[None])

    def test_batch_paths_match_per_image_path(self):
        """`_resize_batch` and `_call_tf` equal `_convert_image` exactly.

        All three must apply the longest-edge `size` step first and the
        same resample. With `size=2048` the batch paths used to skip that
        step and go straight to `max_image_size`, so `fit()` and
        `generate()` saw different pixels (14 u8 apart on COCO).
        """
        converter = SmolVLM2ImageConverter(
            max_image_size=512,
            size=2048,
            do_image_splitting=False,
            scale=[1 / 255.0] * 3,
            offset=[0.0] * 3,
            interpolation="lanczos3",
        )
        rows, cols = np.mgrid[0:24, 0:32]
        image = np.where(cols * 24 > rows * 32, 255, 0).astype("uint8")
        image = np.ascontiguousarray(
            np.broadcast_to(image[:, :, None], (24, 32, 3))
        )
        images = image[None]

        per_image = ops.convert_to_numpy(
            converter._convert_image(image)["pixel_values"]
        )
        batch = ops.convert_to_numpy(
            converter._resize_batch(images)["pixel_values"]
        )
        tf_path = ops.convert_to_numpy(
            converter._call_tf(images)["pixel_values"]
        )
        expected = self._pillow_pipeline(image, converter)[None] / 255.0
        self.assertAllEqual(batch, per_image)
        self.assertAllEqual(tf_path, per_image)
        self.assertAllClose(per_image, expected, atol=1e-6)
