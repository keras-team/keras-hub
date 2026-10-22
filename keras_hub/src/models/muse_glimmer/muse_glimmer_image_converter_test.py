import keras
import numpy as np

from keras_hub.src.models.muse_glimmer.muse_glimmer_image_converter import (
    MuseGlimmerImageConverter,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_image_converter import (
    _smart_resize,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_image_converter import (
    _smart_resize_tf,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.tensor_utils import tf

# Expected targets of the HF `smart_resize` for `patch_size=28` and
# `max_tokens=144`. Each (height, width) is a size where two grids have the
# same aspect error, so the HF `set` order decides the result.
HF_SMART_RESIZE_TIES = {
    (143, 153): (168, 168),
    (104, 108): (112, 112),
    (58, 62): (84, 84),
    (546, 28): (560, 28),
    (805, 770): (336, 308),
    (3842, 2034): (448, 224),
    (1590, 3816): (224, 504),
}


class MuseGlimmerImageConverterTest(TestCase):
    def test_smart_resize_matches_hf_tie_breaks(self):
        for (height, width), expected in HF_SMART_RESIZE_TIES.items():
            self.assertEqual(
                _smart_resize(height, width, 14, 2, 144),
                expected,
                msg=f"size {(height, width)}",
            )

    def test_smart_resize_tf_matches_hf_tie_breaks(self):
        if tf is None:
            self.skipTest("TensorFlow is not installed.")

        @tf.function(
            input_signature=[
                tf.TensorSpec([], tf.int32),
                tf.TensorSpec([], tf.int32),
            ]
        )
        def resize(height, width):
            return _smart_resize_tf(height, width, 14, 2, 144)

        for (height, width), expected in HF_SMART_RESIZE_TIES.items():
            target_h, target_w = resize(height, width)
            self.assertEqual(
                (int(target_h), int(target_w)),
                expected,
                msg=f"size {(height, width)}",
            )

    def test_integer_inputs_round_after_resize(self):
        converter = MuseGlimmerImageConverter(
            patch_size=2,
            patch_temporal=2,
            merge_size=1,
            max_image_tokens=1,
            scale=2 / 255.0,
            offset=-1.0,
            interpolation="bilinear",
            antialias=True,
        )
        image = np.arange(4 * 4 * 3, dtype="uint8").reshape(4, 4, 3)

        output = converter(image)
        normalized = (output["patches"] + 1.0) / (2.0 / 255.0)

        self.assertAllClose(normalized, keras.ops.round(normalized))

    def test_patch_layout_is_channel_first(self):
        converter = MuseGlimmerImageConverter(
            patch_size=2,
            patch_temporal=2,
            merge_size=1,
            max_image_tokens=16,
            interpolation="nearest",
        )
        image = np.arange(4 * 4 * 3, dtype="float32").reshape(4, 4, 3)

        output = converter(image)

        expected_patch = np.array(
            [
                0,
                3,
                12,
                15,
                1,
                4,
                13,
                16,
                2,
                5,
                14,
                17,
            ]
            * 2,
            dtype="float32",
        )
        self.assertAllClose(output["patches"][0], expected_patch)

    def test_convert(self):
        converter = MuseGlimmerImageConverter(
            patch_size=4,
            patch_temporal=2,
            merge_size=2,
            max_image_tokens=64,
            scale=1 / 255.0,
        )
        image = np.random.randint(0, 255, (16, 16, 3)).astype("float32")
        output = converter(image)
        grid_thw = output["grid_thw"]
        t, h, w = (int(v) for v in grid_thw)
        self.assertEqual(t, 1)
        num_patches = t * h * w
        patch_dim = 2 * 3 * 4 * 4  # patch_temporal * 3 * patch_size**2
        self.assertEqual(output["patches"].shape, (num_patches, patch_dim))

    def test_batched_input_raises(self):
        converter = MuseGlimmerImageConverter(
            patch_size=2, merge_size=1, max_image_tokens=16
        )
        with self.assertRaisesRegex(ValueError, "rank 3"):
            converter(np.ones((2, 8, 8, 3), dtype="float32"))

    def test_batched_input_raises_in_tf_function(self):
        if tf is None:
            self.skipTest("TensorFlow is not installed.")
        converter = MuseGlimmerImageConverter(
            patch_size=2, merge_size=1, max_image_tokens=16
        )
        with self.assertRaisesRegex(ValueError, "rank 3"):
            tf.data.Dataset.from_tensor_slices(
                np.ones((2, 2, 8, 8, 3), dtype="float32")
            ).map(converter)

    def test_smart_resize_matches_hf_float64_grids(self):
        # Each case is `(height, width, max_tokens)` with the HF target.
        # A float32 computation picks another grid for these sizes.
        cases = {
            (1472, 2783, 4096): (1288, 2464),
            (654, 5886, 4096): (588, 5376),
            (961, 144, 144): (840, 112),
        }
        for (height, width, max_tokens), expected in cases.items():
            self.assertEqual(
                _smart_resize(height, width, 14, 2, max_tokens),
                expected,
                msg=f"size {(height, width)}",
            )

    def test_channels_first_matches_channels_last(self):
        kwargs = {
            "patch_size": 2,
            "merge_size": 1,
            "max_image_tokens": 16,
            "interpolation": "nearest",
            "scale": [2 / 255.0] * 3,
            "offset": [-1.0] * 3,
        }
        image = np.random.uniform(0, 255, (8, 8, 3)).astype("float32")
        expected = MuseGlimmerImageConverter(**kwargs)(image)
        output = MuseGlimmerImageConverter(
            data_format="channels_first", **kwargs
        )(image.transpose(2, 0, 1))
        self.assertAllClose(output["patches"], expected["patches"])
        self.assertAllClose(output["grid_thw"], expected["grid_thw"])

    def test_channels_first_in_tf_data(self):
        if tf is None:
            self.skipTest("TensorFlow is not installed.")
        kwargs = {
            "patch_size": 2,
            "merge_size": 1,
            "max_image_tokens": 16,
            "interpolation": "nearest",
            "scale": [2 / 255.0] * 3,
            "offset": [-1.0] * 3,
        }
        image = np.random.uniform(0, 255, (8, 8, 3)).astype("float32")
        expected = MuseGlimmerImageConverter(**kwargs)(image)
        converter = MuseGlimmerImageConverter(
            data_format="channels_first", **kwargs
        )
        dataset = tf.data.Dataset.from_tensors(image.transpose(2, 0, 1)).map(
            converter
        )
        output = next(iter(dataset))
        self.assertAllClose(output["patches"], expected["patches"])
        self.assertAllClose(output["grid_thw"], expected["grid_thw"])

    def test_scalar_scale_with_channels_first(self):
        kwargs = {
            "patch_size": 2,
            "merge_size": 1,
            "max_image_tokens": 16,
            "interpolation": "nearest",
            "scale": 1 / 255.0,
        }
        image = np.random.uniform(0, 255, (8, 8, 3)).astype("float32")
        expected = MuseGlimmerImageConverter(**kwargs)(image)
        output = MuseGlimmerImageConverter(
            data_format="channels_first", **kwargs
        )(image.transpose(2, 0, 1))
        self.assertAllClose(output["patches"], expected["patches"])

    def test_global_channels_first_format(self):
        kwargs = {
            "patch_size": 2,
            "merge_size": 1,
            "max_image_tokens": 16,
            "interpolation": "bilinear",
            "scale": [2 / 255.0] * 3,
            "offset": [-1.0] * 3,
        }
        image = np.random.uniform(0, 255, (8, 12, 3)).astype("float32")
        expected = MuseGlimmerImageConverter(**kwargs)(image)
        previous_format = keras.config.image_data_format()
        keras.config.set_image_data_format("channels_first")
        try:
            output = MuseGlimmerImageConverter(**kwargs)(
                image.transpose(2, 0, 1)
            )
        finally:
            keras.config.set_image_data_format(previous_format)
        self.assertAllClose(output["patches"], expected["patches"], atol=1e-5)

    def test_serialization(self):
        converter = MuseGlimmerImageConverter(patch_size=14)
        self.run_serialization_test(converter)
