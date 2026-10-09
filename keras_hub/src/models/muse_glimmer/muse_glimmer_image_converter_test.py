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

        self.assertAllClose(normalized, np.round(normalized))

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

    def test_serialization(self):
        converter = MuseGlimmerImageConverter(patch_size=14)
        self.run_serialization_test(converter)
