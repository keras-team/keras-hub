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
            image_mean=127.5,
            image_std=127.5,
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
            image_mean=0.0,
            image_std=255.0,
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
            "image_mean": [127.5] * 3,
            "image_std": [127.5] * 3,
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
            "image_mean": [127.5] * 3,
            "image_std": [127.5] * 3,
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
            "image_mean": 0.0,
            "image_std": 255.0,
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
            "image_mean": [127.5] * 3,
            "image_std": [127.5] * 3,
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

    def test_matches_hf_reference_pixel_values(self):
        # The reference cases come from `MuseGlimmerImageProcessor` in
        # `transformers` 5.19.0, run on CPU with torchvision 0.28.0. HF falls
        # back to bicubic on CUDA or on torchvision older than 0.27. The
        # processor used `patch_size=2`, `temporal_patch_size=2`,
        # `merge_size=2`, `max_image_tokens=6`, mean 0.5, and std 0.5. The
        # input is a crop of `test_data/test_image.jpg`, given as `(top,
        # left, height, width)`. Each case stores the first temporal slot
        # of each patch as uint8 levels. The pixel value is
        # `(level - 127.5) / 127.5` in float32. HF repeats that slot in the
        # second temporal slot.
        reference_cases = [
            (
                (0, 0, 557, 707),
                [1, 4, 6],
                (
                    "201 202 200 200 189 189 188 188 130 131 128 131 203 205 "
                    "202 203 191 192 190 190 133 135 133 132 204 206 209 202 "
                    "191 194 197 191 132 136 136 132 212 216 177 145 198 202 "
                    "167 135 140 143 121 98 209 204 173 201 196 193 162 191 "
                    "137 132 110 129 204 204 203 203 192 192 192 191 132 132 "
                    "130 130 201 200 201 199 188 188 189 187 131 132 134 132 "
                    "201 211 212 173 187 199 198 160 131 137 141 112 163 107 "
                    "79 81 151 95 66 70 105 69 45 49 107 108 89 128 "
                    "103 109 86 126 90 106 70 108 190 208 206 211 181 196 "
                    "194 196 135 135 150 144 206 206 209 208 194 193 194 193 "
                    "136 137 145 143 198 202 198 198 183 186 182 182 135 136 "
                    "134 134 195 80 148 144 180 68 135 128 133 44 96 86 "
                    "67 77 171 127 52 64 155 114 28 41 107 80 77 155 "
                    "154 201 69 136 140 184 51 91 100 132 209 201 200 199 "
                    "191 184 184 183 138 135 132 130 200 197 198 197 182 179 "
                    "182 181 133 130 129 127 206 202 201 204 189 186 183 186 "
                    "146 143 138 141 202 219 202 206 185 202 185 189 140 152 "
                    "138 139 160 116 137 87 152 111 130 83 106 74 85 47 "
                    "141 204 133 200 134 190 127 186 94 138 86 136 205 205 "
                    "201 202 188 189 185 187 137 141 134 135 206 206 203 202 "
                    "190 189 187 187 141 138 135 131"
                ),
            ),
            (
                (50, 200, 200, 150),
                [1, 6, 4],
                (
                    "204 205 204 204 192 192 192 191 134 134 134 133 206 205 "
                    "206 204 191 191 191 192 132 132 132 132 205 204 204 204 "
                    "192 193 192 190 133 134 132 132 205 206 204 204 192 192 "
                    "192 192 133 133 132 132 202 204 203 205 191 190 191 191 "
                    "133 133 134 133 205 203 206 204 190 192 190 192 133 132 "
                    "132 132 204 204 203 202 192 190 193 190 132 132 132 131 "
                    "203 203 202 203 191 191 190 191 131 131 130 131 204 205 "
                    "205 204 190 191 190 193 133 133 133 133 205 205 204 203 "
                    "192 193 194 191 132 133 133 132 202 205 209 198 192 193 "
                    "199 188 132 132 136 130 208 201 151 129 197 190 141 117 "
                    "135 131 104 91 204 204 204 203 193 194 194 192 133 133 "
                    "133 131 204 210 211 168 191 198 199 156 132 136 137 111 "
                    "177 120 100 97 166 108 87 85 118 82 63 62 105 111 "
                    "92 97 94 100 79 84 72 79 51 57 203 209 205 204 "
                    "192 198 193 193 132 136 132 134 183 104 127 94 172 91 "
                    "115 82 120 67 80 54 93 92 92 92 80 79 80 78 "
                    "52 50 55 53 99 102 82 76 86 92 68 68 62 71 "
                    "46 46 212 160 197 107 200 149 185 94 138 108 133 66 "
                    "89 89 75 86 76 76 63 73 49 48 34 48 96 94 "
                    "85 88 81 80 70 77 59 59 51 56 82 97 75 72 "
                    "69 88 65 61 47 66 42 41"
                ),
            ),
        ]
        # XLA on CPU replaces `x / 127.5` with `x * (1 / 127.5)`. That
        # changes the last float32 bit. TensorFlow and PyTorch divide
        # exactly like HF.
        atol = 1e-6 if keras.config.backend() == "jax" else 0
        photo = self.load_test_image()
        converter = MuseGlimmerImageConverter(
            patch_size=2,
            patch_temporal=2,
            merge_size=2,
            max_image_tokens=6,
            interpolation="lanczos3",
            antialias=True,
            image_mean=[127.5] * 3,
            image_std=[127.5] * 3,
        )
        for (top, left, height, width), grid_thw, levels in reference_cases:
            outputs = converter(photo[top : top + height, left : left + width])
            slot = np.array(levels.split(), dtype="float32")
            slot = (slot - np.float32(127.5)) / np.float32(127.5)
            slot = slot.reshape((grid_thw[1] * grid_thw[2], -1))
            self.assertAllEqual(outputs["grid_thw"], grid_thw)
            self.assertAllClose(
                outputs["patches"],
                np.concatenate([slot, slot], axis=-1),
                atol=atol,
                rtol=0,
            )

    def test_scale_and_offset_are_rejected_and_not_saved(self):
        with self.assertRaisesRegex(ValueError, "image_mean"):
            MuseGlimmerImageConverter(scale=1 / 255.0)
        with self.assertRaisesRegex(ValueError, "image_mean"):
            MuseGlimmerImageConverter(offset=-1.0)
        config = MuseGlimmerImageConverter(
            image_mean=[127.5] * 3, image_std=[127.5] * 3
        ).get_config()
        self.assertNotIn("scale", config)
        self.assertNotIn("offset", config)
        self.assertNotIn("crop_to_aspect_ratio", config)

    def test_image_mean_and_std_must_be_set_together(self):
        with self.assertRaisesRegex(ValueError, "together"):
            MuseGlimmerImageConverter(image_mean=[127.5] * 3)
        with self.assertRaisesRegex(ValueError, "together"):
            MuseGlimmerImageConverter(image_std=[127.5] * 3)

    def test_serialization(self):
        converter = MuseGlimmerImageConverter(patch_size=14)
        self.run_serialization_test(converter)
