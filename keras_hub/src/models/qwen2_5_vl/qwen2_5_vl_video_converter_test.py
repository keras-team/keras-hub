import numpy as np

from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_image_converter import (
    Qwen2_5VLImageConverter,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_video_converter import (
    Qwen2_5VLVideoConverter,
)
from keras_hub.src.models.qwen2_vl import qwen2_vl_video_converter_test


class Qwen2_5VLVideoConverterTest(
    qwen2_vl_video_converter_test.Qwen2VLVideoConverterTest,
):
    """Qwen2.5-VL runs the shared video-converter suite against
    ``Qwen2_5VLVideoConverter``/``Qwen2_5VLImageConverter``, plus the
    delegation and serialization paths specific to this subclass."""

    video_converter_cls = Qwen2_5VLVideoConverter
    image_converter_cls = Qwen2_5VLImageConverter

    def test_default_builds_2_5_image_converter(self):
        converter = Qwen2_5VLVideoConverter()
        self.assertIsInstance(
            converter.image_converter, Qwen2_5VLImageConverter
        )

    def test_dynamic_bound_delegation(self):
        """`None` bounds delegate live to `image_converter`, so mutating
        it updates video bounds without touching the converter."""
        image_converter = Qwen2_5VLImageConverter()
        converter = Qwen2_5VLVideoConverter(image_converter=image_converter)
        self.assertEqual(converter.max_pixels, image_converter.max_pixels)
        image_converter.max_pixels = 56 * 56
        self.assertEqual(converter.max_pixels, 56 * 56)

    def test_none_bounds_roundtrip(self):
        converter = Qwen2_5VLVideoConverter()
        revived = Qwen2_5VLVideoConverter.from_config(converter.get_config())
        self.assertIsNone(revived._min_pixels)
        self.assertIsNone(revived._max_pixels)
        # Bounds still delegate to the deserialized image converter.
        self.assertEqual(revived.min_pixels, 3136)
        self.assertEqual(revived.max_pixels, 12845056)

    def test_cap_pixels_per_frame_roundtrip(self):
        converter = Qwen2_5VLVideoConverter(
            cap_pixels_per_frame=True, max_video_tokens=64
        )
        self.run_serialization_test(converter)
        revived = Qwen2_5VLVideoConverter.from_config(converter.get_config())
        self.assertTrue(revived.cap_pixels_per_frame)
        self.assertEqual(revived.max_video_tokens, 64)

    def test_token_budget_floor(self):
        """The ~`min_pixels` floor keeps frames from shrinking below the
        image bounds even when the token budget is tiny."""
        rng = np.random.default_rng(0)
        video = rng.integers(0, 256, size=(4, 200, 200, 3)).astype("float32")
        converter = Qwen2_5VLVideoConverter(
            cap_pixels_per_frame=True, max_video_tokens=1
        )
        # Floor at int(3136 * 1.05) = 3292 px → 56x56 frames.
        self.assertAllEqual(converter(video)["grid_thw"], [[2, 4, 4]])
