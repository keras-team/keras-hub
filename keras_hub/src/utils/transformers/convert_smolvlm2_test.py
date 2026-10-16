from unittest import mock

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_smolvlm2


class ResampleToInterpolationTest(TestCase):
    def test_hf_resample_codes(self):
        mapping = convert_smolvlm2._resample_to_interpolation
        self.assertEqual(mapping(1), "lanczos3")
        self.assertEqual(mapping(2), "bilinear")
        self.assertEqual(mapping(3), "bicubic")

    def test_unknown_resample_raises(self):
        with self.assertRaisesRegex(ValueError, "resample"):
            convert_smolvlm2._resample_to_interpolation(0)

    def test_missing_resample_defaults_to_lanczos(self):
        config = {
            "image_mean": [0.5] * 3,
            "image_std": [0.5] * 3,
            "rescale_factor": 1 / 255,
            "max_image_size": {"longest_edge": 512},
        }
        with mock.patch.object(
            convert_smolvlm2, "_load_preprocessor_config", return_value=config
        ):
            shared = convert_smolvlm2._load_vision_normalization_config("x")
        self.assertEqual(shared["interpolation"], "lanczos3")
