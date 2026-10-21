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


class ConvertBackboneConfigTest(TestCase):
    def test_2_2b_config_resolution(self):
        """The `.get` defaults resolve a 2.2B-shaped HF config."""
        hf_config = {
            "text_config": {
                "hidden_size": 2048,
                "head_dim": 64,
                "vocab_size": 49152,
                "intermediate_size": 8192,
                "num_hidden_layers": 1,
                "rope_theta": 10000,
                "rms_norm_eps": 1e-5,
            },
            "vision_config": {"image_size": 384, "patch_size": 14},
            "scale_factor": 2,
            "image_token_id": 49152,
            "tie_word_embeddings": False,
        }
        resolved = convert_smolvlm2.convert_backbone_config(hf_config)
        self.assertEqual(resolved["num_query_heads"], 32)
        self.assertEqual(resolved["num_key_value_heads"], 32)
        self.assertEqual(resolved["vision_hidden_dim"], 1152)
        self.assertEqual(resolved["vision_num_heads"], 16)
