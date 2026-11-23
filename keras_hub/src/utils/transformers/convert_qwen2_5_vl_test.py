from unittest import mock

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_qwen2_5_vl


def _text_config():
    return {
        "vocab_size": 151936,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "hidden_size": 2048,
        "intermediate_size": 11008,
        "rope_theta": 1000000,
        "rms_norm_eps": 1e-6,
        "use_sliding_window": False,
        "sliding_window": 32768,
        "tie_word_embeddings": True,
        "rope_parameters": {"mrope_section": [16, 24, 24]},
    }


class ConvertBackboneConfigTest(TestCase):
    def test_full_vision_config(self):
        """7B-style configs spell out every vision key."""
        hf_config = {
            "text_config": _text_config(),
            "vision_config": {
                "depth": 32,
                "hidden_size": 1280,
                "intermediate_size": 3420,
                "num_heads": 16,
                "in_chans": 3,
                "out_hidden_size": 2048,
                "patch_size": 14,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "window_size": 112,
                "fullatt_block_indexes": [7, 15, 23, 31],
                "tokens_per_second": 4,
            },
        }
        result = convert_qwen2_5_vl.convert_backbone_config(hf_config)
        vision = result["vision_encoder"]
        self.assertEqual(vision.embed_dim, 1280)
        self.assertEqual(vision.out_dim, 2048)
        self.assertEqual(vision.intermediate_dim, 3420)
        self.assertEqual(vision.depth, 32)
        self.assertEqual(result["hidden_dim"], 2048)
        self.assertEqual(result["rope_max_wavelength"], 1000000)
        self.assertEqual(result["mrope_section"], [16, 24, 24])
        self.assertTrue(result["tie_word_embeddings"])

    def test_sparse_vision_config_uses_defaults(self):
        """32B omits keys that match `Qwen2_5_VLVisionConfig` defaults."""
        hf_config = {
            "text_config": _text_config(),
            "vision_config": {
                "depth": 64,
                "hidden_size": 1280,
                "intermediate_size": 3456,
                "num_heads": 16,
                "out_hidden_size": 3584,
            },
        }
        result = convert_qwen2_5_vl.convert_backbone_config(hf_config)
        vision = result["vision_encoder"]
        self.assertEqual(vision.patch_size, 14)
        self.assertEqual(vision.temporal_patch_size, 2)
        self.assertEqual(vision.spatial_merge_size, 2)
        self.assertEqual(vision.window_size, 112)
        self.assertEqual(list(vision.fullatt_block_indexes), [7, 15, 23, 31])
        self.assertEqual(vision.intermediate_dim, 3456)

    def test_text_only_config(self):
        result = convert_qwen2_5_vl.convert_backbone_config(
            {"text_config": _text_config()}
        )
        self.assertNotIn("vision_encoder", result)


class LoadImageConverterConfigTest(TestCase):
    def _preprocessor_config(self, size_nested):
        config = {
            "patch_size": 14,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "image_mean": [0.5] * 3,
            "image_std": [0.5] * 3,
        }
        if size_nested:
            config["size"] = {
                "shortest_edge": 3136,
                "longest_edge": 12845056,
            }
        else:
            config["min_pixels"] = 3136
            config["max_pixels"] = 12845056
        return config

    def test_size_nesting(self):
        """transformers 5.x nests pixel budgets under ``size``."""
        with mock.patch.object(
            convert_qwen2_5_vl,
            "load_json",
            return_value=self._preprocessor_config(size_nested=True),
        ):
            kwargs = convert_qwen2_5_vl.load_image_converter_config(
                "x", {"vision_config": {}}
            )
        self.assertEqual(kwargs["min_pixels"], 3136)
        self.assertEqual(kwargs["max_pixels"], 12845056)
        self.assertEqual(kwargs["merge_size"], 2)

    def test_flat_pixels(self):
        """transformers 4.x ships flat ``min_pixels``/``max_pixels``."""
        with mock.patch.object(
            convert_qwen2_5_vl,
            "load_json",
            return_value=self._preprocessor_config(size_nested=False),
        ):
            kwargs = convert_qwen2_5_vl.load_image_converter_config(
                "x", {"vision_config": {}}
            )
        self.assertEqual(kwargs["min_pixels"], 3136)

    def test_text_only_returns_none(self):
        self.assertIsNone(
            convert_qwen2_5_vl.load_image_converter_config("x", {})
        )


class LoadPreprocessorConfigTest(TestCase):
    def test_tokens_per_second(self):
        kwargs = convert_qwen2_5_vl.load_preprocessor_config(
            "x", {"vision_config": {"tokens_per_second": 2}}
        )
        self.assertEqual(kwargs, {"tokens_per_second": 2})

    def test_text_only_returns_none(self):
        self.assertIsNone(convert_qwen2_5_vl.load_preprocessor_config("x", {}))
