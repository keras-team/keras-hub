from unittest import mock

import numpy as np
from keras import ops

from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_backbone import (
    Qwen2_5VLBackbone,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_image_converter import (
    Qwen2_5VLImageConverter,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_tokenizer import (
    Qwen2_5VLTokenizer,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_qwen2_5_vl
from keras_hub.src.utils.transformers import convert_qwen2_vl
from keras_hub.src.utils.transformers.preset_loader import (
    TransformersPresetLoader,
)


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

    def test_rope_parameters_null(self):
        """transformers 5.x may serialize `rope_parameters` as null."""
        tc = _text_config()
        tc["rope_parameters"] = None
        tc["rope_scaling"] = {"mrope_section": [16, 24, 24]}
        result = convert_qwen2_5_vl.convert_backbone_config({"text_config": tc})
        self.assertEqual(result["mrope_section"], [16, 24, 24])
        self.assertEqual(result["rope_max_wavelength"], 1000000)

    def test_nested_rope_theta(self):
        """transformers 5.x may nest `rope_theta` in `rope_parameters`."""
        tc = _text_config()
        del tc["rope_theta"]
        tc["rope_parameters"] = {
            "rope_theta": 5000000.0,
            "mrope_section": [16, 24, 24],
        }
        result = convert_qwen2_5_vl.convert_backbone_config({"text_config": tc})
        self.assertEqual(result["rope_max_wavelength"], 5000000.0)

    def test_sliding_window_null_keeps_default(self):
        """transformers 5.x writes `sliding_window: null` when unused."""
        tc = _text_config()
        tc["sliding_window"] = None
        result = convert_qwen2_5_vl.convert_backbone_config({"text_config": tc})
        self.assertEqual(result["sliding_window_size"], 32768)
        self.assertFalse(result["use_sliding_window_attention"])

    def test_attention_dropout_forwarded(self):
        tc = _text_config()
        tc["attention_dropout"] = 0.1
        result = convert_qwen2_5_vl.convert_backbone_config({"text_config": tc})
        self.assertEqual(result["dropout"], 0.1)


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
            convert_qwen2_vl,
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
            convert_qwen2_vl,
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


class LoadVideoConverterConfigTest(TestCase):
    def _preprocessor_config(self):
        return {
            "patch_size": 14,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "image_mean": [0.5] * 3,
            "image_std": [0.5] * 3,
            "min_pixels": 3136,
            "max_pixels": 12845056,
        }

    def test_returns_2_5_image_converter(self):
        """Qwen2.5-VL ships no video config; bounds come from the image
        preprocessor, wrapped in a ``Qwen2_5VLImageConverter``."""
        with mock.patch.object(
            convert_qwen2_vl,
            "load_json",
            return_value=self._preprocessor_config(),
        ):
            kwargs = convert_qwen2_5_vl.load_video_converter_config(
                "x", {"vision_config": {}}
            )
        converter = kwargs["image_converter"]
        self.assertIsInstance(converter, Qwen2_5VLImageConverter)
        self.assertEqual(converter.min_pixels, 3136)
        self.assertEqual(converter.max_pixels, 12845056)

    def test_text_only_returns_none(self):
        self.assertIsNone(
            convert_qwen2_5_vl.load_video_converter_config("x", {})
        )


class ConvertTokenizerTest(TestCase):
    def _tokenizer_files(self, merges):
        tokenizer_json = {
            "model": {
                "vocab": {"a": 0, "b": 1, "ab": 2},
                "merges": merges,
            },
            "added_tokens": [
                {"id": 10, "content": "<|endoftext|>"},
                {"id": 15, "content": "<|im_start|>"},
                {"id": 16, "content": "<|im_end|>"},
                {"id": 17, "content": "<|vision_start|>"},
                {"id": 18, "content": "<|vision_end|>"},
                {"id": 19, "content": "<|reserved_special_token_0|>"},
            ],
        }
        tokenizer_config = {
            "added_tokens_decoder": {
                "20": {"content": "<|image_pad|>"},
                "21": {"content": "<|video_pad|>"},
                "22": {"content": "<|reserved_special_token_1|>"},
            },
        }
        return {
            "tokenizer.json": tokenizer_json,
            "tokenizer_config.json": tokenizer_config,
        }

    def _convert(self, merges):
        files = self._tokenizer_files(merges)
        with mock.patch.object(
            convert_qwen2_vl,
            "load_json",
            side_effect=lambda preset, fname: files[fname],
        ):
            return convert_qwen2_5_vl.convert_tokenizer(Qwen2_5VLTokenizer, "x")

    def test_string_merges(self):
        tokenizer = self._convert(["a b"])
        self.assertEqual(tokenizer.token_to_id("<|image_pad|>"), 20)
        self.assertEqual(tokenizer.token_to_id("<|video_pad|>"), 21)

    def test_list_merges(self):
        """32B `tokenizer.json` stores merges as two-element lists."""
        tokenizer = self._convert([["a", "b"]])
        self.assertEqual(tokenizer.token_to_id("ab"), 2)

    def test_special_tokens_sorted_and_reserved_skipped(self):
        tokenizer = self._convert(["a b"])
        special = tokenizer.unsplittable_tokens
        self.assertEqual(special, sorted(special))
        for token in ("<|vision_start|>", "<|image_pad|>", "<|video_pad|>"):
            self.assertIn(token, special)
        for token in special:
            self.assertFalse(token.startswith("<|reserved_special_token_"))


class ConvertWeightsTest(TestCase):
    class _RecordingLoader:
        """Stand-in for `SafetensorLoader` with synthesized sources."""

        def __init__(self):
            self.ported = {}

        def port_weight(self, keras_variable, hf_weight_key, hook_fn=None):
            shape = [int(d) for d in keras_variable.shape]
            size = int(np.prod(shape))
            flat = np.arange(size, dtype="float32")
            name = getattr(hook_fn, "__name__", "")
            if name == "<lambda>":
                # Patch embed: HF (E, C, T, kH, kW) -> (T, kH, kW, C, E).
                source = np.transpose(flat.reshape(shape), (4, 3, 0, 1, 2))
            elif name == "transpose_and_reshape":
                # HF stores 2-D (out, in); transposed then reshaped.
                source = flat.reshape((size // shape[0], shape[0]))
            elif hook_fn is convert_qwen2_vl._transpose_2d:
                source = flat.reshape(shape[::-1])
            else:
                source = flat.reshape(shape)
            self.ported[hf_weight_key] = source
            if hook_fn is not None:
                source = hook_fn(source, shape)
            keras_variable.assign(source)

    def _tiny_backbone(self):
        from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_vision_encoder import (
            Qwen2_5VLVisionEncoder,
        )

        return Qwen2_5VLBackbone(
            vision_encoder=Qwen2_5VLVisionEncoder(
                patch_size=2,
                temporal_patch_size=2,
                in_channels=3,
                embed_dim=8,
                out_dim=16,
                depth=1,
                num_heads=2,
                intermediate_dim=8,
                spatial_merge_size=1,
                window_size=4,
                fullatt_block_indexes=[0],
            ),
            vocabulary_size=32,
            num_layers=1,
            num_query_heads=2,
            num_key_value_heads=2,
            hidden_dim=16,
            intermediate_dim=8,
            # head_dim//2 = 4; sections must sum to it.
            mrope_section=[2, 1, 1],
            tie_word_embeddings=False,
        )

    def test_convert_weights_ports_all_keys(self):
        backbone = self._tiny_backbone()
        loader = self._RecordingLoader()
        convert_qwen2_5_vl.convert_weights(backbone, loader, {})

        expected = {
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
            "visual.patch_embed.proj.weight",
            "visual.merger.ln_q.weight",
            "visual.merger.mlp.0.weight",
            "visual.merger.mlp.0.bias",
            "visual.merger.mlp.2.weight",
            "visual.merger.mlp.2.bias",
        }
        for i in range(1):
            p = f"model.layers.{i}"
            expected |= {
                f"{p}.input_layernorm.weight",
                f"{p}.post_attention_layernorm.weight",
                f"{p}.mlp.gate_proj.weight",
                f"{p}.mlp.up_proj.weight",
                f"{p}.mlp.down_proj.weight",
                f"{p}.self_attn.o_proj.weight",
            }
            for proj in ("q", "k", "v"):
                expected |= {
                    f"{p}.self_attn.{proj}_proj.weight",
                    f"{p}.self_attn.{proj}_proj.bias",
                }
        for i in range(1):
            p = f"visual.blocks.{i}"
            expected |= {
                f"{p}.norm1.weight",
                f"{p}.norm2.weight",
                f"{p}.attn.qkv.weight",
                f"{p}.attn.qkv.bias",
                f"{p}.attn.proj.weight",
                f"{p}.attn.proj.bias",
            }
            for proj in ("gate", "up", "down"):
                expected |= {
                    f"{p}.mlp.{proj}_proj.weight",
                    f"{p}.mlp.{proj}_proj.bias",
                }
        self.assertEqual(set(loader.ported), expected)

    def test_convert_weights_transpose_values(self):
        backbone = self._tiny_backbone()
        loader = self._RecordingLoader()
        convert_qwen2_5_vl.convert_weights(backbone, loader, {})
        merger = backbone.vision_encoder.merger
        self.assertAllClose(
            ops.convert_to_numpy(merger.mlp_fc1.kernel),
            loader.ported["visual.merger.mlp.0.weight"].T,
        )
        # RMSNorm ports raw (no hook).
        self.assertAllClose(
            ops.convert_to_numpy(merger.ln_q.scale),
            loader.ported["visual.merger.ln_q.weight"],
        )

    def test_convert_weights_tied_embeddings_skips_lm_head(self):
        backbone = self._tiny_backbone()
        backbone.tie_word_embeddings = True
        loader = self._RecordingLoader()
        convert_qwen2_5_vl.convert_weights(backbone, loader, {})
        self.assertNotIn("lm_head.weight", loader.ported)
        self.assertIn("model.embed_tokens.weight", loader.ported)


class PresetRoutingTest(TestCase):
    def test_qwen2_5_vl_model_type(self):
        loader = TransformersPresetLoader("x", {"model_type": "qwen2_5_vl"})
        self.assertIs(loader.converter, convert_qwen2_5_vl)
