import json
import os
import tempfile
from unittest import mock

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_backbone import (
    Qwen2_5VLBackbone,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_causal_lm import (
    Qwen2_5VLCausalLM,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_causal_lm_preprocessor import (
    Qwen2_5VLCausalLMPreprocessor,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_image_converter import (
    Qwen2_5VLImageConverter,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_tokenizer import (
    Qwen2_5VLTokenizer,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_video_converter import (
    Qwen2_5VLVideoConverter,
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
        "attention_dropout": 0.0,
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
        self.assertEqual(result["sliding_window_size"], 32768)

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

    def test_null_outer_rope_theta_loses_to_nested_value(self):
        """An explicit null outer `rope_theta` must not mask the nested one."""
        tc = _text_config()
        tc["rope_theta"] = None
        tc["rope_parameters"] = {
            "rope_theta": 5000000.0,
            "mrope_section": [16, 24, 24],
        }
        result = convert_qwen2_5_vl.convert_backbone_config({"text_config": tc})
        self.assertEqual(result["rope_max_wavelength"], 5000000.0)

    def test_sliding_window_null_is_omitted(self):
        """transformers 5.x writes `sliding_window: null` when unused."""
        tc = _text_config()
        tc["sliding_window"] = None
        result = convert_qwen2_5_vl.convert_backbone_config({"text_config": tc})
        self.assertNotIn("sliding_window_size", result)
        self.assertFalse(result["use_sliding_window_attention"])

    def test_missing_required_keys_fail_loudly(self):
        tc = _text_config()
        del tc["tie_word_embeddings"]
        with self.assertRaises(KeyError):
            convert_qwen2_5_vl.convert_backbone_config({"text_config": tc})

        tc = _text_config()
        del tc["attention_dropout"]
        with self.assertRaises(KeyError):
            convert_qwen2_5_vl.convert_backbone_config({"text_config": tc})

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
        """Stand-in for `SafetensorLoader` with fixed HF-layout sources."""

        def __init__(self):
            self.sources = self._make_sources()
            self.ported = {}

        @staticmethod
        def _make_sources():
            # Correct HF layouts for `_tiny_backbone`: linear weights are
            # `(out, in)`, and patch embed is `(E, C, T, kH, kW)`.
            shapes = {
                "model.embed_tokens.weight": (32, 16),
                "lm_head.weight": (32, 16),
                "model.norm.weight": (16,),
                "visual.patch_embed.proj.weight": (8, 3, 2, 2, 2),
                "visual.merger.ln_q.weight": (8,),
                "visual.merger.mlp.0.weight": (8, 8),
                "visual.merger.mlp.0.bias": (8,),
                "visual.merger.mlp.2.weight": (16, 8),
                "visual.merger.mlp.2.bias": (16,),
            }
            for i in range(1):
                p = f"model.layers.{i}"
                shapes.update(
                    {
                        f"{p}.input_layernorm.weight": (16,),
                        f"{p}.post_attention_layernorm.weight": (16,),
                        f"{p}.mlp.gate_proj.weight": (8, 16),
                        f"{p}.mlp.up_proj.weight": (8, 16),
                        f"{p}.mlp.down_proj.weight": (16, 8),
                        f"{p}.self_attn.o_proj.weight": (16, 16),
                    }
                )
                for proj in ("q", "k", "v"):
                    shapes[f"{p}.self_attn.{proj}_proj.weight"] = (16, 16)
                    shapes[f"{p}.self_attn.{proj}_proj.bias"] = (16,)
            for i in range(1):
                p = f"visual.blocks.{i}"
                shapes.update(
                    {
                        f"{p}.norm1.weight": (8,),
                        f"{p}.norm2.weight": (8,),
                        f"{p}.attn.qkv.weight": (24, 8),
                        f"{p}.attn.qkv.bias": (24,),
                        f"{p}.attn.proj.weight": (8, 8),
                        f"{p}.attn.proj.bias": (8,),
                        f"{p}.mlp.gate_proj.weight": (16, 8),
                        f"{p}.mlp.gate_proj.bias": (16,),
                        f"{p}.mlp.up_proj.weight": (16, 8),
                        f"{p}.mlp.up_proj.bias": (16,),
                        f"{p}.mlp.down_proj.weight": (8, 16),
                        f"{p}.mlp.down_proj.bias": (8,),
                    }
                )

            sources = {}
            offset = 0
            for key, shape in shapes.items():
                size = int(np.prod(shape))
                values = offset + np.arange(size, dtype="float32")
                sources[key] = values.reshape(shape) / 1000.0
                offset += size
            return sources

        def port_weight(self, keras_variable, hf_weight_key, hook_fn=None):
            source = self.sources[hf_weight_key]
            self.ported[hf_weight_key] = source
            if hook_fn is not None:
                source = hook_fn(source, [int(d) for d in keras_variable.shape])
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
                intermediate_dim=16,
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
        vision = backbone.vision_encoder
        merger = vision.merger
        self.assertAllClose(
            ops.convert_to_numpy(merger.mlp_fc1.kernel),
            loader.ported["visual.merger.mlp.0.weight"].T,
        )
        # RMSNorm ports raw (no hook).
        self.assertAllClose(
            ops.convert_to_numpy(merger.ln_q.scale),
            loader.ported["visual.merger.ln_q.weight"],
        )
        self.assertAllClose(
            ops.convert_to_numpy(vision.patch_embed.kernel),
            np.transpose(
                loader.ported["visual.patch_embed.proj.weight"],
                (2, 3, 4, 1, 0),
            ),
        )

        block = vision.blocks[0]
        for proj in ("gate", "up", "down"):
            key = f"visual.blocks.0.mlp.{proj}_proj.weight"
            kernel = getattr(block.mlp, f"{proj}_proj").kernel
            self.assertAllClose(
                ops.convert_to_numpy(kernel), loader.ported[key].T
            )
        self.assertAllClose(
            ops.convert_to_numpy(block.attn.qkv.kernel),
            loader.ported["visual.blocks.0.attn.qkv.weight"].T,
        )

        decoder = backbone.get_layer("transformer_layer_0")
        self.assertAllClose(
            ops.convert_to_numpy(decoder._feedforward_gate_dense.kernel),
            loader.ported["model.layers.0.mlp.gate_proj.weight"].T,
        )
        self.assertAllClose(
            ops.convert_to_numpy(
                decoder._feedforward_intermediate_dense.kernel
            ),
            loader.ported["model.layers.0.mlp.up_proj.weight"].T,
        )
        self.assertAllClose(
            ops.convert_to_numpy(decoder._feedforward_output_dense.kernel),
            loader.ported["model.layers.0.mlp.down_proj.weight"].T,
        )

    def test_convert_weights_tied_embeddings_skips_lm_head(self):
        backbone = self._tiny_backbone()
        backbone.tie_word_embeddings = True
        loader = self._RecordingLoader()
        convert_qwen2_5_vl.convert_weights(backbone, loader, {})
        self.assertNotIn("lm_head.weight", loader.ported)
        self.assertIn("model.embed_tokens.weight", loader.ported)


class TransformersPresetEndToEndTest(TestCase):
    def _write_tokenizer_files(self, preset_dir):
        tokenizer_json = {
            "model": {
                "vocab": {"a": 0, "b": 1, "ab": 2},
                "merges": ["a b"],
            },
            "added_tokens": [
                {"id": 3, "content": "<|endoftext|>"},
                {"id": 4, "content": "<|im_start|>"},
                {"id": 5, "content": "<|im_end|>"},
                {"id": 6, "content": "<|vision_start|>"},
                {"id": 7, "content": "<|vision_end|>"},
            ],
        }
        tokenizer_config = {
            "added_tokens_decoder": {
                "8": {"content": "<|image_pad|>"},
                "9": {"content": "<|video_pad|>"},
            },
        }
        for filename, contents in (
            ("tokenizer.json", tokenizer_json),
            ("tokenizer_config.json", tokenizer_config),
        ):
            with open(os.path.join(preset_dir, filename), "w") as f:
                json.dump(contents, f)

    def _write_preprocessor_config(self, preset_dir):
        contents = {
            "patch_size": 2,
            "temporal_patch_size": 2,
            "merge_size": 1,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "size": {"shortest_edge": 16, "longest_edge": 100},
        }
        with open(
            os.path.join(preset_dir, "preprocessor_config.json"), "w"
        ) as f:
            json.dump(contents, f)

    def _write_tiny_hf_preset(self, preset_dir, tie_word_embeddings):
        transformers = pytest.importorskip("transformers")
        torch = pytest.importorskip("torch")
        if not hasattr(transformers, "Qwen2_5_VLTextConfig"):
            pytest.skip("test requires nested Qwen2.5-VL transformers configs")

        text_config = transformers.Qwen2_5_VLTextConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            rms_norm_eps=1e-6,
            use_sliding_window=False,
            sliding_window=None,
            attention_dropout=0.0,
            rope_parameters={
                "rope_theta": 1000000.0,
                "mrope_section": [2, 1, 1],
            },
            bos_token_id=None,
            eos_token_id=None,
            pad_token_id=None,
        )
        vision_config = transformers.Qwen2_5_VLVisionConfig(
            depth=1,
            hidden_size=8,
            intermediate_size=16,
            num_heads=2,
            in_channels=3,
            patch_size=2,
            spatial_merge_size=1,
            temporal_patch_size=2,
            tokens_per_second=4,
            window_size=4,
            out_hidden_size=16,
            fullatt_block_indexes=[0],
        )
        config = transformers.Qwen2_5_VLConfig(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=8,
            video_token_id=9,
            vision_start_token_id=6,
            vision_end_token_id=7,
            tie_word_embeddings=tie_word_embeddings,
        )
        torch.manual_seed(0)
        hf_model = transformers.Qwen2_5_VLForConditionalGeneration(config)
        hf_model.save_pretrained(preset_dir)
        self._write_tokenizer_files(preset_dir)
        self._write_preprocessor_config(preset_dir)

    def test_tiny_hf_preset_loads_end_to_end(self):
        """A local HF checkpoint round-trips through the real from_preset
        machinery for both tied and untied embedding configs."""
        safetensors_numpy = pytest.importorskip("safetensors.numpy")
        for tied in (False, True):
            with self.subTest(tie_word_embeddings=tied):
                with tempfile.TemporaryDirectory() as preset_dir:
                    self._write_tiny_hf_preset(preset_dir, tied)

                    backbone = Qwen2_5VLBackbone.from_preset(preset_dir)
                    self.assertIsInstance(backbone, Qwen2_5VLBackbone)
                    self.assertIsNotNone(backbone.vision_encoder)
                    self.assertEqual(backbone.tie_word_embeddings, tied)

                    weights_path = os.path.join(preset_dir, "model.safetensors")
                    hf_weights = safetensors_numpy.load_file(weights_path)
                    self.assertAllClose(
                        ops.convert_to_numpy(
                            backbone.token_embedding.embeddings
                        ),
                        hf_weights["model.embed_tokens.weight"],
                    )
                    if not tied:
                        self.assertAllClose(
                            ops.convert_to_numpy(
                                backbone.token_embedding.reverse_embeddings
                            ),
                            hf_weights["lm_head.weight"].T,
                        )

                    causal_lm = Qwen2_5VLCausalLM.from_preset(preset_dir)
                    self.assertIsInstance(causal_lm, Qwen2_5VLCausalLM)
                    self.assertIsInstance(causal_lm.backbone, Qwen2_5VLBackbone)
                    preprocessor = causal_lm.preprocessor
                    self.assertIsInstance(
                        preprocessor, Qwen2_5VLCausalLMPreprocessor
                    )
                    self.assertIsInstance(
                        preprocessor.image_converter, Qwen2_5VLImageConverter
                    )
                    self.assertIsInstance(
                        preprocessor.video_converter, Qwen2_5VLVideoConverter
                    )
                    self.assertIs(
                        preprocessor.video_converter.image_converter,
                        preprocessor.image_converter,
                    )
                    self.assertEqual(preprocessor.tokens_per_second, 4)

                    inputs = preprocessor.generate_preprocess(
                        ["ab"], sequence_length=8
                    )
                    logits = ops.convert_to_numpy(causal_lm(inputs))
                    self.assertEqual(logits.shape, (1, 8, 32))
                    self.assertTrue(np.isfinite(logits).all())


class PresetRoutingTest(TestCase):
    def test_qwen2_5_vl_model_type(self):
        loader = TransformersPresetLoader("x", {"model_type": "qwen2_5_vl"})
        self.assertIs(loader.converter, convert_qwen2_5_vl)
