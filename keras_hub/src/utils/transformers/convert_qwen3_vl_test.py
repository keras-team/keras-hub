import json
import os

import numpy as np
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.qwen3_vl.qwen3_vl_backbone import Qwen3VLBackbone
from keras_hub.src.models.qwen3_vl.qwen3_vl_causal_lm import Qwen3VLCausalLM
from keras_hub.src.models.qwen3_vl.qwen3_vl_causal_lm_preprocessor import (
    _mrope_positions,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_tokenizer import Qwen3VLTokenizer
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import running_on_gpu
from keras_hub.src.utils.transformers import convert_qwen3_vl

try:
    import torch
    from safetensors import safe_open
    from transformers import Qwen3VLConfig
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    torch = None

requires_torch = pytest.mark.skipif(
    torch is None, reason="Requires torch, safetensors and transformers."
)


def _tiny_hf_config(tie_word_embeddings):
    return Qwen3VLConfig(
        text_config=dict(
            vocab_size=300,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            rope_parameters=dict(
                rope_type="default",
                rope_theta=10000.0,
                mrope_section=[4, 2, 2],
                mrope_interleaved=True,
            ),
        ),
        vision_config=dict(
            depth=4,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=64,
            num_position_embeddings=64,
            deepstack_visual_indexes=[1, 2],
        ),
        image_token_id=250,
        video_token_id=251,
        vision_start_token_id=252,
        vision_end_token_id=253,
        tie_word_embeddings=tie_word_embeddings,
    )


class TestQwen3VLConverter(TestCase):
    def _save_tiny(self, tie_word_embeddings, randomize=False):
        torch.manual_seed(0)
        model = Qwen3VLForConditionalGeneration(
            _tiny_hf_config(tie_word_embeddings)
        )
        if randomize:
            # Give every parameter, including norms and biases, a nonzero
            # value.
            with torch.no_grad():
                for param in model.parameters():
                    param.copy_(torch.randn_like(param) * 0.2)
        path = os.path.join(self.get_temp_dir(), f"tiny_{tie_word_embeddings}")
        model.save_pretrained(path)
        return path

    @pytest.mark.large
    def test_class_detection(self):
        model = Backbone.from_preset(
            "hf://yujiepan/qwen3-vl-tiny-random",
            load_weights=False,
        )
        self.assertIsInstance(model, Qwen3VLBackbone)
        model = CausalLM.from_preset(
            "hf://yujiepan/qwen3-vl-tiny-random",
            load_weights=False,
        )
        self.assertIsInstance(model, Qwen3VLCausalLM)

    @requires_torch
    def test_key_mapping_covers_hf_checkpoint(self):
        for tie in (True, False):
            path = self._save_tiny(tie)
            with safe_open(
                os.path.join(path, "model.safetensors"), framework="np"
            ) as f:
                hf_keys = set(f.keys())
            expected = convert_qwen3_vl.expected_hf_keys(
                num_layers=4,
                vision_depth=4,
                num_deepstack=2,
                tie_word_embeddings=tie,
            )
            self.assertEqual(hf_keys, expected)

    @requires_torch
    def test_backbone_config(self):
        path = self._save_tiny(False)
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        keras_config = convert_qwen3_vl.convert_backbone_config(config)
        self.assertEqual(keras_config["rope_max_wavelength"], 10000.0)
        self.assertEqual(keras_config["mrope_section"], [4, 2, 2])
        self.assertFalse(keras_config["tie_word_embeddings"])
        encoder = keras_config["vision_encoder"]
        self.assertEqual(list(encoder.deepstack_visual_indexes), [1, 2])
        config["text_config"]["attention_dropout"] = 0.1
        keras_config = convert_qwen3_vl.convert_text_config(config)
        self.assertEqual(keras_config["dropout"], 0.1)
        config["text_config"]["rope_parameters"]["mrope_interleaved"] = False
        with self.assertRaisesRegex(ValueError, "interleaved"):
            convert_qwen3_vl.convert_text_config(config)

    def test_converter_config_reads_rescale_and_resample(self):
        vision_config = {
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
        }
        processor_config = {
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "rescale_factor": 1.0 / 127.5,
            "resample": 2,
            "size": {"shortest_edge": 64, "longest_edge": 1024},
        }
        config = convert_qwen3_vl._converter_config(
            processor_config, vision_config
        )
        self.assertEqual(config["interpolation"], "bilinear")
        self.assertAllClose(config["scale"], [1.0 / 63.75] * 3)
        processor_config["resample"] = 5
        with self.assertRaisesRegex(ValueError, "resample"):
            convert_qwen3_vl._converter_config(processor_config, vision_config)

    def _write_json(self, name, content):
        path = os.path.join(self.get_temp_dir(), "preset")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, name), "w") as f:
            json.dump(content, f)
        return path

    @requires_torch
    def test_load_video_converter_config(self):
        config = _tiny_hf_config(False).to_dict()
        path = self._write_json(
            "video_preprocessor_config.json",
            {
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "size": {"shortest_edge": 4096, "longest_edge": 25165824},
                "patch_size": 16,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "fps": 4,
                "min_frames": 2,
                "max_frames": None,
            },
        )
        video = convert_qwen3_vl.load_video_converter_config(path, config)
        self.assertEqual(video["min_pixels"], 4096)
        self.assertEqual(video["max_pixels"], 25165824)
        self.assertEqual(video["fps"], 4)
        self.assertEqual(video["min_frames"], 2)
        # `None` keeps the converter default.
        self.assertNotIn("max_frames", video)
        self.assertEqual(video["interpolation"], "bicubic")
        self.assertAllClose(video["scale"], [2.0 / 255.0] * 3)

    def test_convert_tokenizer(self):
        vocab = {"Ġ": 0, "a": 1, "Ġa": 2}
        special = [
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "<|video_pad|>",
        ]
        path = self._write_json(
            "tokenizer.json",
            {
                "model": {"vocab": vocab, "merges": [["Ġ", "a"]]},
                "added_tokens": [
                    {"id": 10 + i, "content": token}
                    for i, token in enumerate(special)
                ],
            },
        )
        tokenizer = convert_qwen3_vl.convert_tokenizer(Qwen3VLTokenizer, path)
        self.assertIsInstance(tokenizer, Qwen3VLTokenizer)
        self.assertEqual(tokenizer.image_token_id, 15)
        self.assertEqual(tokenizer.video_token_id, 16)
        self.assertEqual(tokenizer.end_token_id, 12)
        self.assertAllEqual(tokenizer(" a<|image_pad|> a"), [2, 15, 2])

    @requires_torch
    def test_unsupported_configs_raise(self):
        def config_with(section, key, value):
            config = _tiny_hf_config(False).to_dict()
            config[section][key] = value
            return config

        for config, message in [
            (config_with("vision_config", "hidden_act", "gelu"), "hidden_act"),
            (config_with("text_config", "hidden_act", "gelu"), "hidden_act"),
            (config_with("text_config", "attention_bias", True), "bias"),
            (
                config_with("vision_config", "out_hidden_size", 32),
                "out_hidden_size",
            ),
        ]:
            with self.assertRaisesRegex(ValueError, message):
                convert_qwen3_vl.convert_backbone_config(config)
        config = _tiny_hf_config(False).to_dict()
        config["text_config"]["rope_parameters"]["rope_type"] = "yarn"
        with self.assertRaisesRegex(ValueError, "rope_type"):
            convert_qwen3_vl.convert_backbone_config(config)

    @requires_torch
    def test_rope_theta_legacy_format(self):
        path = self._save_tiny(True)
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        text = config["text_config"]
        text["rope_scaling"] = {
            "rope_type": "default",
            "mrope_section": [4, 2, 2],
        }
        text["rope_theta"] = 5000000.0
        del text["rope_parameters"]
        keras_config = convert_qwen3_vl.convert_backbone_config(config)
        self.assertEqual(keras_config["rope_max_wavelength"], 5000000.0)

    @requires_torch
    def test_convert_tiny_preset_dtype(self):
        path = self._save_tiny(False)
        # The vision tower runs in float32 unless a dtype is given.
        model = Backbone.from_preset(path, load_weights=False)
        self.assertEqual(model.vision_encoder.dtype_policy.name, "float32")
        model = Backbone.from_preset(path, dtype="bfloat16")
        self.assertEqual(model.dtype_policy.name, "bfloat16")
        self.assertEqual(model.vision_encoder.dtype_policy.name, "bfloat16")
        self.assertEqual(
            model.vision_encoder.get_layer("blocks_0").compute_dtype,
            "bfloat16",
        )

    @requires_torch
    def test_convert_tiny_preset(self):
        for tie in (True, False):
            path = self._save_tiny(tie)
            model = Backbone.from_preset(path)
            self.assertIsInstance(model, Qwen3VLBackbone)
            self.assertEqual(model.tie_word_embeddings, tie)
            hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
                path
            ).eval()
            token_ids = np.random.RandomState(0).randint(0, 250, (2, 7))
            with torch.no_grad():
                expected = hf_model.model(
                    input_ids=torch.from_numpy(token_ids)
                ).last_hidden_state.numpy()
            actual = model(
                {
                    "token_ids": token_ids.astype("int32"),
                    "padding_mask": np.ones((2, 7), "int32"),
                }
            )
            atol = 5e-3 if running_on_gpu() else 1e-4
            self.assertAllClose(actual, expected, atol=atol, rtol=0)

    @requires_torch
    def test_causal_lm_logits_match_hf_with_image(self):
        # 16 patches of 16x16 pixels over 2 frames, merged into 4 tokens.
        grid = [1, 4, 4]
        rng = np.random.RandomState(0)
        pixels = rng.rand(1, 16, 2, 16, 16, 3).astype("float32")
        # HF flattens each patch as (channels, time, rows, cols).
        hf_pixels = pixels[0].transpose(0, 4, 1, 2, 3).reshape(16, -1)
        token_ids = np.array([[5, 6, 252, 250, 250, 250, 250, 253, 7, 8]])
        token_types = [1 if t == 250 else 0 for t in token_ids[0]]
        position_ids = np.array(_mrope_positions(token_types, [grid], [], 2))
        for tie in (True, False):
            path = self._save_tiny(tie, randomize=True)
            hf_model = Qwen3VLForConditionalGeneration.from_pretrained(
                path
            ).eval()
            with torch.no_grad():
                expected = hf_model(
                    input_ids=torch.from_numpy(token_ids),
                    pixel_values=torch.from_numpy(hf_pixels),
                    image_grid_thw=torch.tensor([grid]),
                    mm_token_type_ids=torch.tensor([token_types]),
                ).logits.numpy()
            causal_lm = Qwen3VLCausalLM(
                backbone=Backbone.from_preset(path), preprocessor=None
            )
            actual = causal_lm(
                {
                    "token_ids": token_ids.astype("int32"),
                    "padding_mask": np.ones_like(token_ids, "int32"),
                    "position_ids": position_ids[None].astype("int32"),
                    "pixel_values": pixels,
                    "image_grid_thw": np.array([[grid]], "int32"),
                    "vision_indices": np.array([[3, 4, 5, 6]], "int32"),
                }
            )
            atol = 5e-3 if running_on_gpu() else 1e-4
            self.assertAllClose(actual, expected, atol=atol, rtol=0)

    @requires_torch
    def test_convert_tiny_preset_without_vision_encoder(self):
        # A full checkpoint loads as a text-only backbone on request.
        path = self._save_tiny(False)
        model = Backbone.from_preset(path, vision_encoder=None)
        self.assertIsNone(model.vision_encoder)
        hf_model = Qwen3VLForConditionalGeneration.from_pretrained(path).eval()
        token_ids = np.random.RandomState(0).randint(0, 250, (2, 7))
        with torch.no_grad():
            expected = hf_model.model(
                input_ids=torch.from_numpy(token_ids)
            ).last_hidden_state.numpy()
        actual = model(
            {
                "token_ids": token_ids.astype("int32"),
                "padding_mask": np.ones((2, 7), "int32"),
            }
        )
        atol = 5e-3 if running_on_gpu() else 1e-4
        self.assertAllClose(actual, expected, atol=atol, rtol=0)
