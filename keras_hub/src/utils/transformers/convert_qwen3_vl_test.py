import json
import os

import torch
from safetensors import safe_open
from transformers import Qwen3VLConfig
from transformers import Qwen3VLForConditionalGeneration

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.qwen3_vl.qwen3_vl_backbone import Qwen3_VLBackbone
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_qwen3_vl


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


class TestQwen3_VLConverter(TestCase):
    def _save_tiny(self, tie_word_embeddings):
        torch.manual_seed(0)
        model = Qwen3VLForConditionalGeneration(
            _tiny_hf_config(tie_word_embeddings)
        )
        path = os.path.join(self.get_temp_dir(), f"tiny_{tie_word_embeddings}")
        model.save_pretrained(path)
        return path

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

    def test_convert_tiny_preset(self):
        for tie in (True, False):
            path = self._save_tiny(tie)
            model = Backbone.from_preset(path)
            self.assertIsInstance(model, Qwen3_VLBackbone)
            self.assertEqual(model.tie_word_embeddings, tie)
