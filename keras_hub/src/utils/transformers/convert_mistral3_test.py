import json
import os
import tempfile

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.mistral3.mistral3_backbone import Mistral3Backbone
from keras_hub.src.models.mistral3.mistral3_vision_encoder import (
    Mistral3MultiModalProjector,
)
from keras_hub.src.models.mistral3.mistral3_vision_encoder import (
    Mistral3VisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_mistral3


class TestTask(TestCase):
    @pytest.mark.large
    def test_convert_multimodal_preset_matches_hf(self):
        # Build a tiny Mistral3 (Pixtral vision tower + Mistral text model)
        # checkpoint and check that the converted `Mistral3Backbone` matches
        # HF's reference forward pass end to end, including the vision
        # tower, multimodal projector, and image/text embedding merge.
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")

        text_config = transformers.MistralConfig(
            vocab_size=100,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            sliding_window=None,
            rope_theta=1_000_000.0,
            rms_norm_eps=1e-5,
        )
        vision_config = transformers.PixtralVisionConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_channels=3,
            image_size=16,
            patch_size=4,
            rope_parameters={"rope_theta": 10000.0},
        )
        config = transformers.Mistral3Config(
            text_config=text_config,
            vision_config=vision_config,
            image_token_index=10,
            spatial_merge_size=2,
        )
        torch.manual_seed(0)
        hf_model = transformers.Mistral3ForConditionalGeneration(config).eval()

        with tempfile.TemporaryDirectory() as preset_dir:
            hf_model.save_pretrained(preset_dir)
            keras_backbone = Mistral3Backbone.from_preset(preset_dir)

        self.assertIsNotNone(keras_backbone.vision_encoder)

        # A single 16x16 image with a 4x4 patch size produces a 4x4 patch
        # grid (16 patches); a spatial merge size of 2 merges these into a
        # 2x2 grid of 4 tokens, so 4 placeholder tokens at id 10 are needed.
        input_ids = np.array([[1, 10, 10, 10, 10, 3, 4]], dtype="int32")
        padding_mask = np.ones_like(input_ids)
        pixel_values = np.random.rand(1, 3, 16, 16).astype("float32")
        image_sizes = np.array([[16, 16]], dtype="int32")
        placeholder_indices = np.array([[1, 2, 3, 4]], dtype="int32")

        keras_out = ops.convert_to_numpy(
            keras_backbone(
                {
                    "token_ids": input_ids,
                    "padding_mask": padding_mask,
                    "pixel_values": pixel_values,
                    "image_sizes": image_sizes,
                    "placeholder_indices": placeholder_indices,
                }
            )
        )
        with torch.no_grad():
            hf_out = (
                hf_model.model(
                    input_ids=torch.tensor(input_ids),
                    attention_mask=torch.tensor(padding_mask),
                    pixel_values=torch.tensor(pixel_values),
                    image_sizes=torch.tensor(image_sizes),
                )
                .last_hidden_state.detach()
                .cpu()
                .numpy()
            )
        self.assertEqual(keras_out.shape, hf_out.shape)
        # fp16 weight storage dominates the parity bound, as in the
        # text-only converter test.
        self.assertAllClose(keras_out, hf_out, atol=1e-2)

    def test_convert_backbone_config_detects_mistral3(self):
        transformers_config = {
            "text_config": {
                "vocab_size": 100,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "hidden_size": 32,
                "intermediate_size": 48,
                "num_key_value_heads": 2,
                "rope_theta": 1_000_000.0,
                "rms_norm_eps": 1e-5,
                "sliding_window": None,
            },
            "vision_config": {
                "hidden_size": 16,
                "intermediate_size": 24,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_channels": 3,
                "image_size": 32,
                "patch_size": 8,
                "hidden_act": "gelu",
                "attention_dropout": 0.0,
                "rope_parameters": {"rope_theta": 10000.0},
            },
            "image_token_index": 10,
            "spatial_merge_size": 2,
            "projector_hidden_act": "gelu",
            "multimodal_projector_bias": False,
        }
        keras_config = convert_mistral3.convert_backbone_config(
            transformers_config
        )
        self.assertIsInstance(
            keras_config["vision_encoder"], Mistral3VisionEncoder
        )
        self.assertIsInstance(
            keras_config["multimodal_projector"],
            Mistral3MultiModalProjector,
        )
        self.assertEqual(keras_config["image_token_index"], 10)
        self.assertEqual(keras_config["rope_max_wavelength"], 1_000_000.0)
        self.assertEqual(
            keras_config["vision_encoder"].get_config()["rope_theta"],
            10000.0,
        )

    def test_load_image_converter_config_without_preprocessor_config(self):
        # Some checkpoints (e.g. Mistral Small 3.2) ship no
        # `preprocessor_config.json`; the image normalization mean/std
        # should come from `mistral_common`, not a local hardcoded copy.
        pytest.importorskip("mistral_common")
        from mistral_common.tokens.tokenizers.image import DATASET_MEAN
        from mistral_common.tokens.tokenizers.image import DATASET_STD

        transformers_config = {
            "vision_config": {"patch_size": 14, "image_size": 1540},
            "spatial_merge_size": 2,
        }
        with tempfile.TemporaryDirectory() as dir_path:
            with open(os.path.join(dir_path, "config.json"), "w") as f:
                json.dump(transformers_config, f)
            config = convert_mistral3.load_image_converter_config(
                dir_path, transformers_config
            )
        expected_offset = [-m / s for m, s in zip(DATASET_MEAN, DATASET_STD)]
        expected_scale = [(1 / 255) / s for s in DATASET_STD]
        self.assertAllClose(config["offset"], expected_offset)
        self.assertAllClose(config["scale"], expected_scale)
        self.assertEqual(config["patch_size"], 14)
        self.assertEqual(config["longest_edge"], 1540)
        self.assertEqual(config["spatial_merge_size"], 2)
