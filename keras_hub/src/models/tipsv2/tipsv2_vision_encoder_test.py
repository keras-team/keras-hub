"""Tests for TIPSv2 Vision Encoder."""

import numpy as np
import pytest
from keras import ops

from keras_hub.src.models.tipsv2.tipsv2_vision_encoder import (
    TIPSv2VisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class TIPSv2VisionEncoderTest(TestCase):
    def setUp(self):
        self.image_size = 28
        self.patch_size = 14
        self.hidden_dim = 32
        self.num_patches = (self.image_size // self.patch_size) ** 2  # 4
        self.num_register_tokens = 1

        self.init_kwargs = {
            "patch_size": self.patch_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "init_values": 1.0,
            "num_register_tokens": self.num_register_tokens,
            "ffn_layer": "mlp",
            "image_shape": (self.image_size, self.image_size, 3),
        }
        self.input_data = {
            "images": ops.ones((2, self.image_size, self.image_size, 3)),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=TIPSv2VisionEncoder,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "cls_token": (2, 1, self.hidden_dim),
                "register_tokens": (
                    2,
                    self.num_register_tokens,
                    self.hidden_dim,
                ),
                "patch_tokens": (2, self.num_patches, self.hidden_dim),
            },
        )

    def test_output_values(self):
        encoder = TIPSv2VisionEncoder(**self.init_kwargs)
        outputs = encoder(self.input_data)

        # Check output shapes.
        self.assertEqual(outputs["cls_token"].shape, (2, 1, self.hidden_dim))
        self.assertEqual(
            outputs["register_tokens"].shape,
            (2, self.num_register_tokens, self.hidden_dim),
        )
        self.assertEqual(
            outputs["patch_tokens"].shape,
            (2, self.num_patches, self.hidden_dim),
        )

        # All outputs should be finite.
        for key in ["cls_token", "register_tokens", "patch_tokens"]:
            self.assertTrue(
                np.all(np.isfinite(ops.convert_to_numpy(outputs[key]))),
                f"{key} has non-finite values",
            )

    def test_swiglu_ffn(self):
        """Test with SwiGLU FFN layer (used by g/14 variant)."""
        init_kwargs = {**self.init_kwargs, "ffn_layer": "swiglu"}
        encoder = TIPSv2VisionEncoder(**init_kwargs)
        outputs = encoder(self.input_data)
        self.assertEqual(outputs["cls_token"].shape, (2, 1, self.hidden_dim))

    def test_no_register_tokens(self):
        """Test with num_register_tokens=0."""
        init_kwargs = {**self.init_kwargs, "num_register_tokens": 0}
        encoder = TIPSv2VisionEncoder(**init_kwargs)
        outputs = encoder(self.input_data)
        self.assertEqual(outputs["cls_token"].shape, (2, 1, self.hidden_dim))
        self.assertEqual(
            outputs["patch_tokens"].shape,
            (2, self.num_patches, self.hidden_dim),
        )

    def test_get_config_roundtrip(self):
        encoder = TIPSv2VisionEncoder(**self.init_kwargs)
        config = encoder.get_config()

        self.assertEqual(config["patch_size"], self.patch_size)
        self.assertEqual(config["hidden_dim"], self.hidden_dim)
        self.assertEqual(config["num_layers"], 2)
        self.assertEqual(config["num_heads"], 4)
        self.assertEqual(config["mlp_ratio"], 2.0)
        self.assertEqual(config["init_values"], 1.0)
        self.assertEqual(
            config["num_register_tokens"], self.num_register_tokens
        )
        self.assertEqual(config["ffn_layer"], "mlp")
        self.assertEqual(
            config["image_shape"],
            (self.image_size, self.image_size, 3),
        )

        # Roundtrip.
        restored = TIPSv2VisionEncoder.from_config(config)
        outputs = restored(self.input_data)
        self.assertEqual(outputs["cls_token"].shape, (2, 1, self.hidden_dim))

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=TIPSv2VisionEncoder,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.skipTest("Presets are not uploaded yet.")
        self.run_preset_test(
            cls=TIPSv2VisionEncoder,
            preset="tipsv2_b14",
            input_data={
                "images": ops.ones((1, 448, 448, 3)),
            },
            expected_output_shape={
                "cls_token": (1, 1, 768),
                "register_tokens": (1, 1, 768),
                "patch_tokens": (1, 1024, 768),
            },
        )
