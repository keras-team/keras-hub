"""Tests for TIPSv2 backbone."""

import numpy as np
from keras import ops

from keras_hub.src.models.tipsv2.tipsv2_backbone import TIPSv2Backbone
from keras_hub.src.models.tipsv2.tipsv2_text_encoder import TIPSv2TextEncoder
from keras_hub.src.models.tipsv2.tipsv2_vision_encoder import (
    TIPSv2VisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class TIPSv2BackboneTest(TestCase):
    def setUp(self):
        self.image_size = 28
        self.patch_size = 14
        self.hidden_dim = 32
        self.num_patches = (self.image_size // self.patch_size) ** 2  # 4

        self.vision_encoder = TIPSv2VisionEncoder(
            patch_size=self.patch_size,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            num_heads=4,
            mlp_ratio=2.0,
            init_values=1.0,
            num_register_tokens=1,
            ffn_layer="mlp",
            image_shape=(self.image_size, self.image_size, 3),
        )
        self.text_encoder = TIPSv2TextEncoder(
            vocabulary_size=100,
            embedding_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            num_heads=4,
            intermediate_dim=64,
            max_sequence_length=16,
        )
        self.init_kwargs = {
            "vision_encoder": self.vision_encoder,
            "text_encoder": self.text_encoder,
            "temperature": 0.01,
        }
        self.input_data = {
            "images": np.random.rand(
                2, self.image_size, self.image_size, 3
            ).astype("float32"),
            "token_ids": np.array(
                [[1, 2, 3, 4, 0, 0], [5, 6, 7, 0, 0, 0]], dtype="int32"
            ),
            "padding_mask": np.array(
                [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]], dtype="int32"
            ),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=TIPSv2Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "vision_cls_embedding": (2, self.hidden_dim),
                "vision_patch_embeddings": (
                    2,
                    self.num_patches,
                    self.hidden_dim,
                ),
                "text_embedding": (2, self.hidden_dim),
            },
        )

    def test_output_values(self):
        backbone = TIPSv2Backbone(**self.init_kwargs)
        outputs = backbone(self.input_data)

        # Check output shapes.
        self.assertEqual(
            outputs["vision_cls_embedding"].shape, (2, self.hidden_dim)
        )
        self.assertEqual(
            outputs["vision_patch_embeddings"].shape,
            (2, self.num_patches, self.hidden_dim),
        )
        self.assertEqual(outputs["text_embedding"].shape, (2, self.hidden_dim))

        # Outputs should be finite.
        self.assertTrue(
            np.all(
                np.isfinite(
                    ops.convert_to_numpy(outputs["vision_cls_embedding"])
                )
            )
        )
        self.assertTrue(
            np.all(np.isfinite(ops.convert_to_numpy(outputs["text_embedding"])))
        )

    def test_swiglu_variant(self):
        """Test with SwiGLU FFN (used by g/14 variant)."""
        vision_encoder = TIPSv2VisionEncoder(
            patch_size=self.patch_size,
            hidden_dim=self.hidden_dim,
            num_layers=2,
            num_heads=4,
            mlp_ratio=2.0,
            init_values=1.0,
            num_register_tokens=1,
            ffn_layer="swiglu",
            image_shape=(self.image_size, self.image_size, 3),
        )
        backbone = TIPSv2Backbone(
            vision_encoder=vision_encoder,
            text_encoder=self.text_encoder,
        )
        outputs = backbone(self.input_data)
        self.assertEqual(
            outputs["vision_cls_embedding"].shape, (2, self.hidden_dim)
        )
