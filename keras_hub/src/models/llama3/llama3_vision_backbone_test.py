import numpy as np
import pytest

from keras_hub.src.models.llama3.llama3_vision_backbone import (
    Llama3VisionBackbone,
)
from keras_hub.src.tests.test_case import TestCase


class Llama3VisionBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 100,
            "num_layers": 4,
            "hidden_dim": 16,
            "num_query_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_dim": 32,
            "vision_hidden_dim": 16,
            "vision_num_layers": 2,
            "vision_num_heads": 2,
            "vision_intermediate_dim": 32,
            "vision_patch_size": 4,
            "vision_image_size": 16,
            "cross_attention_layers": [1, 3],
        }
        batch_size = 2
        seq_len = 10
        self.input_data = {
            "pixel_values": np.random.uniform(
                size=(batch_size, 16, 16, 3)
            ).astype("float32"),
            "token_ids": np.ones((batch_size, seq_len), dtype="int32"),
            "padding_mask": np.ones((batch_size, seq_len), dtype="int32"),
        }

    def test_backbone_basics(self):
        """Test basic backbone functionality and serialization."""
        backbone = Llama3VisionBackbone(**self.init_kwargs)
        outputs = backbone(self.input_data)
        self.assertEqual(outputs.shape, (2, 10, 16))

        # Test from_config
        config = backbone.get_config()
        restored = Llama3VisionBackbone.from_config(config)
        restored_outputs = restored(self.input_data)
        self.assertEqual(restored_outputs.shape, (2, 10, 16))

    def test_cross_attention_blocks_created(self):
        """Verify cross-attention blocks are created at specified layers."""
        backbone = Llama3VisionBackbone(**self.init_kwargs)
        self.assertIn(1, backbone.cross_attention_blocks)
        self.assertIn(3, backbone.cross_attention_blocks)
        self.assertNotIn(0, backbone.cross_attention_blocks)
        self.assertNotIn(2, backbone.cross_attention_blocks)

    def test_no_cross_attention(self):
        """Test backbone with empty cross-attention layers."""
        init_kwargs = dict(self.init_kwargs)
        init_kwargs["cross_attention_layers"] = []
        backbone = Llama3VisionBackbone(**init_kwargs)
        outputs = backbone(self.input_data)
        self.assertEqual(outputs.shape, (2, 10, 16))

    def test_two_stage_encoder(self):
        """Test backbone with two-stage vision encoder."""
        init_kwargs = dict(self.init_kwargs)
        init_kwargs["vision_local_layers"] = 3
        init_kwargs["vision_global_layers"] = 1
        init_kwargs["vision_num_layers"] = 4
        backbone = Llama3VisionBackbone(**init_kwargs)
        outputs = backbone(self.input_data)
        self.assertEqual(outputs.shape, (2, 10, 16))

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Llama3VisionBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_freeze_vision_encoder(self):
        """Test freezing the vision encoder."""
        backbone = Llama3VisionBackbone(**self.init_kwargs)
        self.assertTrue(backbone.vision_encoder.trainable)
        backbone.freeze_vision_encoder()
        self.assertFalse(backbone.vision_encoder.trainable)

    def test_freeze_text_backbone(self):
        """Test freezing the text backbone."""
        backbone = Llama3VisionBackbone(**self.init_kwargs)
        self.assertTrue(backbone.text_backbone.trainable)
        backbone.freeze_text_backbone()
        self.assertFalse(backbone.text_backbone.trainable)

    def test_freeze_cross_attention(self):
        """Test freezing cross-attention blocks."""
        backbone = Llama3VisionBackbone(**self.init_kwargs)
        for ca_block in backbone.cross_attention_blocks.values():
            self.assertTrue(ca_block.trainable)
        backbone.freeze_cross_attention()
        for ca_block in backbone.cross_attention_blocks.values():
            self.assertFalse(ca_block.trainable)

    def test_unfreeze_all(self):
        """Test unfreezing all components."""
        backbone = Llama3VisionBackbone(**self.init_kwargs)
        backbone.freeze_vision_encoder()
        backbone.freeze_text_backbone()
        backbone.freeze_cross_attention()
        backbone.unfreeze_all()
        self.assertTrue(backbone.vision_encoder.trainable)
        self.assertTrue(backbone.text_backbone.trainable)
        for ca_block in backbone.cross_attention_blocks.values():
            self.assertTrue(ca_block.trainable)
