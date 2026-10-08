import numpy as np
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.image_segmenter import ImageSegmenter
from keras_hub.src.models.sam3.sam3_pc_backbone import (
    SAM3PromptableConceptBackbone,
)
from keras_hub.src.models.sam3.sam3_pc_image_segmenter import (
    SAM3PromptableConceptImageSegmenter,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.transformers import convert_sam3


class SAM3ConverterTest(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = SAM3PromptableConceptImageSegmenter.from_preset(
            "hf://facebook/sam3",
        )
        images = np.random.rand(1, 32, 32, 3).astype("float32")
        outputs = model.predict(
            {
                "images": images,
                "prompts": ["cat"],
            }
        )
        # Verify output keys and shapes.
        self.assertIn("scores", outputs)
        self.assertIn("boxes", outputs)
        self.assertIn("masks", outputs)
        self.assertEqual(len(outputs["scores"].shape), 2)
        self.assertEqual(len(outputs["boxes"].shape), 3)
        self.assertEqual(outputs["boxes"].shape[-1], 4)

    @pytest.mark.extra_large
    def test_class_detection(self):
        preset_name = "hf://facebook/sam3"
        model = ImageSegmenter.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, SAM3PromptableConceptImageSegmenter)
        model = Backbone.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, SAM3PromptableConceptBackbone)

    def test_convert_backbone_config_rope_theta(self):
        # transformers < 5 format
        backbone_config = {
            "image_size": 32,
            "patch_size": 16,
            "num_hidden_layers": 2,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_attention_heads": 4,
            "pretrain_image_size": 32,
            "hidden_act": "gelu",
            "rope_theta": 10000.0,
            "window_size": 0,
            "global_attn_indexes": [],
            "attention_dropout": 0.0,
            "hidden_dropout": 0.0,
            "layer_norm_eps": 1e-6,
        }
        transformers_config = {
            "detector_config": {
                "vision_config": {
                    "backbone_config": backbone_config,
                    "fpn_hidden_size": 32,
                    "scale_factors": [1, 2],
                },
                "text_config": {
                    "vocab_size": 100,
                    "hidden_size": 32,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "intermediate_size": 48,
                    "hidden_act": "gelu",
                    "max_position_embeddings": 32,
                    "layer_norm_eps": 1e-6,
                },
                "geometry_encoder_config": {
                    "num_layers": 2,
                    "hidden_size": 32,
                    "intermediate_size": 48,
                    "num_attention_heads": 4,
                    "roi_size": 7,
                    "hidden_act": "gelu",
                    "hidden_dropout": 0.0,
                    "layer_norm_eps": 1e-6,
                },
                "detr_encoder_config": {
                    "num_layers": 2,
                    "hidden_size": 32,
                    "intermediate_size": 48,
                    "num_attention_heads": 4,
                    "hidden_act": "gelu",
                    "dropout": 0.0,
                    "layer_norm_eps": 1e-6,
                },
                "detr_decoder_config": {
                    "num_layers": 2,
                    "hidden_size": 32,
                    "intermediate_size": 48,
                    "num_attention_heads": 4,
                    "num_queries": 100,
                    "hidden_act": "gelu",
                    "dropout": 0.0,
                    "layer_norm_eps": 1e-6,
                },
                "mask_decoder_config": {
                    "num_upsampling_stages": 2,
                    "hidden_size": 32,
                    "num_attention_heads": 4,
                    "layer_norm_eps": 1e-6,
                },
            }
        }
        keras_config = convert_sam3.convert_backbone_config(
            transformers_config, cls=SAM3PromptableConceptBackbone
        )
        self.assertEqual(keras_config["vision_encoder"].rope_theta, 10000.0)

        # transformers >= 5 format
        backbone_config = {
            "image_size": 32,
            "patch_size": 16,
            "num_hidden_layers": 2,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_attention_heads": 4,
            "pretrain_image_size": 32,
            "hidden_act": "gelu",
            "rope_parameters": {"rope_theta": 20000.0},
            "window_size": 0,
            "global_attn_indexes": [],
            "attention_dropout": 0.0,
            "hidden_dropout": 0.0,
            "layer_norm_eps": 1e-6,
        }
        transformers_config["detector_config"]["vision_config"][
            "backbone_config"
        ] = backbone_config
        keras_config = convert_sam3.convert_backbone_config(
            transformers_config, cls=SAM3PromptableConceptBackbone
        )
        self.assertEqual(keras_config["vision_encoder"].rope_theta, 20000.0)
