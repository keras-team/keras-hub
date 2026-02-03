import numpy as np

from keras_hub.src.models.sam3.sam3_detr_decoder import SAM3DetrDecoder
from keras_hub.src.models.sam3.sam3_detr_encoder import SAM3DetrEncoder
from keras_hub.src.models.sam3.sam3_geometry_encoder import SAM3GeometryEncoder
from keras_hub.src.models.sam3.sam3_mask_decoder import SAM3MaskDecoder
from keras_hub.src.models.sam3.sam3_pc_backbone import (
    SAM3PromptableConceptBackbone,
)
from keras_hub.src.models.sam3.sam3_text_encoder import SAM3TextEncoder
from keras_hub.src.models.sam3.sam3_vision_encoder import SAM3VisionEncoder
from keras_hub.src.tests.test_case import TestCase


class SAM3PromptableConceptBackboneTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.image_size = 28
        self.vision_encoder = SAM3VisionEncoder(
            image_shape=(self.image_size, self.image_size, 3),
            patch_size=14,
            num_layers=2,
            hidden_dim=16,
            intermediate_dim=32,
            num_heads=2,
            fpn_hidden_dim=16,
            fpn_scale_factors=[4.0, 2.0, 1.0, 0.5],
            pretrain_image_shape=(42, 42, 3),
            window_size=2,
            global_attn_indexes=[1, 2],
        )
        self.text_encoder = SAM3TextEncoder(
            vocabulary_size=32,
            embedding_dim=16,
            hidden_dim=16,
            num_layers=2,
            num_heads=2,
            intermediate_dim=32,
        )
        self.geometry_encoder = SAM3GeometryEncoder(
            num_layers=2,
            hidden_dim=16,
            intermediate_dim=32,
            num_heads=2,
            roi_size=7,
        )
        self.detr_encoder = SAM3DetrEncoder(
            num_layers=2,
            hidden_dim=16,
            intermediate_dim=32,
            num_heads=2,
        )
        self.detr_decoder = SAM3DetrDecoder(
            image_shape=(self.image_size, self.image_size, 3),
            patch_size=14,
            num_layers=2,
            hidden_dim=16,
            intermediate_dim=32,
            num_heads=2,
            num_queries=8,
        )
        self.mask_decoder = SAM3MaskDecoder(
            num_upsampling_stages=3,
            hidden_dim=16,
            num_heads=2,
        )
        self.init_kwargs = {
            "vision_encoder": self.vision_encoder,
            "text_encoder": self.text_encoder,
            "geometry_encoder": self.geometry_encoder,
            "detr_encoder": self.detr_encoder,
            "detr_decoder": self.detr_decoder,
            "mask_decoder": self.mask_decoder,
        }

        self.input_data = {
            "pixel_values": np.ones(
                (self.batch_size, self.image_size, self.image_size, 3),
                dtype="float32",
            ),
            "token_ids": np.ones((self.batch_size, 32), dtype="int32"),
            "padding_mask": np.ones((self.batch_size, 32), dtype="bool"),
            "boxes": np.ones((self.batch_size, 1, 5), dtype="float32"),
            "box_labels": np.ones((self.batch_size, 1), dtype="int32"),
        }

    def test_backbone_basics(self):
        output_size = self.image_size // self.vision_encoder.patch_size * 4
        num_queries = self.detr_decoder.num_queries
        self.run_backbone_test(
            cls=SAM3PromptableConceptBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "pred_masks": (
                    self.batch_size,
                    output_size,
                    output_size,
                    num_queries,
                ),
                "pred_boxes": (self.batch_size, num_queries, 4),
                "pred_logits": (self.batch_size, num_queries),
                "presence_logits": (self.batch_size, 1),
                "semantic_segs": (self.batch_size, output_size, output_size, 1),
            },
        )
