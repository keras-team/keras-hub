import numpy as np
import pytest

from keras_hub.src.models.sam3.sam3_detr_decoder import SAM3DetrDecoder
from keras_hub.src.models.sam3.sam3_detr_encoder import SAM3DetrEncoder
from keras_hub.src.models.sam3.sam3_geometry_encoder import SAM3GeometryEncoder
from keras_hub.src.models.sam3.sam3_image_converter import SAM3ImageConverter
from keras_hub.src.models.sam3.sam3_mask_decoder import SAM3MaskDecoder
from keras_hub.src.models.sam3.sam3_pc_backbone import (
    SAM3PromptableConceptBackbone,
)
from keras_hub.src.models.sam3.sam3_pc_image_segmenter import (
    SAM3PromptableConceptImageSegmenter,
)
from keras_hub.src.models.sam3.sam3_pc_image_segmenter_preprocessor import (
    SAM3PromptableConceptImageSegmenterPreprocessor,
)
from keras_hub.src.models.sam3.sam3_text_encoder import SAM3TextEncoder
from keras_hub.src.models.sam3.sam3_tokenizer import SAM3Tokenizer
from keras_hub.src.models.sam3.sam3_vision_encoder import SAM3VisionEncoder
from keras_hub.src.tests.test_case import TestCase


class SAM3PromptableConceptImageSegmenterTest(TestCase):
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
        self.backbone = SAM3PromptableConceptBackbone(
            vision_encoder=self.vision_encoder,
            text_encoder=self.text_encoder,
            geometry_encoder=self.geometry_encoder,
            detr_encoder=self.detr_encoder,
            detr_decoder=self.detr_decoder,
            mask_decoder=self.mask_decoder,
        )
        self.image_converter = SAM3ImageConverter(
            image_size=(self.image_size, self.image_size),
            scale=[1.0 / 255.0 / s for s in [0.5, 0.5, 0.5]],
            offset=[-m / s for m, s in zip([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])],
            crop_to_aspect_ratio=False,
            antialias=True,
        )
        self.tokenizer = SAM3Tokenizer(
            {
                "!": 0,
                '"': 1,
                "#": 2,
                "$": 3,
                "%": 4,
                "<|endoftext|>": 5,
                "<|startoftext|>": 6,
            },
            ["i n", "t h", "a n"],
        )
        self.preprocessor = SAM3PromptableConceptImageSegmenterPreprocessor(
            self.tokenizer, self.image_converter
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "preprocessor": self.preprocessor,
        }

        self.train_data = {
            "images": np.ones(
                (self.batch_size, self.image_size, self.image_size, 3),
                dtype="float32",
            ),
            "prompts": ["!"] * self.batch_size,
            "boxes": np.ones((self.batch_size, 1, 5), dtype="float32"),
            "box_labels": np.ones((self.batch_size, 1), dtype="int32"),
        }
        self.input_data = self.preprocessor(self.train_data)

    def test_sam3_pc_basics(self):
        pytest.skip(reason="TODO: enable after fit flow is figured out.")
        self.run_task_test(
            cls=SAM3PromptableConceptImageSegmenter,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=None,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=SAM3PromptableConceptImageSegmenter,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            atol=1e-4,  # Numerical discrepancies when running on the JAX GPU.
        )

    def test_end_to_end_model_predict(self):
        model = SAM3PromptableConceptImageSegmenter(**self.init_kwargs)
        outputs = model.predict(self.train_data)
        scores = outputs["scores"]
        boxes = outputs["boxes"]
        masks = outputs["masks"]

        output_size = self.image_size // self.vision_encoder.patch_size * 4
        num_queries = self.detr_decoder.num_queries
        self.assertAllEqual(scores.shape, (self.batch_size, num_queries))
        self.assertAllEqual(
            masks.shape,
            (self.batch_size, num_queries, output_size, output_size),
        )
        self.assertAllEqual(boxes.shape, (self.batch_size, num_queries, 4))

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in SAM3PromptableConceptImageSegmenter.presets:
            self.run_preset_test(
                cls=SAM3PromptableConceptImageSegmenter,
                preset=preset,
                input_data=self.input_data,
                expected_output_shape={
                    # TODO.
                    "scores": None,
                    "boxes": None,
                    "masks": None,
                },
            )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=SAM3PromptableConceptImageSegmenter,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            comparison_mode="statistical",
            output_thresholds={"*": {"max": 1e-2, "mean": 5e-3}},
        )
