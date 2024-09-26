import numpy as np
import pytest

from keras_hub.src.models.sam.sam_backbone import SAMBackbone
from keras_hub.src.models.sam.sam_image_segmenter import SAMImageSegmenter
from keras_hub.src.models.sam.sam_mask_decoder import SAMMaskDecoder
from keras_hub.src.models.sam.sam_prompt_encoder import SAMPromptEncoder
from keras_hub.src.models.vit_det.vit_det_backbone import ViTDetBackbone
from keras_hub.src.tests.test_case import TestCase


class SAMImageSegmenterTest(TestCase):
    def setUp(self):
        # Setup model.
        self.image_size = 128
        self.batch_size = 2
        self.images = np.ones(
            (self.batch_size, self.image_size, self.image_size, 3),
            dtype="float32",
        )
        self.image_encoder = ViTDetBackbone(
            hidden_size=16,
            num_layers=16,
            intermediate_dim=16 * 4,
            num_heads=16,
            global_attention_layer_indices=[2, 5, 8, 11],
            patch_size=16,
            num_output_channels=8,
            window_size=2,
            image_shape=(self.image_size, self.image_size, 3),
        )
        self.prompt_encoder = SAMPromptEncoder(
            hidden_size=8,
            image_embedding_size=(8, 8),
            input_image_size=(
                self.image_size,
                self.image_size,
            ),
            mask_in_channels=16,
        )
        self.mask_decoder = SAMMaskDecoder(
            num_layers=2,
            hidden_size=8,
            intermediate_dim=32,
            num_heads=8,
            embedding_dim=8,
            num_multimask_outputs=3,
            iou_head_depth=3,
            iou_head_hidden_dim=8,
        )
        self.backbone = SAMBackbone(
            image_encoder=self.image_encoder,
            prompt_encoder=self.prompt_encoder,
            mask_decoder=self.mask_decoder,
        )
        self.init_kwargs = {
            "backbone": self.backbone,
        }
        self.inputs = {
            "images": self.images,
            "points": np.ones((self.batch_size, 1, 2), dtype="float32"),
            "labels": np.ones((self.batch_size, 1), dtype="float32"),
            "boxes": np.ones((self.batch_size, 1, 2, 2), dtype="float32"),
            "masks": np.zeros(
                (self.batch_size, 0, self.image_size, self.image_size, 1)
            ),
        }
        self.labels = {
            "masks": np.ones((self.batch_size, 2), dtype="float32"),
            "iou_pred": np.ones(self.batch_size, dtype="float32"),
        }
        self.train_data = (
            self.inputs,
            self.labels,
        )

    def test_sam_basics(self):
        pytest.skip(
            reason="TODO: enable after preprocessor flow is figured out"
        )
        self.run_task_test(
            cls=SAMImageSegmenter,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape={
                "masks": [2, 2, 1],
                "iou_pred": [2],
            },
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=SAMImageSegmenter,
            init_kwargs=self.init_kwargs,
            input_data=self.inputs,
        )

    def test_end_to_end_model_predict(self):
        model = SAMImageSegmenter(**self.init_kwargs)
        outputs = model.predict(self.inputs)
        masks, iou_pred = outputs["masks"], outputs["iou_pred"]
        self.assertAllEqual(masks.shape, (2, 4, 32, 32))
        self.assertAllEqual(iou_pred.shape, (2, 4))
