import pytest
from keras import ops

from keras_hub.src.models.siglip.siglip_backbone import SigLIPBackbone
from keras_hub.src.models.siglip.siglip_text_encoder import SigLIPTextEncoder
from keras_hub.src.models.siglip.siglip_vision_encoder import (
    SigLIPVisionEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class SigLIPBackboneTest(TestCase):
    def setUp(self):
        vision_encoder = SigLIPVisionEncoder(
            16, 64, 2, 2, 128, name="vision_encoder"
        )
        text_encoder = SigLIPTextEncoder(
            64, 64, 64, 2, 2, 128, name="text_encoder"
        )
        self.init_kwargs = {
            "vision_encoder": vision_encoder,
            "text_encoder": text_encoder,
        }
        self.input_data = {
            "images": ops.ones((2, 224, 224, 3)),
            "token_ids": ops.ones((2, 64), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=SigLIPBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "vision_logits": (2, 2),
                "text_logits": (2, 2),
            },
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=SigLIPBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=SigLIPBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=SigLIPBackbone,
            preset="siglip_base_patch16_224",
            input_data=self.input_data,
            expected_output_shape={
                "vision_logits": (2, 2),
                "text_logits": (2, 2),
            },
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in SigLIPBackbone.presets:
            self.run_preset_test(
                cls=SigLIPBackbone,
                preset=preset,
                input_data=self.input_data,
            )


class SigLIP2BackboneTest(TestCase):
    def setUp(self):
        vision_encoder = SigLIPVisionEncoder(
            16, 128, 2, 2, 128, name="vision_encoder"
        )
        text_encoder = SigLIPTextEncoder(
            64, 64, 64, 2, 2, 128, projection_dim=128, name="text_encoder"
        )
        self.init_kwargs = {
            "vision_encoder": vision_encoder,
            "text_encoder": text_encoder,
        }
        self.input_data = {
            "images": ops.ones((2, 224, 224, 3)),
            "token_ids": ops.ones((2, 64), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=SigLIPBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "vision_logits": (2, 2),
                "text_logits": (2, 2),
            },
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=SigLIPBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_litert_export(self):
        self.run_litert_export_test(
            cls=SigLIPBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.kaggle_key_required
    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=SigLIPBackbone,
            preset="siglip2_base_patch16_224",
            input_data=self.input_data,
            expected_output_shape={
                "vision_logits": (2, 2),
                "text_logits": (2, 2),
            },
        )
