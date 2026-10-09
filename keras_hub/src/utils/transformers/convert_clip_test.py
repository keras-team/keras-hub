import numpy as np
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.clip.clip_backbone import CLIPBackbone
from keras_hub.src.models.clip.clip_preprocessor import CLIPPreprocessor
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_preset(self):
        # `openai/clip-vit-base-patch32` does not ship `model.safetensors`.
        model = CLIPBackbone.from_preset(
            "hf://laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
        )
        images = np.ones((1, 224, 224, 3), dtype="float32")
        token_ids = np.ones((1, 77), dtype="int32")
        outputs = model.predict({"images": images, "token_ids": token_ids})
        self.assertEqual(outputs["vision_logits"].shape, (1, 1))
        self.assertEqual(outputs["text_logits"].shape, (1, 1))

    @pytest.mark.large
    def test_class_detection(self):
        model = Backbone.from_preset(
            "hf://openai/clip-vit-base-patch32",
            load_weights=False,
        )
        self.assertIsInstance(model, CLIPBackbone)

    @pytest.mark.large
    def test_dtype_propagation(self):
        model = CLIPBackbone.from_preset(
            "hf://openai/clip-vit-base-patch32",
            load_weights=False,
            dtype="bfloat16",
        )
        self.assertEqual(model.dtype_policy.name, "bfloat16")
        self.assertEqual(model.vision_encoder.dtype_policy.name, "bfloat16")
        self.assertEqual(model.text_encoder.dtype_policy.name, "bfloat16")

    @pytest.mark.large
    def test_preprocessor(self):
        preprocessor = CLIPPreprocessor.from_preset(
            "hf://openai/clip-vit-base-patch32"
        )
        outputs = preprocessor(
            {
                "images": np.ones((1, 224, 224, 3), dtype="float32"),
                "prompts": ["a photo of a cat"],
            }
        )
        self.assertEqual(outputs["token_ids"].shape, (1, 77))
        self.assertEqual(outputs["images"].shape, (1, 224, 224, 3))
