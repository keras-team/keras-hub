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
