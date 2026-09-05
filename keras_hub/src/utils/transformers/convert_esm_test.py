import numpy as np
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.esm.esm_backbone import ESMBackbone
from keras_hub.src.tests.test_case import TestCase


class TestESMConverter(TestCase):
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = ESMBackbone.from_preset("hf://facebook/esm2_t6_8M_UR50D")
        input_data = {"token_ids": np.ones((1, 10), dtype="int32")}
        output = model.predict(input_data)
        self.assertEqual(output.shape[:2], (1, 10))

    @pytest.mark.large
    def test_class_detection(self):
        model = Backbone.from_preset(
            "hf://facebook/esm2_t6_8M_UR50D",
            load_weights=False,
        )
        self.assertIsInstance(model, ESMBackbone)
