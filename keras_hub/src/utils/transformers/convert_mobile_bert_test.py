import numpy as np
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.mobile_bert.mobile_bert_backbone import (
    MobileBertBackbone,
)
from keras_hub.src.tests.test_case import TestCase


class ConvertMobileBertTest(TestCase):
    @pytest.mark.large
    def test_convert_mnli_preset(self):
        model = MobileBertBackbone.from_preset(
            "hf://typeform/mobilebert-uncased-mnli"
        )
        self.assertIsInstance(model, MobileBertBackbone)

    @pytest.mark.large
    def test_class_detection(self):
        model = Backbone.from_preset(
            "hf://typeform/mobilebert-uncased-mnli",
            load_weights=False,
        )
        self.assertIsInstance(model, MobileBertBackbone)

        token_ids = np.zeros((2, 8), dtype="int32")
        segment_ids = np.zeros((2, 8), dtype="int32")
        padding_mask = np.ones((2, 8), dtype="int32")

        outputs = model(
            {
                "token_ids": token_ids,
                "segment_ids": segment_ids,
                "padding_mask": padding_mask,
            }
        )
        self.assertEqual(outputs["sequence_output"].shape, (2, 8, 512))
        self.assertEqual(outputs["pooled_output"].shape, (2, 512))

    @pytest.mark.large
    def test_tokenizer_from_preset(self):
        from keras_hub.src.models.mobile_bert.mobile_bert_tokenizer import (
            MobileBertTokenizer,
        )

        tokenizer = MobileBertTokenizer.from_preset(
            "hf://typeform/mobilebert-uncased-mnli"
        )
        self.assertIsInstance(tokenizer, MobileBertTokenizer)
        outputs = tokenizer("The quick brown fox")
        self.assertAllEqual(outputs, [1996, 4248, 2829, 4419])
