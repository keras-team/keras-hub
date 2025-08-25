import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
from keras_hub.src.models.t5gemma.t5gemma_backbone import T5GemmaBackbone
from keras_hub.src.models.t5gemma.t5gemma_seq_2_seq_lm import T5GemmaSeq2SeqLM
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.large
    def test_convert_tiny_preset(self):
        model = T5GemmaSeq2SeqLM.from_preset(
            "hf://harshaljanjani/tiny-t5gemma-test"
        )
        prompt = "What is the capital of France?"
        model.generate([prompt], max_length=15)

    @pytest.mark.large
    def test_class_detection(self):
        preset_name = "hf://harshaljanjani/tiny-t5gemma-test"
        model = Seq2SeqLM.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, T5GemmaSeq2SeqLM)
        model = Backbone.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, T5GemmaBackbone)
