import keras
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
from keras_hub.src.models.t5gemma2.t5gemma2_backbone import T5Gemma2Backbone
from keras_hub.src.models.t5gemma2.t5gemma2_seq_2_seq_lm import (
    T5Gemma2Seq2SeqLM,
)
from keras_hub.src.tests.test_case import TestCase


class TestT5Gemma2Converter(TestCase):
    @pytest.mark.skipif(
        keras.backend.backend() == "tensorflow",
        reason="TensorFlow GPU CI OOM (ResourceExhaustedError)",
    )
    @pytest.mark.extra_large
    def test_convert_tiny_preset(self):
        model = T5Gemma2Seq2SeqLM.from_preset("hf://google/t5gemma-2-270m-270m")
        prompt = "What is the capital of France?"
        model.generate([prompt], max_length=15)

    @pytest.mark.large
    def test_class_detection(self):
        preset_name = "hf://google/t5gemma-2-270m-270m"
        model = Seq2SeqLM.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, T5Gemma2Seq2SeqLM)
        model = Backbone.from_preset(
            preset_name,
            load_weights=False,
        )
        self.assertIsInstance(model, T5Gemma2Backbone)
