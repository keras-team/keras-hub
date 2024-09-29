import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.bart.bart_backbone import BartBackbone
from keras_hub.src.models.bart.bart_seq_2_seq_lm import BartSeq2SeqLM
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.large
    def test_convert_tiny_preset(self):
        model = BartSeq2SeqLM.from_preset("hf://cosmo3769/tiny-bart-test")
        prompt = "What is your favorite condiment?"
        model.generate([prompt], max_length=15)

    @pytest.mark.large
    def test_class_detection(self):
        model = Seq2SeqLM.from_preset(
            "hf://cosmo3769/tiny-bart-test",
            load_weights=False,
        )
        self.assertIsInstance(model, BartSeq2SeqLM)
        model = Backbone.from_preset(
            "hf://cosmo3769/tiny-bart-test",
            load_weights=False,
        )
        self.assertIsInstance(model, BartBackbone)

    # TODO: compare numerics with huggingface model
