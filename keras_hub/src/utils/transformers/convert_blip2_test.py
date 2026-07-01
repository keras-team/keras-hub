import numpy as np
import pytest

from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.blip2.blip2_backbone import BLIP2Backbone
from keras_hub.src.models.blip2.blip2_causal_lm import BLIP2CausalLM
from keras_hub.src.models.blip2.blip2_seq_2_seq_lm import BLIP2Seq2SeqLM
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
from keras_hub.src.tests.test_case import TestCase


class TestTask(TestCase):
    @pytest.mark.extra_large
    def test_convert_opt_preset(self):
        # BLIP-2 with an OPT (decoder-only) language model -> BLIP2CausalLM.
        model = BLIP2CausalLM.from_preset("hf://Salesforce/blip2-opt-2.7b")
        image = np.random.rand(224, 224, 3)
        model.generate({"images": image, "text": "a photo of"}, max_length=15)

    @pytest.mark.extra_large
    def test_convert_flan_t5_preset(self):
        # BLIP-2 with a Flan-T5 (encoder-decoder) LM -> BLIP2Seq2SeqLM.
        model = BLIP2Seq2SeqLM.from_preset("hf://Salesforce/blip2-flan-t5-xl")
        image = np.random.rand(224, 224, 3)
        model.generate(
            {"images": image, "encoder_text": "a photo of"}, max_length=15
        )

    @pytest.mark.large
    def test_class_detection_opt(self):
        preset = "hf://Salesforce/blip2-opt-2.7b"
        model = CausalLM.from_preset(preset, load_weights=False)
        self.assertIsInstance(model, BLIP2CausalLM)
        model = Backbone.from_preset(preset, load_weights=False)
        self.assertIsInstance(model, BLIP2Backbone)

    @pytest.mark.large
    def test_class_detection_flan_t5(self):
        preset = "hf://Salesforce/blip2-flan-t5-xl"
        model = Seq2SeqLM.from_preset(preset, load_weights=False)
        self.assertIsInstance(model, BLIP2Seq2SeqLM)
        model = Backbone.from_preset(preset, load_weights=False)
        self.assertIsInstance(model, BLIP2Backbone)
