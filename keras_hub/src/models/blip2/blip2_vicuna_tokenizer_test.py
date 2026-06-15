import os

from keras_hub.src.models.blip2.blip2_backbone import BLIP2Backbone
from keras_hub.src.models.blip2.blip2_vicuna_tokenizer import (
    BLIP2VicunaTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class BLIP2VicunaTokenizerTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "proto": os.path.join(
                self.get_test_data_dir(), "llama_test_vocab.spm"
            )
        }
        self.input_data = ["the quick brown fox", "the earth is round"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=BLIP2VicunaTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[3, 8, 4, 6], [3, 5, 7, 9]],
        )

    def test_backbone_cls(self):
        # backbone_cls must be BLIP2Backbone (not LlamaBackbone) so that
        # from_preset() wires to the InstructBLIP backbone.
        self.assertEqual(BLIP2VicunaTokenizer.backbone_cls, BLIP2Backbone)

    def test_special_tokens(self):
        tokenizer = BLIP2VicunaTokenizer(**self.init_kwargs)
        # InstructBLIP-Vicuna sets bos == eos == "</s>", so the start token is
        # "</s>" (not LLaMA's default "<s>") to match HF's LM prompt.
        self.assertEqual(tokenizer.start_token, "</s>")
        self.assertEqual(tokenizer.end_token, "</s>")
        self.assertEqual(tokenizer.padding_side, "right")
