from keras_hub.src.models.blip2.blip2_backbone import BLIP2Backbone
from keras_hub.src.models.blip2.blip2_qformer_tokenizer import (
    BLIP2QFormerTokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class BLIP2QFormerTokenizerTest(TestCase):
    def setUp(self):
        self.vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        self.vocab += ["the", "quick", "brown", "fox", "earth", "is", "round"]
        self.init_kwargs = {"vocabulary": self.vocab}
        self.input_data = ["the quick brown fox", "the earth is round"]

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=BLIP2QFormerTokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=[[5, 6, 7, 8], [5, 9, 10, 11]],
        )

    def test_backbone_cls(self):
        self.assertEqual(BLIP2QFormerTokenizer.backbone_cls, BLIP2Backbone)

    def test_special_tokens(self):
        tokenizer = BLIP2QFormerTokenizer(**self.init_kwargs)
        self.assertEqual(tokenizer.cls_token, "[CLS]")
        self.assertEqual(tokenizer.sep_token, "[SEP]")
        self.assertEqual(tokenizer.pad_token, "[PAD]")
        # Aliases used by the preprocessor / cross-tokenizer code.
        self.assertEqual(tokenizer.start_token, "[CLS]")
        self.assertEqual(tokenizer.end_token, "[SEP]")

    def test_dedicated_config_file(self):
        # The Q-Former tokenizer must use its own preset asset subdir so its
        # vocabulary never collides with the language-model tokenizer.
        tokenizer = BLIP2QFormerTokenizer(**self.init_kwargs)
        self.assertEqual(tokenizer.config_file, "qformer_tokenizer.json")
