import os

from keras_hub.src.models.embedding_gemma2.embedding_gemma2_backbone import (
    EmbeddingGemma2Backbone,
)
from keras_hub.src.models.embedding_gemma2.embedding_gemma2_tokenizer import (
    EmbeddingGemma2Tokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class EmbeddingGemma2TokenizerTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "proto": os.path.join(
                self.get_test_data_dir(), "gemma4_test_vocab.spm"
            )
        }
        self.input_data = ["the quick brown fox", "the earth is round"]

    def test_tokenizer_basics(self):
        self.assertIs(
            EmbeddingGemma2Tokenizer.backbone_cls, EmbeddingGemma2Backbone
        )
        self.run_preprocessing_layer_test(
            cls=EmbeddingGemma2Tokenizer,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
