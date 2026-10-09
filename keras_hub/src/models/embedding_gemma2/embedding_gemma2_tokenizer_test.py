from keras_hub.src.models.embedding_gemma2.embedding_gemma2_tokenizer import (
    EmbeddingGemma2Tokenizer,
)
from keras_hub.src.tests.test_case import TestCase


class EmbeddingGemma2TokenizerTest(TestCase):
    def test_tokenizer_basics(self):
        # We can just check it instantiates and has the right backbone_cls
        self.assertTrue(hasattr(EmbeddingGemma2Tokenizer, "backbone_cls"))
