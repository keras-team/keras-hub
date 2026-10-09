from keras_hub.src.models.embedding_gemma2.embedding_gemma2_image_converter import (  # noqa: E501
    EmbeddingGemma2ImageConverter,
)
from keras_hub.src.tests.test_case import TestCase


class EmbeddingGemma2ImageConverterTest(TestCase):
    def test_image_converter_basics(self):
        converter = EmbeddingGemma2ImageConverter(
            image_size=(16, 16),
            patch_size=16,
            pooling_kernel_size=1,
            max_soft_tokens=10,
        )
        self.assertTrue(hasattr(converter, "backbone_cls"))
