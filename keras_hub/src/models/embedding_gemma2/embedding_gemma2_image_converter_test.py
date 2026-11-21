import numpy as np
from keras import ops

from keras_hub.src.models.embedding_gemma2.embedding_gemma2_backbone import (
    EmbeddingGemma2Backbone,
)
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
        self.assertIs(converter.backbone_cls, EmbeddingGemma2Backbone)
        out = converter(np.ones((1, 16, 16, 3), dtype="float32"))
        # The image is resized to fill the `max_soft_tokens` budget: a square
        # image with a budget of 10 lands on a 3x3 patch grid, so 9 valid
        # patches (16*16*3 values each, with a (row, col) position id) and one
        # padded slot marked -1.
        self.assertEqual(
            tuple(ops.shape(out["pixel_values"])), (1, 10, 16 * 16 * 3)
        )
        position_ids = ops.convert_to_numpy(out["pixel_position_ids"])
        self.assertEqual(position_ids.shape, (1, 10, 2))
        self.assertEqual(int((position_ids[0, :, 0] != -1).sum()), 9)
