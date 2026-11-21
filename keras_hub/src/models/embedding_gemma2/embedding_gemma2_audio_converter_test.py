import numpy as np

from keras_hub.src.models.embedding_gemma2.embedding_gemma2_audio_converter import (  # noqa: E501
    EmbeddingGemma2AudioConverter,
)
from keras_hub.src.models.embedding_gemma2.embedding_gemma2_backbone import (
    EmbeddingGemma2Backbone,
)
from keras_hub.src.tests.test_case import TestCase


class EmbeddingGemma2AudioConverterTest(TestCase):
    def test_audio_converter_basics(self):
        converter = EmbeddingGemma2AudioConverter(
            num_mels=8,
            num_fft_bins=16,
            stride=8,
            max_audio_length=1,
            sampling_rate=160,
        )
        self.assertIs(converter.backbone_cls, EmbeddingGemma2Backbone)
        # 160 samples / stride 8 -> 20 frames of 8 mel bins.
        out = converter(np.ones((2, 160), dtype="float32"))
        self.assertEqual(tuple(out.shape), (2, 20, 8))
