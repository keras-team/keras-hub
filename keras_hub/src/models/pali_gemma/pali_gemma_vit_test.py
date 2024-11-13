import numpy as np

from keras_hub.src.models.pali_gemma.pali_gemma_vit import PaliGemmaVit
from keras_hub.src.models.pali_gemma.pali_gemma_vit import (
    PaliGemmaVitEmbeddings,
)
from keras_hub.src.models.pali_gemma.pali_gemma_vit import PaliGemmaVitEncoder
from keras_hub.src.tests.test_case import TestCase


class PaliGemmaVitTest(TestCase):
    def test_vit_encoder(self):
        #  encoder calls Attention and CLIPLayer, both of which gets
        # verified with this test
        batch_size = 2
        image_size = 16
        hidden_dim = 8
        intermediate_dim = 16
        vit_encoder = PaliGemmaVitEncoder(
            image_size=image_size,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=2,
            num_heads=2,
            patch_size=4,
        )
        dummy_input = np.random.rand(batch_size, image_size, image_size, 3)
        output = vit_encoder(dummy_input)
        self.assertEqual(
            output.shape, (batch_size, intermediate_dim, hidden_dim)
        )

    def test_vision_embeddings(self):
        embeddings_layer = PaliGemmaVitEmbeddings(
            image_size=16,
            patch_size=4,
            hidden_dim=8,
        )
        dummy_input = np.ones([1, 16, 16, 3])
        vision_embeddings = embeddings_layer(dummy_input)
        self.assertEqual(vision_embeddings.shape, (1, 16, 8))

    def test_vit_output_shape(self):
        embeddings_layer = PaliGemmaVit(
            image_size=16,
            patch_size=4,
            hidden_dim=8,
            num_layers=2,
            num_heads=2,
            intermediate_dim=16,
            num_classes=32,
        )
        dummy_input = np.ones([1, 16, 16, 3])
        image_embeddings = embeddings_layer(dummy_input)
        self.assertEqual(image_embeddings.shape, (1, 16, 32))
