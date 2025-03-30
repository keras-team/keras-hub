import numpy as np

from keras_hub.src.models.gemma3.gemma3_vit import Gemma3ViT
from keras_hub.src.models.gemma3.gemma3_vit import Gemma3ViTEmbeddings
from keras_hub.src.models.gemma3.gemma3_vit import Gemma3ViTEncoder
from keras_hub.src.tests.test_case import TestCase


class Gemma3ViTTest(TestCase):
    def test_vit_encoder(self):
        batch_size = 2
        image_size = 16
        hidden_dim = 8
        intermediate_dim = 16
        vit_encoder = Gemma3ViTEncoder(
            image_size=image_size,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_layers=2,
            num_heads=2,
            patch_size=4,
        )

        max_images_per_prompt = 2
        dummy_input = np.random.rand(
            batch_size, max_images_per_prompt, image_size, image_size, 3
        )
        output = vit_encoder(dummy_input)
        self.assertEqual(
            output.shape,
            (batch_size * max_images_per_prompt, intermediate_dim, hidden_dim),
        )

    def test_vision_embeddings(self):
        embeddings_layer = Gemma3ViTEmbeddings(
            image_size=16,
            patch_size=4,
            hidden_dim=8,
        )
        dummy_input = np.ones([1, 16, 16, 3])
        vision_embeddings = embeddings_layer(dummy_input)
        self.assertEqual(vision_embeddings.shape, (1, 16, 8))

    def test_vit_output_shape(self):
        embeddings_layer = Gemma3ViT(
            image_size=64,
            patch_size=4,
            hidden_dim=8,
            num_layers=2,
            num_heads=2,
            intermediate_dim=16,
            output_dim=128,
            pool_size=4,
        )
        dummy_input = np.ones([2, 2, 64, 64, 3])
        image_embeddings = embeddings_layer(dummy_input)
        self.assertEqual(image_embeddings.shape, (4, 16, 128))
