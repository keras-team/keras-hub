# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import pytest

from keras_nlp.src.models.pali_gemma.pali_gemma_vit import PaliGemmaVit
from keras_nlp.src.models.pali_gemma.pali_gemma_vit import (
    PaliGemmaVitEmbeddings,
)
from keras_nlp.src.models.pali_gemma.pali_gemma_vit import PaliGemmaVitEncoder
from keras_nlp.src.tests.test_case import TestCase


@pytest.mark.keras_3_only
class PaliGemmaVitTest(TestCase):
    def test_vit_encoder(self):
        #  encoder calls Attention and CLIPLayer, both of which gets
        # verified with this test
        batch_size = 32
        image_size = 224
        hidden_dim = 64
        num_layers = 4
        heads = 8
        intermediate_dim = 256
        vit_encoder = PaliGemmaVitEncoder(
            hidden_dim,
            num_layers,
            heads,
            intermediate_dim,
            patch_size=14,
            image_size=image_size,
        )
        dummy_input = np.random.rand(batch_size, image_size, image_size, 3)
        output = vit_encoder(dummy_input)
        self.assertEqual(
            output.shape, (batch_size, intermediate_dim, hidden_dim)
        )

    def test_vision_embeddings(self):
        embeddings_layer = PaliGemmaVitEmbeddings(3)
        dummy_input = np.ones([1, 224, 224, 3])
        vision_embeddings = embeddings_layer(dummy_input)
        self.assertEqual(vision_embeddings.shape, (1, 256, 3))

    def test_vit_output_shape(self):
        embeddings_layer = PaliGemmaVit(image_size=224)
        dummy_input = np.ones([1, 224, 224, 3])
        image_embeddings = embeddings_layer(dummy_input)
        self.assertEqual(image_embeddings.shape, (1, 256, 2048))
