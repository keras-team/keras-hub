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

from keras_nlp.src.models.paligemma.vision_embeddings import VisionEmbeddings
from keras_nlp.src.models.paligemma.vit import VitEncoder
from keras_nlp.src.tests.test_case import TestCase


@pytest.mark.keras_3_only
class VITTest(TestCase):
    def test_vit_encoder(self):
        #  encoder calls Attention and CLIPLayer, both of which gets
        # verified with this test
        batch_size = 32
        image_resolution = 224
        hidden_dim = 64
        num_layers = 4
        heads = 8
        intermediate_size = 256
        vit_encoder = VitEncoder(
            hidden_dim, num_layers, heads, intermediate_size
        )
        dummy_input = np.random.rand(
            batch_size, image_resolution, image_resolution, 3
        )
        output = vit_encoder(dummy_input)
        self.assertEqual(
            output.shape, (batch_size, intermediate_size, hidden_dim)
        )

    def test_vision_embeddings(self):
        embeddings_layer = VisionEmbeddings(3)
        dummy_input = np.ones([1, 224, 224, 3])
        vision_embeddings = embeddings_layer(dummy_input)
        self.assertEqual(vision_embeddings.shape, (1, 256, 3))
