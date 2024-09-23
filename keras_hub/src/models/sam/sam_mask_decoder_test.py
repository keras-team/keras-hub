# Copyright 2024 The KerasHub Authors
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


from keras import random

from keras_hub.src.models.sam.sam_mask_decoder import SAMMaskDecoder
from keras_hub.src.tests.test_case import TestCase


class SAMMaskDecoderTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.image_size = 128
        self.init_kwargs = {
            "num_layers": 2,
            "hidden_size": 8,
            "intermediate_dim": 32,
            "num_heads": 8,
            "embedding_dim": 8,
            "num_multimask_outputs": 3,
            "iou_head_depth": 3,
            "iou_head_hidden_dim": 8,
        }
        self.inputs = {
            "image_embeddings": random.uniform(
                minval=0, maxval=1, shape=(1, 8, 8, 8)
            ),
            "prompt_sparse_embeddings": random.uniform(
                minval=0, maxval=1, shape=(1, 12, 8)
            ),
            "prompt_dense_embeddings": random.uniform(
                minval=0, maxval=1, shape=(1, 8, 8, 8)
            ),
            "prompt_dense_positional_embeddings": random.uniform(
                minval=0, maxval=1, shape=(1, 8, 8, 8)
            ),
        }

    def test_layer_basics(self):
        self.run_layer_test(
            cls=SAMMaskDecoder,
            init_kwargs=self.init_kwargs,
            input_data=self.inputs,
            expected_output_shape={
                "masks": (1, 4, 32, 32),
                "iou_pred": (1, 4),
            },
            expected_num_trainable_weights=120,
            run_precision_checks=False,
        )
