# Copyright 2024 The kerasCV Authors
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

from keras_hub.src.models.sam.sam_backbone import SAMBackbone
from keras_hub.src.models.sam.sam_mask_decoder import SAMMaskDecoder
from keras_hub.src.models.sam.sam_prompt_encoder import SAMPromptEncoder
from keras_hub.src.models.vit_det.vit_det_backbone import ViTDetBackbone
from keras_hub.src.tests.test_case import TestCase


class SAMBackboneTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.image_size = 16
        self.image_encoder = ViTDetBackbone(
            hidden_size=16,
            num_layers=16,
            intermediate_dim=16 * 4,
            num_heads=16,
            global_attention_layer_indices=[2, 5, 8, 11],
            patch_size=16,
            num_output_channels=8,
            window_size=2,
            image_shape=(self.image_size, self.image_size, 3),
        )
        self.prompt_encoder = SAMPromptEncoder(
            hidden_size=8,
            image_embedding_size=(8, 8),
            input_image_size=(
                self.image_size,
                self.image_size,
            ),
            mask_in_channels=16,
        )
        self.mask_decoder = SAMMaskDecoder(
            transformer_num_layers=2,
            transformer_hidden_size=8,
            transformer_intermediate_dim=32,
            transformer_num_heads=8,
            transformer_dim=8,
            num_multimask_outputs=3,
            iou_head_depth=3,
            iou_head_hidden_dim=8,
        )
        self.init_kwargs = {
            "image_encoder": self.image_encoder,
            "prompt_encoder": self.prompt_encoder,
            "mask_decoder": self.mask_decoder,
            "image_shape": (self.image_size, self.image_size, 3),
        }

        self.input_data = {
            "images": np.ones(
                (self.batch_size, self.image_size, self.image_size, 3),
                dtype="float32",
            ),
            "points": np.ones((self.batch_size, 1, 2), dtype="float32"),
            "labels": np.ones((self.batch_size, 1), dtype="float32"),
            "boxes": np.ones((self.batch_size, 1, 2, 2), dtype="float32"),
            "masks": np.zeros(
                (self.batch_size, 0, self.image_size, self.image_size, 1)
            ),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=SAMBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape={
                "image_embeddings": (2, 1, 1, 8),
                "prompt_sparse_embeddings": (2, 3, 8),
                "prompt_dense_embeddings": (2, 8, 8, 8),
                "prompt_dense_positional_embeddings": (1, 8, 8, 8),
            },
            run_mixed_precision_check=False,
            run_quantization_check=False,
        )
