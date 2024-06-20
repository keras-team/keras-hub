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
import os

import numpy as np

from keras_nlp.src.models.pali_gemma.pali_gemma_backbone import (
    PaliGemmaBackbone,
)
from keras_nlp.src.models.pali_gemma.pali_gemma_causal_lm_preprocessor import (
    PaliGemmaCausalLMPreprocessor,
)
from keras_nlp.src.models.pali_gemma.pali_gemma_tokenizer import (
    PaliGemmaTokenizer,
)
from keras_nlp.src.tests.test_case import TestCase


class PaliGemmaBackboneTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.vocabulary_size = 256
        self.text_sequence_length = 64
        self.image_size = 16
        self.dummy_text = [
            "the quick brown fox" for _ in range(self.batch_size)
        ]
        self.dummy_images = np.random.uniform(
            size=(self.batch_size, self.image_size, self.image_size, 3)
        )

        proto = "gemma_test_vocab.spm"
        tokenizer = PaliGemmaTokenizer(
            os.path.join(self.get_test_data_dir(), proto)
        )
        self.preprocessor = PaliGemmaCausalLMPreprocessor(
            tokenizer, self.text_sequence_length, False, False
        )

        self.backbone = PaliGemmaBackbone(
            vocabulary_size=self.vocabulary_size,
            image_size=self.image_size,
            num_layers=2,
            num_query_heads=2,
            num_key_value_heads=1,
            hidden_dim=8,
            intermediate_dim=16,
            head_dim=4,
            vit_patch_size=4,
            vit_num_layers=2,
            vit_num_heads=2,
            vit_hidden_dim=8,
            vit_intermediate_dim=16,
        )
        self.dummy_imgs = np.random.rand(
            self.batch_size, self.image_size, self.image_size, 3
        )
        self.dummy_text_token_ids = np.random.rand(
            self.batch_size, self.text_sequence_length
        )
        self.dummy_text = [
            "answer en the quick brown fox" for i in range(self.batch_size)
        ]

    def test_pali_gemma_backbone(self):
        output = self.backbone(
            {
                "token_ids": self.dummy_text_token_ids,
                "images": self.dummy_imgs,
                "padding_mask": np.ones(
                    (self.batch_size, self.text_sequence_length),
                    dtype="int32",
                ),
                "response_mask": np.zeros(
                    (self.batch_size, self.text_sequence_length),
                    dtype="int32",
                ),
            }
        )
        self.assertEqual(
            (
                self.batch_size,
                self.text_sequence_length + self.backbone.image_sequence_length,
                8,
            ),
            output.shape,
        )

    def test_pali_gemma_backbone_with_preprocessing(self):
        x, _, _ = self.preprocessor(
            {
                "images": self.dummy_images,
                "prompts": self.dummy_text,
                "responses": self.dummy_text,
            }
        )
        output = self.backbone(x)
        self.assertEqual(
            (
                self.batch_size,
                self.text_sequence_length + self.backbone.image_sequence_length,
                8,
            ),
            output.shape,
        )
