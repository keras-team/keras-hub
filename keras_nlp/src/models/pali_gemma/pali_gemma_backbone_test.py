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
        self.image_sequence_length = int((self.image_size / 4) ** 2)

        proto = "gemma_test_vocab.spm"
        tokenizer = PaliGemmaTokenizer(
            os.path.join(self.get_test_data_dir(), proto)
        )
        self.preprocessor = PaliGemmaCausalLMPreprocessor(
            tokenizer, self.text_sequence_length, False, False
        )

        self.init_kwargs = {
            "vocabulary_size": self.vocabulary_size,
            "image_size": self.image_size,
            "num_layers": 2,
            "num_query_heads": 2,
            "num_key_value_heads": 1,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            "head_dim": 4,
            "vit_patch_size": 4,
            "vit_num_layers": 2,
            "vit_num_heads": 2,
            "vit_hidden_dim": 8,
            "vit_intermediate_dim": 16,
        }

        dummy_images = np.random.rand(
            self.batch_size, self.image_size, self.image_size, 3
        )
        dummy_text_token_ids = np.random.rand(
            self.batch_size, self.text_sequence_length
        )
        dummy_text = ["answer en the quick brown fox"] * self.batch_size
        self.input_data = {
            "token_ids": dummy_text_token_ids,
            "images": dummy_images,
            "padding_mask": np.ones(
                (self.batch_size, self.text_sequence_length),
                dtype="int32",
            ),
            "response_mask": np.zeros(
                (self.batch_size, self.text_sequence_length),
                dtype="int32",
            ),
        }
        self.raw_input_data = {
            "images": dummy_images,
            "prompts": dummy_text,
            "responses": dummy_text,
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=PaliGemmaBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(
                self.batch_size,
                self.text_sequence_length + self.image_sequence_length,
                8,
            ),
            variable_length_data=[self.input_data],
            run_mixed_precision_check=False,  # TODO: Set to `True`
        )

    def test_pali_gemma_backbone_with_preprocessing(self):
        model = PaliGemmaBackbone(**self.init_kwargs)
        x, _, _ = self.preprocessor(self.raw_input_data)
        output = model(x)
        self.assertEqual(
            (
                self.batch_size,
                self.text_sequence_length + self.image_sequence_length,
                8,
            ),
            output.shape,
        )
