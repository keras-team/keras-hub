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
import os.path

import numpy as np
import pytest

from keras_nlp.src.models.paligemma.pali_gemma_backbone import PaliGemmaBackbone
from keras_nlp.src.models.paligemma.pali_gemma_causal_lm import (
    PaliGemmaCausalLM,
)
from keras_nlp.src.models.paligemma.pali_gemma_causal_lm_preprocesor import (
    PaliGemmaCausalLMPreprocessor,
)
from keras_nlp.src.models.paligemma.pali_gemma_tokenizer import (
    PaliGemmaTokenizer,
)
from keras_nlp.src.tests.test_case import TestCase


@pytest.mark.keras_3_only
class PaliGemmaCausalLMTest(TestCase):

    def setUp(self):
        self.batch_size = 2
        self.vocabulary_size = 256
        self.text_sequence_length = 64
        self.image_size = 224
        self.dummy_text = [
            "the quick brown fox" for _ in range(self.batch_size)
        ]
        self.dummy_images = np.random.uniform(
            size=(
                self.batch_size,
                self.image_size,
                self.image_size,
                3,
            )
        )

        proto = "gemma_test_vocab.spm"
        tokenizer = PaliGemmaTokenizer(
            os.path.join(self.get_test_data_dir(), proto)
        )
        self.preprocessor = PaliGemmaCausalLMPreprocessor(
            tokenizer,
            self.text_sequence_length,
            add_start_token=False,
            add_end_token=False,
        )

        self.backbone = PaliGemmaBackbone(
            self.vocabulary_size,
            image_size=224,
            num_layers=27,
            num_query_heads=16,
            num_key_value_heads=16,
            hidden_dim=256,
            intermediate_dim=256,
            head_dim=126,
            dtype="float32",
            vit_patch_size=14,
            vit_num_heads=8,
            vit_hidden_dim=16,
            vit_num_layers=2,
            vit_intermediate_dim=8,
            vit_num_classes=512,
        )

    def test_paligemma_causal_model(self):

        preprocessed, _, _ = self.preprocessor(
            {"images": self.dummy_images, "text": self.dummy_text}
        )

        pali_gemma = PaliGemmaCausalLM(self.preprocessor, self.backbone)

        output = pali_gemma(inputs=preprocessed)

        self.assertAllEqual(
            output["text_output"].shape,
            (
                self.batch_size,
                self.text_sequence_length
                + self.backbone.vit_encoder.output_token_length,
                self.vocabulary_size,
            ),
        )
