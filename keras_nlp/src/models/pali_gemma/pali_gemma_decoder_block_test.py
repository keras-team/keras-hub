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

from keras_nlp.src.models.pali_gemma.pali_gemma_decoder_block import (
    PaliGemmaDecoderBlock,
)
from keras_nlp.src.tests.test_case import TestCase


class PaliGemmaDecoderBlockTest(TestCase):
    def setUp(self):
        self.batch_size = 4

        self.img_sequence_length = 4
        self.text_prompt_length = 2
        self.response_length = 2

        self.text_sequence_length = (
            self.text_prompt_length + self.response_length
        )
        self.total_sequence_length = (
            self.img_sequence_length + self.text_sequence_length
        )

        self.hidden_dim = 64

        self.decoder_block = PaliGemmaDecoderBlock(
            self.hidden_dim, 64, 64, 8, 8
        )

        self.dummy_input = np.random.rand(
            self.batch_size, self.total_sequence_length, self.hidden_dim
        )

    def _build_causal_mask(self):
        mask = np.zeros(
            (
                self.batch_size,
                self.total_sequence_length,
                self.total_sequence_length,
            )
        )
        for i in range(self.total_sequence_length):
            mask[:, i, : i + 1] = True
        return mask

    def test_pali_gemma_attention_mask_computation(self):
        attn_mask = self.decoder_block._compute_attention_mask(
            self.dummy_input, None, None, 0
        )
        expected_mask = self._build_causal_mask()
        self.assertAllEqual(
            expected_mask,
            attn_mask,
        )

    def test_pali_gemma_attention_mask_computation_with_response_mask(self):
        response_mask = np.full(
            (self.batch_size, self.text_sequence_length), False
        )
        response_mask[:, self.text_prompt_length :] = True

        attn_mask = self.decoder_block._compute_attention_mask(
            self.dummy_input, None, None, 0
        )
        expected_mask = self._build_causal_mask()
        self.assertAllEqual(
            expected_mask,
            attn_mask,
        )

    def test_pali_gemma_attention_mask_computation_with_dual_masks(self):
        padding_mask = np.full(
            (self.batch_size, self.text_sequence_length), False
        )
        padding_mask[:, :2] = True

        response_mask = np.full(
            (self.batch_size, self.text_sequence_length), False
        )
        response_mask[:, self.text_prompt_length :] = True

        attn_mask = self.decoder_block._compute_attention_mask(
            self.dummy_input, None, None, 0
        )
        expected_mask = self._build_causal_mask()
        self.assertAllEqual(
            expected_mask,
            attn_mask,
        )
