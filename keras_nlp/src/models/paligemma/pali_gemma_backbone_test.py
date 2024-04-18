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

import keras
import numpy as np
import pytest

from keras_nlp.src.models.gemma.gemma_preprocessor import GemmaPreprocessor
from keras_nlp.src.models.gemma.gemma_preprocessor import GemmaTokenizer
from keras_nlp.src.models.paligemma.pali_gemma_backbone import PaliGemmaBackbone
from keras_nlp.src.tests.test_case import TestCase


@pytest.mark.keras_3_only
class PaliGemmaBackboneTest(TestCase):
    def test_paligemma_preprocessing(self):
        batch_size = 1
        text_sequence_length = 8
        proto = os.path.join(self.get_test_data_dir(), "gemma_test_vocab.spm")

        tokenizer = GemmaTokenizer(proto)

        preprocessor = GemmaPreprocessor(
            tokenizer, text_sequence_length, False, False
        )

        dummy_text = ["answer en the quick brown fox"]

        output = preprocessor(dummy_text)
        self.assertEqual(
            (
                batch_size,
                text_sequence_length,
            ),
            output["token_ids"].shape,
        )

    def test_paligemma_backbone(self):
        batch_size = 2
        img_sequence_length = 128
        vocabulary_size = 256
        text_sequence_length = 64
        hidden_dim = 128
        num_layers = 27
        num_heads = 16
        head_dim = 126
        intermediate_size = 77

        paligemma = PaliGemmaBackbone(
            img_sequence_length,
            vocabulary_size,
            num_layers,
            num_heads,
            num_heads,
            hidden_dim,
            intermediate_size,
            head_dim,
            dtype="float32",
        )

        dummy_imgs = np.random.rand(batch_size, img_sequence_length, hidden_dim)
        dummy_text = np.random.rand(batch_size, text_sequence_length)

        output = paligemma(
            inputs={
                "token_ids": dummy_text,
                "img_embeddings": dummy_imgs,
                "padding_mask": np.ones(
                    (batch_size, text_sequence_length + img_sequence_length),
                    dtype="int32",
                ),
            }
        )
        self.assertEqual(
            (
                batch_size,
                text_sequence_length + img_sequence_length,
                hidden_dim,
            ),
            output.shape,
        )

    def test_complete_paligemma_backbone(self):
        batch_size = 2
        img_sequence_length = 128
        vocabulary_size = 256
        text_sequence_length = 64
        hidden_dim = 128
        num_layers = 27
        num_heads = 16
        head_dim = 126
        intermediate_size = 77
        proto = os.path.join(self.get_test_data_dir(), "gemma_test_vocab.spm")

        tokenizer = GemmaTokenizer(proto)

        preprocessor = GemmaPreprocessor(
            tokenizer, text_sequence_length, False, False
        )

        paligemma = PaliGemmaBackbone(
            img_sequence_length,
            vocabulary_size,
            num_layers,
            num_heads,
            num_heads,
            hidden_dim,
            intermediate_size,
            head_dim,
            dtype="float32",
        )

        dummy_imgs = keras.ops.convert_to_tensor(
            np.random.rand(batch_size, img_sequence_length, hidden_dim)
        )
        dummy_text = [
            "answer en the quick brown fox" for i in range(batch_size)
        ]

        output = preprocessor(dummy_text)

        output = paligemma(
            inputs={
                "token_ids": output["token_ids"],
                "img_embeddings": dummy_imgs,
                "padding_mask": keras.ops.ones(
                    (batch_size, text_sequence_length + img_sequence_length),
                    dtype="int32",
                ),
            }
        )
        self.assertEqual(
            (
                batch_size,
                text_sequence_length + img_sequence_length,
                hidden_dim,
            ),
            output.shape,
        )
