# Copyright 2022 The KerasNLP Authors
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
"""Tests for loading pretrained model checkpoints."""

import pytest
import tensorflow as tf
from absl.testing import parameterized

import keras_nlp


@pytest.mark.slow
class BertCkptTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("tiny_uncased_en", keras_nlp.models.BertTiny, "uncased_en"),
        ("small_uncased_en", keras_nlp.models.BertSmall, "uncased_en"),
        ("medium_uncased_en", keras_nlp.models.BertMedium, "uncased_en"),
        ("base_uncased_en", keras_nlp.models.BertBase, "uncased_en"),
        ("base_cased_en", keras_nlp.models.BertBase, "cased_en"),
        ("base_zh", keras_nlp.models.BertBase, "zh"),
        ("base_multi_cased", keras_nlp.models.BertBase, "multi_cased"),
        ("large_uncased_en", keras_nlp.models.BertLarge, "uncased_en"),
        ("large_cased_en", keras_nlp.models.BertLarge, "cased_en"),
    )
    def test_load_bert(self, bert_variant, weights):
        model = bert_variant(weights=weights)
        input_data = {
            "token_ids": tf.random.uniform(
                shape=(1, 512), dtype=tf.int64, maxval=model.vocabulary_size
            ),
            "segment_ids": tf.constant([0] * 200 + [1] * 312, shape=(1, 512)),
            "padding_mask": tf.constant([1] * 512, shape=(1, 512)),
        }
        model(input_data)

    @parameterized.named_parameters(
        ("uncased_en", "uncased_en"),
        ("cased_en", "cased_en"),
        ("zh", "zh"),
        ("multi_cased", "multi_cased"),
    )
    def test_load_vocabularies(self, vocabulary):
        tokenizer = keras_nlp.models.BertPreprocessor(vocabulary=vocabulary)
        tokenizer("The quick brown fox.")
