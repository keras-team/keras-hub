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
"""Tests for Sampler classes."""

import tensorflow as tf

import keras_nlp
from keras_nlp.samplers.top_k_sampler import TopKSampler


class SamplerTest(tf.test.TestCase):
    def test_serialization(self):
        sampler = TopKSampler(k=5)
        restored = keras_nlp.samplers.deserialize(
            keras_nlp.samplers.serialize(sampler)
        )
        self.assertDictEqual(sampler.get_config(), restored.get_config())

    def test_get(self):
        # Test get from string.
        identifier = "top_k"
        sampler = keras_nlp.samplers.get(identifier)
        self.assertIsInstance(sampler, TopKSampler)

        # Test dict identifier.
        original_sampler = keras_nlp.samplers.TopKSampler(k=7)
        config = keras_nlp.samplers.serialize(original_sampler)
        restored_sampler = keras_nlp.samplers.get(config)
        self.assertDictEqual(
            keras_nlp.samplers.serialize(restored_sampler),
            keras_nlp.samplers.serialize(original_sampler),
        )

        # Test identifier is already a sampler instance.
        original_sampler = keras_nlp.samplers.TopKSampler(k=7)
        restored_sampler = keras_nlp.samplers.get(original_sampler)
        self.assertEqual(original_sampler, restored_sampler)
