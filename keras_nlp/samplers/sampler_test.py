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
from keras_nlp.samplers.greedy_sampler import GreedySampler


class SamplerTest(tf.test.TestCase):
    def test_serialization(self):
        sampler = keras_nlp.samplers.GreedySampler()
        config = keras_nlp.samplers.serialize(sampler)
        expected_config = {
            "class_name": "keras_nlp>GreedySampler",
            "config": {
                "jit_compile": True,
            },
        }
        self.assertDictEqual(expected_config, config)

    def test_deserialization(self):
        # Test get from string.
        identifier = "greedy"
        sampler = keras_nlp.samplers.get(identifier)
        self.assertIsInstance(sampler, GreedySampler)

        # Test dict identifier.
        original_sampler = keras_nlp.samplers.GreedySampler(jit_compile=False)
        config = keras_nlp.samplers.serialize(original_sampler)
        restored_sampler = keras_nlp.samplers.get(config)
        self.assertDictEqual(
            keras_nlp.samplers.serialize(restored_sampler),
            keras_nlp.samplers.serialize(original_sampler),
        )

        # Test identifier is already a sampler instance.
        original_sampler = keras_nlp.samplers.GreedySampler(jit_compile=False)
        restored_sampler = keras_nlp.samplers.get(original_sampler)
        self.assertEqual(original_sampler, restored_sampler)
