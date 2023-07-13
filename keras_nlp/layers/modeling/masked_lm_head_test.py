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
"""Tests for Transformer Encoder."""

import os

from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.layers.modeling import masked_lm_head
from keras_nlp.tests.test_case import TestCase


class MaskedLMHeadTest(TestCase):
    def test_valid_call(self):
        head = masked_lm_head.MaskedLMHead(
            vocabulary_size=100,
            activation="softmax",
        )
        encoded_tokens = keras.Input(shape=(10, 16))
        positions = keras.Input(shape=(5,), dtype="int32")
        outputs = head(encoded_tokens, masked_positions=positions)
        model = keras.Model((encoded_tokens, positions), outputs)

        token_data = ops.random.uniform(shape=(4, 10, 16))
        position_data = ops.random.randint(minval=0, maxval=10, shape=(4, 5))
        model((token_data, position_data))

    def test_valid_call_with_embedding_weights(self):
        embedding = keras.layers.Embedding(100, 16)
        embedding.build((4, 10))
        head = masked_lm_head.MaskedLMHead(
            vocabulary_size=100,
            embedding_weights=embedding.embeddings,
            activation="softmax",
        )
        # Use a difference "hidden dim" for the model than "embedding dim", we
        # need to support this in the layer.
        sequence = keras.Input(shape=(10, 32))
        positions = keras.Input(shape=(5,), dtype="int32")
        outputs = head(sequence, masked_positions=positions)
        model = keras.Model((sequence, positions), outputs)
        sequence_data = ops.random.uniform(shape=(4, 10, 32))
        position_data = ops.random.randint(minval=0, maxval=10, shape=(4, 5))
        model((sequence_data, position_data))

    def test_get_config_and_from_config(self):
        head = masked_lm_head.MaskedLMHead(
            vocabulary_size=100,
            kernel_initializer="HeNormal",
            bias_initializer="Zeros",
            activation="softmax",
        )

        config = head.get_config()

        expected_params = {
            "vocabulary_size": 100,
            "kernel_initializer": keras.initializers.serialize(
                keras.initializers.HeNormal()
            ),
            "bias_initializer": keras.initializers.serialize(
                keras.initializers.Zeros()
            ),
            "activation": keras.activations.serialize(
                keras.activations.softmax
            ),
        }

        self.assertEqual(config, {**config, **expected_params})

        restored = masked_lm_head.MaskedLMHead.from_config(config)
        restored_config = restored.get_config()

        self.assertEqual(
            restored_config, {**restored_config, **expected_params}
        )
        self.assertEqual(restored_config, config)

    def test_value_error_when_neither_embedding_or_vocab_size_set(self):
        with self.assertRaises(ValueError):
            masked_lm_head.MaskedLMHead()

    def test_value_error_when_vocab_size_mismatch(self):
        embedding = keras.layers.Embedding(100, 16)
        embedding.build((4, 10))
        with self.assertRaises(ValueError):
            masked_lm_head.MaskedLMHead(
                vocabulary_size=101,
                embedding_weights=embedding.embeddings,
            )

    def test_one_train_step(self):
        head = masked_lm_head.MaskedLMHead(
            vocabulary_size=100,
        )
        encoded_tokens = keras.Input(shape=(10, 16))
        positions = keras.Input(shape=(5,), dtype="int32")
        outputs = head(encoded_tokens, masked_positions=positions)
        model = keras.Model((encoded_tokens, positions), outputs)

        token_data = ops.random.uniform(shape=(4, 10, 16))
        position_data = ops.random.randint(minval=0, maxval=10, shape=(4, 5))
        label_data = ops.random.randint(minval=0, maxval=2, shape=(4, 5, 1))

        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        optimizer = keras.optimizers.Adam()
        model.compile(loss=loss, optimizer=optimizer)
        loss = model.train_on_batch(x=(token_data, position_data), y=label_data)
        self.assertGreater(loss, 0)

    def test_saved_model(self):
        head = masked_lm_head.MaskedLMHead(
            vocabulary_size=100,
            activation="softmax",
        )
        encoded_tokens = keras.Input(shape=(10, 16))
        positions = keras.Input(shape=(5,), dtype="int32")
        outputs = head(encoded_tokens, masked_positions=positions)
        model = keras.Model((encoded_tokens, positions), outputs)

        token_data = ops.random.uniform(shape=(4, 10, 16))
        position_data = ops.random.randint(minval=0, maxval=10, shape=(4, 5))
        model_output = model((token_data, position_data))
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path, save_format="keras_v3")
        restored_model = keras.models.load_model(path)

        restored_output = restored_model((token_data, position_data))
        self.assertAllClose(model_output, restored_output)
