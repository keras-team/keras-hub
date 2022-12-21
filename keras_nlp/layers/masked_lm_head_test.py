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
"""Tests for Transformer Encoder."""

import os

import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers import masked_lm_head


class MaskedLMHeadTest(tf.test.TestCase):
    def test_valid_call(self):
        head = masked_lm_head.MaskedLMHead(
            vocabulary_size=100,
            activation="softmax",
        )
        encoded_tokens = keras.Input(shape=(10, 16))
        positions = keras.Input(shape=(5,), dtype="int32")
        outputs = head(encoded_tokens, mask_positions=positions)
        model = keras.Model((encoded_tokens, positions), outputs)

        token_data = tf.random.uniform(shape=(4, 10, 16))
        position_data = tf.random.uniform(
            shape=(4, 5), maxval=10, dtype="int32"
        )
        model((token_data, position_data))

    def test_valid_call_with_embedding_weights(self):
        embedding = keras.layers.Embedding(100, 16)
        embedding.build((4, 10))
        head = masked_lm_head.MaskedLMHead(
            vocabulary_size=100,
            embedding_weights=embedding.embeddings,
            activation="softmax",
        )
        encoded_tokens = keras.Input(shape=(10, 16))
        positions = keras.Input(shape=(5,), dtype="int32")
        outputs = head(encoded_tokens, mask_positions=positions)
        model = keras.Model((encoded_tokens, positions), outputs)
        token_data = tf.random.uniform(shape=(4, 10, 16))
        position_data = tf.random.uniform(
            shape=(4, 5), maxval=10, dtype="int32"
        )
        model((token_data, position_data))

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
        outputs = head(encoded_tokens, mask_positions=positions)
        model = keras.Model((encoded_tokens, positions), outputs)

        token_data = tf.random.uniform(shape=(4, 10, 16))
        position_data = tf.random.uniform(
            shape=(4, 5), maxval=10, dtype="int32"
        )
        label_data = tf.random.uniform(shape=(4, 5), maxval=100, dtype="int32")

        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        optimizer = keras.optimizers.Adam()
        with tf.GradientTape() as tape:
            model((token_data, position_data))
            pred = model((token_data, position_data))
            loss = loss_fn(label_data, pred)
        grad = tape.gradient(loss, model.trainable_variables)
        self.assertGreater(len(grad), 1)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

    def test_checkpointing(self):
        head1 = masked_lm_head.MaskedLMHead(
            vocabulary_size=100,
            activation="softmax",
        )
        head2 = masked_lm_head.MaskedLMHead(
            vocabulary_size=100,
            activation="softmax",
        )
        token_data = tf.random.uniform(shape=(4, 10, 16))
        position_data = tf.random.uniform(
            shape=(4, 5), maxval=10, dtype="int32"
        )
        # The weights of head1 and head2 are different.
        head1_output = head1(token_data, mask_positions=position_data)
        head2_output = head2(token_data, mask_positions=position_data)
        self.assertNotAllClose(head1_output, head2_output)

        checkpoint = tf.train.Checkpoint(head1)
        checkpoint2 = tf.train.Checkpoint(head2)
        save_path = checkpoint.save(self.get_temp_dir())
        checkpoint2.restore(save_path)

        head1_output = head1(token_data, mask_positions=position_data)
        head2_output = head2(token_data, mask_positions=position_data)
        self.assertAllClose(head1_output, head2_output)

    def test_saving_model(self):
        head = masked_lm_head.MaskedLMHead(
            vocabulary_size=100,
            activation="softmax",
        )
        encoded_tokens = keras.Input(shape=(10, 16))
        positions = keras.Input(shape=(5,), dtype="int32")
        outputs = head(encoded_tokens, mask_positions=positions)
        model = keras.Model((encoded_tokens, positions), outputs)

        token_data = tf.random.uniform(shape=(4, 10, 16))
        position_data = tf.random.uniform(
            shape=(4, 5), maxval=10, dtype="int32"
        )
        model_output = model((token_data, position_data))
        save_path = os.path.join(self.get_temp_dir(), "model")
        model.save(save_path)
        restored = keras.models.load_model(save_path)

        restored_output = restored((token_data, position_data))
        self.assertAllClose(model_output, restored_output)
