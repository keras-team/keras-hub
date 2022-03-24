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

import tensorflow as tf
from tensorflow import keras

from keras_nlp.preprocessing.lm_mask import LMMask


class LMMaskTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.VOCAB = [
            "[UNK]",
            "[MASK]",
            "[RANDOM]",
            "[CLS]",
            "[SEP]",
            "do",
            "you",
            "like",
            "machine",
            "learning",
            "welcome",
            "to",
            "keras",
        ]
        self.mask_token_id = self.VOCAB.index("[MASK]")

    def test_mask_ragged_tensor(self):
        lm_mask = LMMask(vocabulary=self.VOCAB)
        inputs = tf.ragged.constant([[5, 3, 2], [1, 2, 3, 4, 5]])
        masked_input_ids, masked_positions, masked_ids = lm_mask(
            inputs,
            lm_selection_rate=0.5,
            max_selections=5,
            mask_token_rate=1,
            random_token_rate=0,
        )
        self.assertAllEqual(masked_input_ids.shape, inputs.shape)
        self.assertAllEqual(
            masked_positions.row_lengths(), masked_ids.row_lengths()
        )

        # Test all selected tokens are correctly masked.
        masked_values = tf.gather(
            masked_input_ids,
            masked_positions,
            batch_dims=1,
        )
        self.assertEqual(tf.reduce_mean(masked_values), self.mask_token_id)

    def test_mask_tensor(self):
        lm_mask = LMMask(vocabulary=self.VOCAB)
        inputs = tf.random.uniform(
            shape=[5, 10],
            maxval=len(self.VOCAB),
            dtype=tf.int32,
        )
        masked_input_ids, masked_positions, masked_ids = lm_mask(
            inputs,
            lm_selection_rate=0.5,
            max_selections=5,
            mask_token_rate=1,
            random_token_rate=0,
        )
        self.assertAllEqual(masked_input_ids.shape, inputs.shape)
        self.assertAllEqual(
            masked_positions.row_lengths(), masked_ids.row_lengths()
        )
        # Test all selected tokens are correctly masked.
        masked_values = tf.gather(
            masked_input_ids,
            masked_positions,
            batch_dims=1,
        )
        self.assertEqual(tf.reduce_mean(masked_values), self.mask_token_id)

    def test_number_of_masked_position_as_expected(self):
        lm_mask = LMMask(vocabulary=self.VOCAB)
        inputs = tf.ragged.constant(
            [[0, 1, 2], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4]]
        )
        lm_selection_rate = 0.5
        max_selections = 5
        _, masked_positions, _ = lm_mask(
            inputs,
            lm_selection_rate=lm_selection_rate,
            max_selections=max_selections,
        )

        expected_number_of_masked_tokens = tf.cast(
            tf.math.minimum(
                tf.math.ceil(
                    tf.cast(inputs.row_lengths(), dtype=tf.float32)
                    * lm_selection_rate,
                ),
                max_selections,
            ),
            dtype=tf.int64,
        )
        self.assertAllEqual(
            masked_positions.row_lengths(), expected_number_of_masked_tokens
        )

        # Cap the number of masked tokens at 0, so we can test if
        # max_selections takes effect.
        max_selections = 0
        _, masked_positions, _ = lm_mask(
            inputs,
            lm_selection_rate=lm_selection_rate,
            max_selections=max_selections,
        )
        self.assertAllEqual(
            masked_positions.row_lengths(),
            tf.zeros(shape=[inputs.shape[0]], dtype=tf.int64),
        )

    def test_apply_random_token_not_mask(self):
        lm_mask = LMMask(vocabulary=self.VOCAB)
        inputs = tf.random.uniform(
            shape=[5, 10],
            maxval=len(self.VOCAB),
            dtype=tf.int32,
        )
        masked_input_ids, masked_positions, masked_ids = lm_mask(
            inputs,
            lm_selection_rate=0.5,
            max_selections=5,
            mask_token_rate=0,
            random_token_rate=1,
        )
        self.assertAllEqual(masked_input_ids.shape, inputs.shape)
        self.assertAllEqual(
            masked_positions.row_lengths(), masked_ids.row_lengths()
        )
        masked_values = tf.gather(
            masked_input_ids,
            masked_positions,
            batch_dims=1,
        )
        # Verify that selected tokens are replaced by random tokens.
        self.assertNotEqual(tf.reduce_mean(masked_values), self.mask_token_id)

    def test_invalid_mask_token(self):
        with self.assertRaisesRegex(KeyError, "Mask token *"):
            _ = LMMask(vocabulary=self.VOCAB, mask_token="OHHHHH")

    def test_unselectable_tokens(self):
        unselectable_tokens = [self.VOCAB[-1], self.VOCAB[-2]]
        lm_mask = LMMask(
            vocabulary=self.VOCAB, unselectable_tokens=unselectable_tokens
        )
        inputs = [[len(self.VOCAB) - 1, len(self.VOCAB) - 2]]
        _, masked_positions, _ = lm_mask(
            inputs,
            lm_selection_rate=1,
            max_selections=5,
            mask_token_rate=1,
            random_token_rate=0,
        )
        # Verify that no token is masked out.
        self.assertAllEqual(
            masked_positions.row_lengths(),
            tf.zeros(shape=[len(inputs)], dtype=tf.int64),
        )

    def test_config(self):
        unselectable_tokens = [self.VOCAB[-1], self.VOCAB[-2]]
        lm_mask = LMMask(
            vocabulary=self.VOCAB, unselectable_tokens=unselectable_tokens
        )
        config = lm_mask.get_config()
        expected_config = {
            "vocabulary": self.VOCAB,
            "unselectable_tokens": unselectable_tokens,
        }
        self.assertEqual(config, config | expected_config)

        # Test cloned lm_mask can be run.
        cloned_lm_mask = LMMask.from_config(config)
        inputs = tf.ragged.constant(
            [[0, 1, 2], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4]]
        )
        cloned_lm_mask(inputs, lm_selection_rate=0.5, max_selections=5)

    def test_save_and_load(self):
        unselectable_tokens = [self.VOCAB[-1], self.VOCAB[-2]]
        lm_mask = LMMask(
            vocabulary=self.VOCAB, unselectable_tokens=unselectable_tokens
        )
        inputs = keras.Input(shape=[None], ragged=True, dtype=tf.int64)
        outputs = lm_mask(inputs, lm_selection_rate=0.5, max_selections=5)
        model = keras.Model(inputs, outputs)
        inputs_data = tf.ragged.constant(
            [[0, 1, 2], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4]]
        )
        model(inputs_data)
        model.save(self.get_temp_dir())
        restored_model = keras.models.load_model(self.get_temp_dir())
        masked_input_ids, masked_positions, masked_ids = restored_model(
            inputs_data
        )

        self.assertAllEqual(masked_input_ids.shape, inputs_data.shape)
        self.assertAllEqual(
            masked_positions.row_lengths(), masked_ids.row_lengths()
        )
