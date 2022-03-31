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

from keras_nlp.preprocessing.mlm_masker import MaskedLanguageModelMasker


class MaskedLanguageModelMaskerTest(tf.test.TestCase):
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
        self.vocabulary_size = len(self.VOCAB)

    def test_mask_ragged_tensor(self):
        mlm_masker = MaskedLanguageModelMasker(
            vocabulary_size=self.vocabulary_size,
            lm_selection_rate=0.5,
            max_selections=5,
            mask_token_id=self.mask_token_id,
            mask_token_rate=1,
            random_token_rate=0,
        )
        inputs = tf.ragged.constant([[5, 3, 2], [1, 2, 3, 4, 5]])
        masked_input_ids, masked_positions, masked_ids = mlm_masker(inputs)
        self.assertEqual(type(masked_input_ids), type(inputs))
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
        mlm_masker = MaskedLanguageModelMasker(
            vocabulary_size=self.vocabulary_size,
            lm_selection_rate=0.5,
            max_selections=5,
            mask_token_id=self.mask_token_id,
            mask_token_rate=1,
            random_token_rate=0,
        )
        inputs = tf.random.uniform(
            shape=[5, 10],
            maxval=len(self.VOCAB),
            dtype=tf.int32,
        )
        masked_input_ids, masked_positions, masked_ids = mlm_masker(inputs)
        self.assertEqual(type(masked_input_ids), type(inputs))
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
        lm_selection_rate = 0.5
        max_selections = 5
        mlm_masker = MaskedLanguageModelMasker(
            vocabulary_size=self.vocabulary_size,
            lm_selection_rate=lm_selection_rate,
            max_selections=max_selections,
        )
        inputs = tf.ragged.constant(
            [[0, 1, 2], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4]]
        )
        _, masked_positions, _ = mlm_masker(inputs)

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
        mlm_masker = MaskedLanguageModelMasker(
            vocabulary_size=self.vocabulary_size,
            lm_selection_rate=lm_selection_rate,
            max_selections=max_selections,
        )
        _, masked_positions, _ = mlm_masker(inputs)
        self.assertAllEqual(
            masked_positions.row_lengths(),
            tf.zeros(shape=[inputs.shape[0]], dtype=tf.int64),
        )

    def test_apply_random_token_not_mask(self):
        mlm_masker = MaskedLanguageModelMasker(
            vocabulary_size=self.vocabulary_size,
            lm_selection_rate=0.5,
            max_selections=5,
            mask_token_rate=0,
            random_token_rate=1,
        )
        inputs = tf.random.uniform(
            shape=[5, 10],
            maxval=len(self.VOCAB),
            dtype=tf.int32,
        )
        masked_input_ids, masked_positions, masked_ids = mlm_masker(inputs)
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
        with self.assertRaisesRegex(ValueError, "Mask token id should be*"):
            _ = MaskedLanguageModelMasker(
                vocabulary_size=self.vocabulary_size,
                lm_selection_rate=0.5,
                max_selections=5,
                mask_token_id=self.vocabulary_size,
            )

    def test_unselectable_tokens(self):
        unselectable_token_ids = [
            self.vocabulary_size - 1,
            self.vocabulary_size - 2,
        ]
        mlm_masker = MaskedLanguageModelMasker(
            vocabulary_size=self.vocabulary_size,
            lm_selection_rate=1,
            max_selections=5,
            unselectable_token_ids=unselectable_token_ids,
            mask_token_rate=1,
            random_token_rate=0,
        )
        inputs = [unselectable_token_ids]
        _, masked_positions, _ = mlm_masker(inputs)
        # Verify that no token is masked out.
        self.assertAllEqual(
            masked_positions.row_lengths(),
            tf.zeros(shape=[len(inputs)], dtype=tf.int64),
        )

    def test_config(self):
        unselectable_token_ids = [
            self.vocabulary_size - 1,
            self.vocabulary_size - 2,
        ]
        mlm_masker = MaskedLanguageModelMasker(
            vocabulary_size=self.vocabulary_size,
            lm_selection_rate=0.5,
            max_selections=5,
            unselectable_token_ids=unselectable_token_ids,
        )
        config = mlm_masker.get_config()
        expected_config = {
            "vocabulary_size": self.vocabulary_size,
            "unselectable_token_ids": unselectable_token_ids,
        }
        self.assertEqual(config, config | expected_config)

        # Test cloned mlm_masker can be run.
        cloned_mlm_masker = MaskedLanguageModelMasker.from_config(config)
        inputs = tf.ragged.constant(
            [[0, 1, 2], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4]]
        )
        cloned_mlm_masker(inputs)

    def test_save_and_load(self):
        unselectable_token_ids = [
            self.vocabulary_size - 1,
            self.vocabulary_size - 2,
        ]
        mlm_masker = MaskedLanguageModelMasker(
            vocabulary_size=self.vocabulary_size,
            lm_selection_rate=0.5,
            max_selections=5,
            unselectable_token_ids=unselectable_token_ids,
        )
        inputs = keras.Input(shape=[None], ragged=True, dtype=tf.int64)
        outputs = mlm_masker(inputs)
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
