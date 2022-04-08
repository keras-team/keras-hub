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

from keras_nlp.preprocessing.mlm_mask_generator import MLMMaskGenerator


class MLMMaskGeneratorTest(tf.test.TestCase):
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
        mlm_masker = MLMMaskGenerator(
            vocabulary_size=self.vocabulary_size,
            mask_selection_rate=0.5,
            max_selections=5,
            mask_token_id=self.mask_token_id,
            mask_token_rate=1,
            random_token_rate=0,
            output_dense_mask_positions=False,
        )
        inputs = tf.ragged.constant([[5, 3, 2], [1, 2, 3, 4, 5]])
        outputs = mlm_masker(inputs)
        masked_input_ids, masked_positions, masked_ids = (
            outputs["masked_input_ids"],
            outputs["masked_positions"],
            outputs["masked_ids"],
        )
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
        mlm_masker = MLMMaskGenerator(
            vocabulary_size=self.vocabulary_size,
            mask_selection_rate=0.5,
            max_selections=5,
            mask_token_id=self.mask_token_id,
            mask_token_rate=1,
            random_token_rate=0,
            output_dense_mask_positions=False,
        )
        inputs = tf.random.uniform(
            shape=[5, 10],
            maxval=len(self.VOCAB),
            dtype=tf.int32,
        )
        outputs = mlm_masker(inputs)
        masked_input_ids, masked_positions, masked_ids = (
            outputs["masked_input_ids"],
            outputs["masked_positions"],
            outputs["masked_ids"],
        )
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

    def test_mask_1d_input(self):
        mlm_masker = MLMMaskGenerator(
            vocabulary_size=self.vocabulary_size,
            mask_selection_rate=0.5,
            max_selections=5,
            mask_token_id=self.mask_token_id,
            mask_token_rate=1,
            random_token_rate=0,
        )
        inputs = tf.constant([1, 2, 3, 4, 5])
        outputs = mlm_masker(inputs)
        self.assertAllEqual(outputs["masked_input_ids"].shape, inputs.shape)

    def test_number_of_masked_position_as_expected(self):
        mask_selection_rate = 0.5
        max_selections = 5
        mlm_masker = MLMMaskGenerator(
            vocabulary_size=self.vocabulary_size,
            mask_selection_rate=mask_selection_rate,
            max_selections=max_selections,
            output_dense_mask_positions=False,
        )
        inputs = tf.ragged.constant(
            [[0, 1, 2], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4]]
        )
        outputs = mlm_masker(inputs)
        expected_number_of_masked_tokens = tf.cast(
            tf.math.minimum(
                tf.math.ceil(
                    tf.cast(inputs.row_lengths(), dtype=tf.float32)
                    * mask_selection_rate,
                ),
                max_selections,
            ),
            dtype=tf.int64,
        )
        self.assertAllEqual(
            outputs["masked_positions"].row_lengths(),
            expected_number_of_masked_tokens,
        )

        # Cap the number of masked tokens at 0, so we can test if
        # max_selections takes effect.
        max_selections = 0
        mlm_masker = MLMMaskGenerator(
            vocabulary_size=self.vocabulary_size,
            mask_selection_rate=mask_selection_rate,
            max_selections=max_selections,
            output_dense_mask_positions=False,
        )
        outputs = mlm_masker(inputs)
        self.assertAllEqual(
            outputs["masked_positions"].row_lengths(),
            tf.zeros(shape=[inputs.shape[0]], dtype=tf.int64),
        )

    def test_apply_random_token_not_mask(self):
        mlm_masker = MLMMaskGenerator(
            vocabulary_size=self.vocabulary_size,
            mask_selection_rate=0.5,
            max_selections=5,
            mask_token_rate=0,
            random_token_rate=1,
            output_dense_mask_positions=False,
        )
        inputs = tf.random.uniform(
            shape=[5, 10],
            maxval=len(self.VOCAB),
            dtype=tf.int32,
        )
        outputs = mlm_masker(inputs)
        masked_input_ids, masked_positions, masked_ids = (
            outputs["masked_input_ids"],
            outputs["masked_positions"],
            outputs["masked_ids"],
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
        with self.assertRaisesRegex(ValueError, "Mask token id should be*"):
            _ = MLMMaskGenerator(
                vocabulary_size=self.vocabulary_size,
                mask_selection_rate=0.5,
                max_selections=5,
                mask_token_id=self.vocabulary_size,
            )

    def test_unselectable_tokens(self):
        unselectable_token_ids = [
            self.vocabulary_size - 1,
            self.vocabulary_size - 2,
        ]
        mlm_masker = MLMMaskGenerator(
            vocabulary_size=self.vocabulary_size,
            mask_selection_rate=1,
            max_selections=5,
            unselectable_token_ids=unselectable_token_ids,
            mask_token_rate=1,
            random_token_rate=0,
            output_dense_mask_positions=False,
        )
        inputs = [unselectable_token_ids]
        outputs = mlm_masker(inputs)
        # Verify that no token is masked out.
        self.assertAllEqual(
            outputs["masked_positions"].row_lengths(),
            tf.zeros(shape=[len(inputs)], dtype=tf.int64),
        )

    def test_config(self):
        unselectable_token_ids = [
            self.vocabulary_size - 1,
            self.vocabulary_size - 2,
        ]
        mlm_masker = MLMMaskGenerator(
            vocabulary_size=self.vocabulary_size,
            mask_selection_rate=0.5,
            max_selections=5,
            unselectable_token_ids=unselectable_token_ids,
        )
        config = mlm_masker.get_config()
        expected_config = {
            "vocabulary_size": self.vocabulary_size,
            "unselectable_token_ids": unselectable_token_ids,
        }
        self.assertDictContainsSubset(expected_config, config)

        # Test cloned mlm_masker can be run.
        cloned_mlm_masker = MLMMaskGenerator.from_config(config)
        inputs = tf.ragged.constant(
            [[0, 1, 2], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4]]
        )
        cloned_mlm_masker(inputs)
