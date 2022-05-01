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

from keras_nlp.layers.mlm_mask_generator import MLMMaskGenerator


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
            mask_selection_length=5,
            mask_token_id=self.mask_token_id,
            mask_token_rate=1,
            random_token_rate=0,
        )
        inputs = tf.ragged.constant([[5, 3, 2], [1, 2, 3, 4, 5]])
        outputs = mlm_masker(inputs)
        tokens, mask_positions, mask_ids = (
            outputs["tokens"],
            outputs["mask_positions"],
            outputs["mask_ids"],
        )
        self.assertEqual(type(tokens), type(inputs))
        self.assertAllEqual(tokens.shape, inputs.shape)
        self.assertAllEqual(mask_positions.shape, mask_ids.shape)

        # Test all selected tokens are correctly masked.
        masked_values = tf.gather(
            tokens,
            mask_positions,
            batch_dims=1,
        )
        self.assertEqual(tf.reduce_mean(masked_values), self.mask_token_id)

    def test_mask_tensor(self):
        mlm_masker = MLMMaskGenerator(
            vocabulary_size=self.vocabulary_size,
            mask_selection_rate=0.5,
            mask_selection_length=5,
            mask_token_id=self.mask_token_id,
            mask_token_rate=1,
            random_token_rate=0,
        )
        inputs = tf.random.uniform(
            shape=[5, 10],
            maxval=len(self.VOCAB),
            dtype=tf.int32,
        )
        outputs = mlm_masker(inputs)
        tokens, mask_positions, mask_ids = (
            outputs["tokens"],
            outputs["mask_positions"],
            outputs["mask_ids"],
        )
        self.assertEqual(type(tokens), type(inputs))
        self.assertAllEqual(tokens.shape, inputs.shape)
        self.assertAllEqual(mask_positions.shape, mask_ids.shape)
        # Test all selected tokens are correctly masked.
        masked_values = tf.gather(
            tokens,
            mask_positions,
            batch_dims=1,
        )
        self.assertEqual(tf.reduce_mean(masked_values), self.mask_token_id)

    def test_mask_1d_input(self):
        mlm_masker = MLMMaskGenerator(
            vocabulary_size=self.vocabulary_size,
            mask_selection_rate=0.5,
            mask_selection_length=5,
            mask_token_id=self.mask_token_id,
            mask_token_rate=1,
            random_token_rate=0,
        )
        inputs = tf.constant([1, 2, 3, 4, 5])
        outputs = mlm_masker(inputs)
        self.assertAllEqual(outputs["tokens"].shape, inputs.shape)

    def test_number_of_masked_position_as_expected(self):
        mask_selection_rate = 0.5
        mask_selection_length = 5
        mlm_masker = MLMMaskGenerator(
            vocabulary_size=self.vocabulary_size,
            mask_selection_rate=mask_selection_rate,
            mask_token_id=self.mask_token_id,
            unselectable_token_ids=None,
        )
        inputs = tf.ragged.constant(
            [[0, 1, 2], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4]]
        )
        outputs = mlm_masker(inputs)
        expected_number_of_masked_tokens = tf.cast(
            tf.math.ceil(
                tf.cast(inputs.row_lengths(), dtype=tf.float32)
                * mask_selection_rate,
            ),
            dtype=tf.int64,
        )

        self.assertAllEqual(
            outputs["mask_positions"].row_lengths(),
            expected_number_of_masked_tokens,
        )

        # Cap the number of masked tokens at 0, so we can test if
        # mask_selection_length takes effect.
        mask_selection_length = 0
        mlm_masker = MLMMaskGenerator(
            vocabulary_size=self.vocabulary_size,
            mask_selection_rate=mask_selection_rate,
            mask_token_id=self.mask_token_id,
            mask_selection_length=mask_selection_length,
        )
        outputs = mlm_masker(inputs)
        self.assertEqual(tf.reduce_sum(outputs["mask_positions"]), 0)

    def test_apply_random_token_not_mask(self):
        mlm_masker = MLMMaskGenerator(
            vocabulary_size=self.vocabulary_size,
            mask_selection_rate=0.5,
            mask_token_id=self.mask_token_id,
            mask_token_rate=0,
            random_token_rate=1,
        )
        inputs = tf.random.uniform(
            shape=[5, 10],
            maxval=len(self.VOCAB),
            dtype=tf.int32,
        )
        outputs = mlm_masker(inputs)
        tokens, mask_positions, mask_ids = (
            outputs["tokens"],
            outputs["mask_positions"],
            outputs["mask_ids"],
        )
        self.assertAllEqual(tokens.shape, inputs.shape)
        self.assertAllEqual(
            mask_positions.row_lengths(), mask_ids.row_lengths()
        )
        masked_values = tf.gather(
            tokens,
            mask_positions,
            batch_dims=1,
        )
        # Verify that selected tokens are replaced by random tokens.
        self.assertNotEqual(tf.reduce_mean(masked_values), self.mask_token_id)

    def test_invalid_mask_token(self):
        with self.assertRaisesRegex(ValueError, "Mask token id should be*"):
            _ = MLMMaskGenerator(
                vocabulary_size=self.vocabulary_size,
                mask_selection_rate=0.5,
                mask_token_id=self.vocabulary_size,
                mask_selection_length=5,
            )

    def test_unselectable_tokens(self):
        unselectable_token_ids = [
            self.vocabulary_size - 1,
            self.vocabulary_size - 2,
        ]
        mlm_masker = MLMMaskGenerator(
            vocabulary_size=self.vocabulary_size,
            mask_selection_rate=1,
            mask_token_id=self.mask_token_id,
            mask_selection_length=5,
            unselectable_token_ids=unselectable_token_ids,
            mask_token_rate=1,
            random_token_rate=0,
        )
        inputs = tf.convert_to_tensor([unselectable_token_ids])
        outputs = mlm_masker(inputs)
        # Verify that no token is masked out.
        self.assertEqual(tf.reduce_sum(outputs["mask_positions"]), 0)

    def test_config(self):
        unselectable_token_ids = [
            self.vocabulary_size - 1,
            self.vocabulary_size - 2,
        ]
        mlm_masker = MLMMaskGenerator(
            vocabulary_size=self.vocabulary_size,
            mask_selection_rate=0.5,
            mask_token_id=self.mask_token_id,
            mask_selection_length=5,
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

    def test_graph_mode_execution(self):
        mlm_masker = MLMMaskGenerator(
            vocabulary_size=self.vocabulary_size,
            mask_selection_rate=0.5,
            mask_token_id=self.mask_token_id,
            mask_selection_length=5,
        )

        @tf.function
        def masker(inputs):
            return mlm_masker(inputs)

        masker(tf.constant([1, 2, 3]))
        masker(tf.constant([[1, 2, 3], [1, 2, 3]]))
        masker(tf.ragged.constant([[3, 5, 7, 7], [4, 6, 7, 5]]))

    def test_with_tf_data(self):
        ds = tf.data.Dataset.from_tensor_slices(
            tf.ones((100, 10), dtype="int32")
        )
        mlm_masker = MLMMaskGenerator(
            vocabulary_size=self.vocabulary_size,
            mask_selection_rate=0.5,
            mask_token_id=self.mask_token_id,
            mask_selection_length=5,
        )
        batch_first = ds.batch(8).map(mlm_masker)
        batch_second = ds.map(mlm_masker).batch(8)
        self.assertEqual(
            batch_first.take(1).get_single_element()["tokens"].shape,
            batch_second.take(1).get_single_element()["tokens"].shape,
        )
