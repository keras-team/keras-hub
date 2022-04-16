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

"""Tests for RougeL."""

import tensorflow as tf

from keras_nlp.metrics import RougeL


class RougeLTest(tf.test.TestCase):
    def test_vars_after_initializing_class(self):
        rouge_l = RougeL()
        self.assertEqual(rouge_l.result().numpy(), 0.0)

    def test_without_mask_token_ids(self):
        rouge_l = RougeL()
        y_true = tf.constant([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]], dtype=tf.int32)
        y_pred = tf.constant([[1, 2, 3, 2, 5], [5, 6, 8, 8, 8]], dtype=tf.int32)

        rouge_l_val = rouge_l(y_true, y_pred)
        self.assertAlmostEqual(rouge_l_val.numpy(), 0.70, delta=1e-3)

    def test_with_mask_token_ids(self):
        rouge_l = RougeL(mask_token_ids=[0, 1])
        y_true = tf.constant(
            [[1, 2, 3, 4, 0, 0], [1, 5, 6, 0, 0, 0]], dtype=tf.int32
        )
        y_pred = tf.constant(
            [[1, 3, 2, 4, 4, 4], [1, 5, 6, 0, 2, 2]], dtype=tf.int32
        )

        rouge_l_val = rouge_l(y_true, y_pred)
        self.assertAlmostEqual(rouge_l_val.numpy(), 0.5833, delta=1e-3)

    def test_ragged_input_without_mask_token_ids(self):
        rouge_l = RougeL()
        y_true = tf.ragged.constant(
            [[3, 4, 5], [5, 6, 7, 8, 9]], dtype=tf.int32
        )
        y_pred = tf.ragged.constant([[1, 4, 3, 2, 5], [5, 6]], dtype=tf.int32)

        rouge_l_val = rouge_l(y_true, y_pred)
        self.assertAlmostEqual(rouge_l_val.numpy(), 0.5357, delta=1e-3)

    def test_ragged_input_with_mask_token_ids(self):
        rouge_l = RougeL(mask_token_ids=[0, 1])
        y_true = tf.ragged.constant(
            [[1, 2, 3, 4], [1, 5, 6, 0, 0]], dtype=tf.int32
        )
        y_pred = tf.ragged.constant(
            [[1, 3, 2, 4, 4, 4], [5, 6, 0, 2, 2]], dtype=tf.int32
        )

        rouge_l_val = rouge_l(y_true, y_pred)
        self.assertAlmostEqual(rouge_l_val.numpy(), 0.583, delta=1e-3)

    def test_precision(self):
        rouge_l = RougeL(mask_token_ids=[0, 1], metric_type="precision")
        y_true = tf.constant(
            [[1, 2, 3, 4, 0, 0], [1, 5, 6, 0, 0, 0]], dtype=tf.int32
        )
        y_pred = tf.constant(
            [[1, 3, 2, 4, 4, 4], [1, 5, 6, 0, 2, 2]], dtype=tf.int32
        )

        rouge_l_val = rouge_l(y_true, y_pred)
        self.assertAlmostEqual(rouge_l_val.numpy(), 0.45, delta=1e-3)

    def test_recall(self):
        rouge_l = RougeL(mask_token_ids=[0, 1], metric_type="recall")
        y_true = tf.constant(
            [[1, 2, 3, 4, 0, 0], [1, 5, 6, 0, 0, 0]], dtype=tf.int32
        )
        y_pred = tf.constant(
            [[1, 3, 2, 4, 4, 4], [1, 5, 6, 0, 2, 2]], dtype=tf.int32
        )

        rouge_l_val = rouge_l(y_true, y_pred)
        self.assertAlmostEqual(rouge_l_val.numpy(), 0.8333, delta=1e-3)

    def test_output_with_alpha(self):
        rouge_l = RougeL(alpha=0.7)
        y_true = tf.constant([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]], dtype=tf.int32)
        y_pred = tf.constant([[1, 2, 3, 2, 5], [5, 6, 8, 8, 8]], dtype=tf.int32)

        rouge_l_val = rouge_l(y_true, y_pred)
        self.assertAlmostEqual(rouge_l_val.numpy(), 0.70, delta=1e-3)

    def test_two_inputs_from_logits(self):
        rouge_l = RougeL(mask_token_ids=[0, 1])
        y_true_1 = tf.constant(
            [[1, 2, 3, 4, 0, 0], [1, 5, 6, 0, 0, 0]], dtype=tf.int32
        )
        y_pred_1 = tf.constant(
            [[1, 3, 2, 4, 4, 4], [1, 5, 6, 0, 2, 2]], dtype=tf.int32
        )

        rouge_l_val = rouge_l(y_true_1, y_pred_1)
        self.assertAlmostEqual(rouge_l_val.numpy(), 0.5833, delta=1e-3)

        y_true_2 = tf.ragged.constant(
            [[1, 2, 3, 4], [1, 5, 6, 7, 8]], dtype=tf.int32
        )
        y_pred_2 = tf.ragged.constant(
            [[1, 3, 2, 2, 3, 4], [5, 6, 7, 8, 2]], dtype=tf.int32
        )

        rouge_l_val = rouge_l(y_true_2, y_pred_2)
        self.assertAlmostEqual(rouge_l_val.numpy(), 0.7014, delta=1e-3)

    def test_reset_state(self):
        rouge_l = RougeL(mask_token_ids=[0, 1])
        y_true = tf.constant(
            [[1, 2, 3, 4, 0, 0], [1, 5, 6, 0, 0, 0]], dtype=tf.int32
        )
        y_pred = tf.constant(
            [[1, 3, 2, 4, 4, 4], [1, 5, 6, 0, 2, 2]], dtype=tf.int32
        )

        rouge_l.update_state(y_true, y_pred)
        self.assertNotEqual(rouge_l.result(), 0.0)

        rouge_l.reset_state()
        self.assertEqual(rouge_l.result(), 0.0)

    def test_update_state(self):
        rouge_l = RougeL(mask_token_ids=[0, 1])
        y_true_1 = tf.constant(
            [[1, 2, 3, 4, 0, 0], [1, 5, 6, 0, 0, 0]], dtype=tf.int32
        )
        y_pred_1 = tf.constant(
            [[1, 3, 2, 4, 4, 4], [1, 5, 6, 0, 2, 2]], dtype=tf.int32
        )

        rouge_l.update_state(y_true_1, y_pred_1)
        rouge_l_val = rouge_l.result()
        self.assertAlmostEqual(rouge_l_val.numpy(), 0.5833, delta=1e-3)

        y_true_2 = tf.ragged.constant(
            [[1, 2, 3, 4], [1, 5, 6, 7, 8]], dtype=tf.int32
        )
        y_pred_2 = tf.ragged.constant(
            [[1, 3, 2, 2, 3, 4], [5, 6, 7, 8, 2]], dtype=tf.int32
        )

        rouge_l.update_state(y_true_2, y_pred_2)
        rouge_l_val = rouge_l.result()
        self.assertAlmostEqual(rouge_l_val.numpy(), 0.7014, delta=1e-3)

    def test_merge_state(self):
        rouge_l_1 = RougeL(mask_token_ids=[0, 1])
        rouge_l_2 = RougeL(mask_token_ids=[0, 1])

        y_true_1 = tf.constant(
            [[1, 2, 3, 4, 0, 0], [1, 5, 6, 0, 0, 0]], dtype=tf.int32
        )
        y_pred_1 = tf.constant(
            [[1, 3, 2, 4, 4, 4], [1, 5, 6, 0, 2, 2]], dtype=tf.int32
        )

        y_true_2 = tf.ragged.constant(
            [[1, 2, 3, 4], [1, 5, 6, 7, 8]], dtype=tf.int32
        )
        y_pred_2 = tf.ragged.constant(
            [[1, 3, 2, 2, 3, 4], [5, 6, 7, 8, 2]], dtype=tf.int32
        )

        y_true_3 = tf.ragged.constant(
            [[9, 8, 7, 1], [10, 5, 1, 2, 3]], dtype=tf.int32
        )
        y_pred_3 = tf.ragged.constant(
            [[1, 2, 7, 9, 8, 0], [10, 1, 2]], dtype=tf.int32
        )

        rouge_l_1.update_state(y_true_1, y_pred_1)
        rouge_l_1.update_state(y_true_2, y_pred_2)
        self.assertAlmostEqual(rouge_l_1.result().numpy(), 0.7014, delta=1e-3)

        rouge_l_2.update_state(y_true_3, y_pred_3)
        self.assertAlmostEqual(rouge_l_2.result().numpy(), 0.6190, delta=1e-3)

        merged_rouge_l = RougeL(mask_token_ids=[0, 1])
        merged_rouge_l.merge_state([rouge_l_1, rouge_l_2])
        self.assertAlmostEqual(
            merged_rouge_l.result().numpy(), 0.6739, delta=1e-3
        )

    def test_get_config(self):
        rouge_l = RougeL(
            alpha=0.7,
            metric_type="precision",
            mask_token_ids=[0],
            dtype=tf.float32,
            name="rouge_l_test",
        )
        config = rouge_l.get_config()
        expected_config = {
            "alpha": 0.7,
            "metric_type": "precision",
            "mask_token_ids": [0],
            "dtype": tf.float32,
            "name": "rouge_l_test",
        }
        self.assertEqual(config, expected_config)
