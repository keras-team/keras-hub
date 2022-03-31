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

"""Tests for Perplexity."""

import tensorflow as tf

from keras_nlp.metrics import Perplexity


class PerplexityTest(tf.test.TestCase):
    @classmethod
    def setUpClass(cls):
        tf.random.set_seed(42)

        cls.y_true_1 = tf.constant(
            [[1, 2, 3, 4, 5, 0, 0, 0], [6, 7, 8, 0, 0, 0, 0, 0]], dtype=tf.int32
        )
        cls.y_pred_1 = tf.random.uniform(
            shape=[2, 8, 10], maxval=1, dtype=tf.float32, seed=42
        )
        cls.y_prob_1 = tf.nn.softmax(cls.y_pred_1, axis=-1)
        cls.sample_wt_1 = tf.cast(cls.y_true_1 != 0, tf.int32)

        cls.y_true_2 = tf.constant(
            [[1, 9, 2, 2, 1, 0, 0, 0], [7, 5, 1, 8, 0, 0, 0, 0]], dtype=tf.int32
        )
        cls.y_pred_2 = tf.random.uniform(
            shape=[2, 8, 10], maxval=1, dtype=tf.float32, seed=42
        )
        cls.y_prob_2 = tf.nn.softmax(cls.y_pred_2, axis=-1)
        cls.sample_wt_2 = tf.cast(cls.y_true_2 != 0, tf.int32)

        cls.y_true_3 = tf.constant(
            [[3, 3, 6, 2, 7, 4, 5, 0], [9, 4, 1, 6, 5, 4, 0, 0]], dtype=tf.int32
        )
        cls.y_pred_3 = tf.random.uniform(
            shape=[2, 8, 10], maxval=1, dtype=tf.float32, seed=42
        )

    def test_output_after_initializing_class(self):
        perplexity = Perplexity()
        self.assertEqual(perplexity.aggregate_cross_entropy_loss, 0.0)
        self.assertEqual(perplexity.number_of_samples, 0.0)

    def test_output_on_two_inputs_from_logits_with_sample_weight(self):
        perplexity = Perplexity(from_logits=True)

        val1 = perplexity(self.y_true_1, self.y_pred_1, self.sample_wt_1)
        self.assertAlmostEqual(val1, 9.682761)

        val2 = perplexity(self.y_true_2, self.y_pred_2, self.sample_wt_2)
        self.assertAlmostEqual(val2, 10.067247)

    def test_output_on_two_inputs_from_logits_without_masking(self):
        perplexity = Perplexity(from_logits=True)

        val1 = perplexity(self.y_true_1, self.y_pred_1)
        self.assertAlmostEqual(val1, 10.599162)

        val2 = perplexity(self.y_true_2, self.y_pred_2)
        self.assertEqual(val2, 10.477932)

    def test_output_on_two_inputs_from_logits_with_pad_token(self):
        perplexity = Perplexity(from_logits=True, pad_token_id=0)

        val1 = perplexity(self.y_true_1, self.y_pred_1)
        self.assertAlmostEqual(val1, 9.682761)

        val2 = perplexity(self.y_true_2, self.y_pred_2)
        self.assertAlmostEqual(val2, 10.067247)

    def test_output_on_two_inputs_from_probabilities_with_sample_weight(self):
        perplexity = Perplexity(from_logits=False)

        val1 = perplexity(self.y_true_1, self.y_prob_1, self.sample_wt_1)
        self.assertAlmostEqual(val1, 9.682761)

        val2 = perplexity(self.y_true_2, self.y_prob_2, self.sample_wt_2)
        self.assertAlmostEqual(val2, 10.067247)

    def test_output_on_two_inputs_from_probabilities_with_pad_token(self):
        perplexity = Perplexity(from_logits=False, pad_token_id=0)

        val1 = perplexity(self.y_true_1, self.y_prob_1)
        self.assertAlmostEqual(val1, 9.682761)

        val2 = perplexity(self.y_true_2, self.y_prob_2)
        self.assertAlmostEqual(val2, 10.067247)

    def test_reset_state(self):
        perplexity = Perplexity(from_logits=True, pad_token_id=0)

        perplexity.update_state(self.y_true_1, self.y_pred_1)
        self.assertNotEqual(perplexity.aggregate_cross_entropy_loss, 0.0)
        self.assertNotEqual(perplexity.number_of_samples, 0.0)
        self.assertNotEqual(perplexity.result(), 0.0)

        perplexity.reset_state()
        self.assertEqual(perplexity.aggregate_cross_entropy_loss, 0.0)
        self.assertEqual(perplexity.number_of_samples, 0.0)
        self.assertEqual(perplexity.result(), 0.0)

    def test_update_state(self):
        perplexity = Perplexity(from_logits=True, pad_token_id=0)

        perplexity.update_state(self.y_true_1, self.y_pred_1)
        self.assertAlmostEqual(
            perplexity.aggregate_cross_entropy_loss, 4.54069424
        )
        self.assertEqual(perplexity.number_of_samples, 2)
        self.assertAlmostEqual(perplexity.result(), 9.682761)

        perplexity.update_state(self.y_true_2, self.y_pred_2)
        self.assertAlmostEqual(
            perplexity.aggregate_cross_entropy_loss, 9.23714924
        )
        self.assertEqual(perplexity.number_of_samples, 4)
        self.assertAlmostEqual(perplexity.result(), 10.067247)

    def test_merge_state(self):
        perplexity_1 = Perplexity(from_logits=True, pad_token_id=0)
        perplexity_2 = Perplexity(from_logits=True, pad_token_id=0)

        perplexity_1.update_state(self.y_true_1, self.y_pred_1)
        perplexity_1.update_state(self.y_true_2, self.y_pred_2)

        perplexity_2.update_state(self.y_true_3, self.y_pred_3)

        merged_perplexity = Perplexity(from_logits=True, pad_token_id=0)
        merged_perplexity.merge_state([perplexity_1, perplexity_2])

        self.assertAlmostEqual(
            merged_perplexity.aggregate_cross_entropy_loss, 14.1800919
        )
        self.assertEqual(merged_perplexity.number_of_samples, 6)
        self.assertAlmostEqual(merged_perplexity.result(), 10.626477)
