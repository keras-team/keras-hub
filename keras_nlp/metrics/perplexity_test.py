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
    def test_vars_after_initializing_class(self):
        perplexity = Perplexity()
        self.assertEqual(perplexity._aggregate_cross_entropy, 0.0)
        self.assertEqual(perplexity._number_of_samples, 0.0)

    def test_from_logits_without_masking(self):
        perplexity = Perplexity(from_logits=True)
        y_true = tf.constant([[1, 3, 0], [2, 1, 3]])
        y_pred = tf.constant(
            [
                [
                    [1.034, 4.797, 2.82, 1.154],
                    [2.258, 1.591, 1.811, 1.852],
                    [3.216, 1.037, 0.3662, 2.7],
                ],
                [
                    [1.363, 1.726, 1.898, 2.582],
                    [1.163, 1.943, 1.761, 1.497],
                    [2.766, 1.453, 2.61, 2.805],
                ],
            ]
        )

        perplexity_val = perplexity(y_true, y_pred)
        self.assertAlmostEqual(perplexity_val.numpy(), 2.6542, delta=1e-3)

    def test_from_logits_with_sample_weight(self):
        perplexity = Perplexity(from_logits=True)

        y_true = tf.constant([[1, 3, 0], [2, 1, 3]])
        y_pred = tf.constant(
            [
                [
                    [1.034, 4.797, 2.82, 1.154],
                    [2.258, 1.591, 1.811, 1.852],
                    [3.216, 1.037, 0.3662, 2.7],
                ],
                [
                    [1.363, 1.726, 1.898, 2.582],
                    [1.163, 1.943, 1.761, 1.497],
                    [2.766, 1.453, 2.61, 2.805],
                ],
            ]
        )
        sample_wt = tf.cast(y_true != 0, tf.int32)

        perplexity_val = perplexity(y_true, y_pred, sample_wt)
        self.assertAlmostEqual(perplexity_val.numpy(), 2.8789, delta=1e-3)

    def test_from_logits_with_pad_token_id(self):
        perplexity = Perplexity(from_logits=True, pad_token_id=0)

        y_true = tf.constant([[1, 3, 0], [2, 1, 3]])
        y_pred = tf.constant(
            [
                [
                    [1.034, 4.797, 2.82, 1.154],
                    [2.258, 1.591, 1.811, 1.852],
                    [3.216, 1.037, 0.3662, 2.7],
                ],
                [
                    [1.363, 1.726, 1.898, 2.582],
                    [1.163, 1.943, 1.761, 1.497],
                    [2.766, 1.453, 2.61, 2.805],
                ],
            ]
        )

        perplexity_val = perplexity(y_true, y_pred)
        self.assertAlmostEqual(perplexity_val.numpy(), 2.8789, delta=1e-3)

    def test_two_inputs_from_logits(self):
        perplexity = Perplexity(from_logits=True, pad_token_id=0)

        y_true_1 = tf.constant([[1, 3, 0], [2, 1, 3]])
        y_pred_1 = tf.constant(
            [
                [
                    [1.034, 4.797, 2.82, 1.154],
                    [2.258, 1.591, 1.811, 1.852],
                    [3.216, 1.037, 0.3662, 2.7],
                ],
                [
                    [1.363, 1.726, 1.898, 2.582],
                    [1.163, 1.943, 1.761, 1.497],
                    [2.766, 1.453, 2.61, 2.805],
                ],
            ]
        )

        perplexity_val = perplexity(y_true_1, y_pred_1)
        self.assertAlmostEqual(perplexity_val.numpy(), 2.8789, delta=1e-3)

        y_true_2 = tf.constant([[2, 0, 0], [1, 2, 3]])
        y_pred_2 = tf.constant(
            [
                [
                    [2.887, 0.885, 2.973, 2.582],
                    [0.3838, 2.629, 1.91, 1.802],
                    [0.2578, 1.081, 1.125, 2.773],
                ],
                [
                    [1.623, 2.784, 0.2109, 2.66],
                    [2.395, 2.01, 0.252, 1.828],
                    [0.4482, 2.629, 0.9697, 0.998],
                ],
            ]
        )
        perplexity_val = perplexity(y_true_2, y_pred_2)
        self.assertEqual(perplexity_val, 3.9998498)

    def test_from_probs_with_sample_weight(self):
        perplexity = Perplexity(from_logits=False)

        y_true = tf.constant([[1, 3, 0], [2, 1, 3]])
        y_pred = tf.constant(
            [
                [
                    [1.034, 4.797, 2.82, 1.154],
                    [2.258, 1.591, 1.811, 1.852],
                    [3.216, 1.037, 0.3662, 2.7],
                ],
                [
                    [1.363, 1.726, 1.898, 2.582],
                    [1.163, 1.943, 1.761, 1.497],
                    [2.766, 1.453, 2.61, 2.805],
                ],
            ]
        )
        y_prob = tf.nn.softmax(y_pred, axis=-1)

        sample_wt = tf.cast(y_true != 0, tf.int32)

        perplexity_val = perplexity(y_true, y_prob, sample_wt)
        self.assertAlmostEqual(perplexity_val, 2.8789, delta=1e-3)

    def test_from_probs_with_pad_token(self):
        perplexity = Perplexity(from_logits=False, pad_token_id=0)

        y_true = tf.constant([[1, 3, 0], [2, 1, 3]])
        y_pred = tf.constant(
            [
                [
                    [1.034, 4.797, 2.82, 1.154],
                    [2.258, 1.591, 1.811, 1.852],
                    [3.216, 1.037, 0.3662, 2.7],
                ],
                [
                    [1.363, 1.726, 1.898, 2.582],
                    [1.163, 1.943, 1.761, 1.497],
                    [2.766, 1.453, 2.61, 2.805],
                ],
            ]
        )
        y_prob = tf.nn.softmax(y_pred, axis=-1)

        perplexity_val = perplexity(y_true, y_prob)
        self.assertAlmostEqual(perplexity_val, 2.8789, delta=1e-3)

    def test_reset_state(self):
        y_true = tf.constant([[1, 3, 0], [2, 1, 3]])
        y_pred = tf.constant(
            [
                [
                    [1.034, 4.797, 2.82, 1.154],
                    [2.258, 1.591, 1.811, 1.852],
                    [3.216, 1.037, 0.3662, 2.7],
                ],
                [
                    [1.363, 1.726, 1.898, 2.582],
                    [1.163, 1.943, 1.761, 1.497],
                    [2.766, 1.453, 2.61, 2.805],
                ],
            ]
        )

        perplexity = Perplexity(from_logits=True, pad_token_id=0)

        perplexity.update_state(y_true, y_pred)
        self.assertNotEqual(perplexity._aggregate_cross_entropy, 0.0)
        self.assertNotEqual(perplexity._number_of_samples, 0.0)
        self.assertNotEqual(perplexity.result(), 0.0)

        perplexity.reset_state()
        self.assertEqual(perplexity._aggregate_cross_entropy, 0.0)
        self.assertEqual(perplexity._number_of_samples, 0.0)
        self.assertEqual(perplexity.result(), 0.0)

    def test_update_state(self):
        perplexity = Perplexity(from_logits=True, pad_token_id=0)

        y_true_1 = tf.constant([[1, 3, 0], [2, 1, 3]])
        y_pred_1 = tf.constant(
            [
                [
                    [1.034, 4.797, 2.82, 1.154],
                    [2.258, 1.591, 1.811, 1.852],
                    [3.216, 1.037, 0.3662, 2.7],
                ],
                [
                    [1.363, 1.726, 1.898, 2.582],
                    [1.163, 1.943, 1.761, 1.497],
                    [2.766, 1.453, 2.61, 2.805],
                ],
            ]
        )

        perplexity.update_state(y_true_1, y_pred_1)
        perplexity_val = perplexity.result()
        self.assertAlmostEqual(
            perplexity._aggregate_cross_entropy.numpy(), 2.1148, delta=1e-3
        )
        self.assertEqual(perplexity._number_of_samples, 2)
        self.assertAlmostEqual(perplexity_val.numpy(), 2.8789, delta=1e-3)

        y_true_2 = tf.constant([[2, 0, 0], [1, 2, 3]])
        y_pred_2 = tf.constant(
            [
                [
                    [2.887, 0.885, 2.973, 2.582],
                    [0.3838, 2.629, 1.91, 1.802],
                    [0.2578, 1.081, 1.125, 2.773],
                ],
                [
                    [1.623, 2.784, 0.2109, 2.66],
                    [2.395, 2.01, 0.252, 1.828],
                    [0.4482, 2.629, 0.9697, 0.998],
                ],
            ]
        )

        perplexity.update_state(y_true_2, y_pred_2)
        perplexity_val = perplexity.result()
        self.assertAlmostEqual(
            perplexity._aggregate_cross_entropy, 5.5450, delta=1e-3
        )
        self.assertEqual(perplexity._number_of_samples, 4)
        self.assertAlmostEqual(perplexity_val, 3.9998, delta=1e-3)

    def test_merge_state(self):
        perplexity_1 = Perplexity(from_logits=True, pad_token_id=0)
        perplexity_2 = Perplexity(from_logits=True, pad_token_id=0)

        y_true_1 = tf.constant([[1, 3, 0], [2, 1, 3]])
        y_pred_1 = tf.constant(
            [
                [
                    [1.034, 4.797, 2.82, 1.154],
                    [2.258, 1.591, 1.811, 1.852],
                    [3.216, 1.037, 0.3662, 2.7],
                ],
                [
                    [1.363, 1.726, 1.898, 2.582],
                    [1.163, 1.943, 1.761, 1.497],
                    [2.766, 1.453, 2.61, 2.805],
                ],
            ]
        )

        y_true_2 = tf.constant([[2, 0, 0], [1, 2, 3]])
        y_pred_2 = tf.constant(
            [
                [
                    [2.887, 0.885, 2.973, 2.582],
                    [0.3838, 2.629, 1.91, 1.802],
                    [0.2578, 1.081, 1.125, 2.773],
                ],
                [
                    [1.623, 2.784, 0.2109, 2.66],
                    [2.395, 2.01, 0.252, 1.828],
                    [0.4482, 2.629, 0.9697, 0.998],
                ],
            ]
        )

        y_true_3 = tf.constant([[1, 3, 1], [2, 2, 1]])
        y_pred_3 = tf.constant(
            [
                [
                    [0.7383, 0.882, 0.7295, 2.64],
                    [0.867, 1.588, 2.291, 0.967],
                    [0.0908, 1.453, 1.5, 0.0703],
                ],
                [
                    [2.783, 0.9785, 2.664, 0.507],
                    [0.741, 1.535, 2.16, 2.531],
                    [2.863, 1.591, 1.403, 0.885],
                ],
            ]
        )

        perplexity_1.update_state(y_true_1, y_pred_1)
        perplexity_1.update_state(y_true_2, y_pred_2)
        self.assertAlmostEqual(
            perplexity_1._aggregate_cross_entropy.numpy(), 5.5450, delta=1e-3
        )

        perplexity_2.update_state(y_true_3, y_pred_3)
        self.assertAlmostEqual(
            perplexity_2._aggregate_cross_entropy.numpy(), 2.9769, delta=1e-3
        )

        merged_perplexity = Perplexity(from_logits=True, pad_token_id=0)
        merged_perplexity.merge_state([perplexity_1, perplexity_2])

        # _aggregate_cross_entropy value should be equal to the sum of the
        # _aggregate_cross_entropy values of the states.
        self.assertAlmostEqual(
            merged_perplexity._aggregate_cross_entropy.numpy(),
            8.521942,
            delta=1e-3,
        )
        self.assertEqual(merged_perplexity._number_of_samples, 6)
        self.assertAlmostEqual(
            merged_perplexity.result().numpy(), 4.1385, delta=1e-3
        )
