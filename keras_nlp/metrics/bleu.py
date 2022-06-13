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

"""BLEU score implementation based on `keras.metrics.Metric`."""

import tensorflow as tf
from tensorflow import keras

from keras_nlp.metrics.ngram_utils import get_ngram_count


class Bleu(keras.metrics.Metric):
    def __init__(
        self,
        max_order=2,
        smooth=False,
        dtype=None,
        name="bleu",
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)

        if not tf.as_dtype(self.dtype).is_floating:
            raise ValueError(
                "`dtype` must be a floating point type. "
                f"Received: dtype={dtype}"
            )

        self.max_order = max_order
        self.smooth = smooth

        self._bleu = self.add_weight(
            name="bleu",
            initializer="zeros",
            dtype=self.dtype,
        )
        self._number_of_samples = self.add_weight(
            name="number_of_samples", initializer="zeros", dtype=self.dtype
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        batch_size = tf.shape(y_true)[0]

        # Tokenise the strings (we will later replace this with a more
        # complicated tokeniser)
        y_true = tf.strings.split(y_true)
        y_pred = tf.strings.split(y_pred)

        agg_reference_len = tf.cast(0, dtype=self.dtype)
        agg_translation_len = tf.cast(0, dtype=self.dtype)
        p_log_sum = tf.cast(0, dtype=self.dtype)
        for idx in range(batch_size):
            reference = y_true[idx]
            translation = y_pred[idx]
            agg_reference_len += tf.cast(
                tf.shape(reference)[0], dtype=self.dtype
            )
            agg_translation_len += tf.cast(
                tf.shape(translation)[0], dtype=self.dtype
            )

        min_precision = tf.cast(1, dtype=self.dtype)
        for order in range(1, self.max_order + 1):
            matches = tf.cast(0, dtype=self.dtype)
            possible_matches = tf.cast(0, dtype=self.dtype)

            for idx in range(batch_size):
                reference = y_true[idx]
                translation = y_pred[idx]
                translation_len = tf.cast(
                    tf.shape(translation)[0], dtype=self.dtype
                )

                # Get n-grams and ngram count.
                reference_ngrams, reference_ngram_freq = get_ngram_count(
                    reference, order
                )
                translation_ngrams, translation_ngram_freq = get_ngram_count(
                    translation, order
                )

                # Get the intersection of the two ngram tensors.
                common_ngrams = tf.sets.intersection(
                    reference_ngrams[tf.newaxis, :],
                    translation_ngrams[tf.newaxis, :],
                ).values

                common_reference_ngram_freq = tf.gather(
                    reference_ngram_freq,
                    tf.argmax(
                        (reference_ngrams[:, None] == common_ngrams), axis=0
                    ),
                )
                common_translation_ngram_freq = tf.gather(
                    translation_ngram_freq,
                    tf.argmax(
                        (translation_ngrams[:, None] == common_ngrams), axis=0
                    ),
                )

                # Compute number of ngram matches.
                matches += tf.cast(
                    tf.reduce_sum(
                        tf.minimum(
                            common_reference_ngram_freq,
                            common_translation_ngram_freq,
                        )
                    ),
                    dtype=self.dtype,
                )
                if translation_len - order + 1 > 0:
                    possible_matches += translation_len

            if self.smooth:
                precision = (matches + tf.cast(1, dtype=self.dtype)) / (
                    possible_matches + tf.cast(1, dtype=self.dtype)
                )
            else:
                if possible_matches > 0:
                    precision = matches / possible_matches
                else:
                    precision = tf.cast(0, dtype=self.dtype)

            if precision > 0:
                p_log_sum += (
                    tf.cast(1, dtype=self.dtype)
                    / tf.cast(self.max_order, dtype=self.dtype)
                ) * tf.math.log(precision)
            min_precision = tf.minimum(min_precision, precision)

        if min_precision > 0:
            geo_mean = tf.exp(p_log_sum)
        else:
            geo_mean = tf.cast(0, dtype=self.dtype)

        # Compute the brevity penalty.
        ratio = agg_translation_len / agg_reference_len
        if ratio > 1:
            bp = tf.cast(1, dtype=self.dtype)
        else:
            bp = tf.exp(
                tf.cast(1, dtype=self.dtype)
                - tf.cast(1, dtype=self.dtype) / ratio
            )

        self._bleu.assign_add(geo_mean * bp)
        self._number_of_samples.assign_add(
            tf.cast(batch_size, dtype=tf.float32)
        )

    def result(self):
        if self._number_of_samples == 0:
            return 0.0
        bleu = self._bleu / self._number_of_samples

        return bleu

    def reset_state(self):
        self._bleu.assign(0.0)
        self._number_of_samples.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"max_order": self.max_order, "smooth": self.smooth})
        return config
