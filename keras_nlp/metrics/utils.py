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

"""Utility functions used by metrics classes."""

import tensorflow as tf


def get_ngrams(segment, order):
    """Given a tensor of tokens, finds n-grams.

    Args:
        segment: tf.Tensor. A one-dimensional tensor of token IDs.
        order: int. The order of the n-grams. For example, if the order is 2,
            we find bigrams.

    Returns:
        (ngrams, max_idx): (tf.Tensor, int). A two-dimensional tensor of n-grams
            , along with the number of n-grams.
    """
    dtype = segment.dtype

    def _found(ngrams, ngram, max_idx):
        """Inner function to check whether `ngrams` has the ngram `ngram`."""
        found_flag = 0
        for idx in tf.range(max_idx):  # iterate over the existing ngrams
            if (
                tf.reduce_sum(
                    tf.cast(tf.equal(ngrams.read(idx), ngram), tf.int32)
                )
                == tf.shape(ngram)[0]
            ):
                # tensorflow graph ops don't support break statements.
                found_flag = 1
        if found_flag == 1:
            return True
        return False

    max_idx = 0
    ngrams = tf.TensorArray(
        dtype=dtype, size=0, dynamic_size=True, clear_after_read=False
    )
    for i in range(tf.shape(segment)[0] - order + 1):
        formed_ngram = segment[i : i + order]
        if max_idx == 0:
            ngrams = ngrams.write(0, formed_ngram)
            max_idx += 1
        else:
            found_flag = _found(ngrams, formed_ngram, max_idx)
            if not (found_flag):
                ngrams = ngrams.write(max_idx, formed_ngram)
                max_idx += 1
    return ngrams.stack(), max_idx


def get_intersection_of_ngrams(ngrams1, len1, ngrams2, len2):
    """Given two tensors having ngrams, find common ngrams.

    Args:
        ngrams1: tf.Tensor. A two-dimensional tensor of n-grams.
        len1: int. The number of n-grams in ngrams1.
        ngrams2: tf.Tensor. A two-dimensional tensor of n-grams.
        len2: int. The number of n-grams in ngrams2.

    Returns:
        (ngrams, max_idx): (tf.Tensor, int). A two-dimensional tensor of n-grams
            common to `ngram1` and `ngram2`. Also, returns the number of n-grams
            common to `ngram1` and `ngram2`.
    """
    dtype = ngrams1.dtype
    max_idx = 0
    common_ngrams = tf.TensorArray(
        dtype=dtype, size=0, dynamic_size=True, clear_after_read=False
    )
    for i in tf.range(len1):
        for j in tf.range(len2):
            ngram1 = ngrams1[i]
            ngram2 = ngrams2[j]
            if (
                tf.reduce_sum(tf.cast(tf.equal(ngram1, ngram2), tf.int32))
                == tf.shape(ngram1)[0]
            ):
                common_ngrams = common_ngrams.write(max_idx, ngram1)
                max_idx += 1
    return common_ngrams, max_idx
