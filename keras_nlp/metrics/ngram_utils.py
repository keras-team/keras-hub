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
import tensorflow_text as tf_text


def get_ngram_count(tokenized_string, order):
    ngrams = tf_text.ngrams(
        data=tokenized_string,
        width=order,
        axis=-1,
        reduction_type=tf_text.Reduction.STRING_JOIN,
    )
    unique_ngrams, _, ngram_freq = tf.unique_with_counts(ngrams)
    return unique_ngrams, ngram_freq
