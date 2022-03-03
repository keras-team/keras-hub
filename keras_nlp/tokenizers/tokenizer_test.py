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

from keras_nlp.tokenizers.tokenizer import Tokenizer


class SimpleTokenizer(Tokenizer):
    __test__ = False  # for pytest

    def tokenize(self, inputs):
        return tf.strings.split(inputs).to_tensor()

    def detokenize(self, inputs):
        return tf.strings.reduce_join([inputs], separator=" ", axis=-1)


class TokenizerTest(tf.test.TestCase):
    def test_tokenize(self):
        input_data = ["the quick brown fox"]
        tokenizer = SimpleTokenizer()
        tokenize_output = tokenizer.tokenize(input_data)
        call_output = tokenizer(input_data)
        self.assertAllEqual(tokenize_output, [["the", "quick", "brown", "fox"]])
        self.assertAllEqual(call_output, [["the", "quick", "brown", "fox"]])

    def test_detokenize(self):
        input_data = ["the", "quick", "brown", "fox"]
        tokenizer = SimpleTokenizer()
        detokenize_output = tokenizer.detokenize(input_data)
        self.assertAllEqual(detokenize_output, ["the quick brown fox"])

    def test_functional_model(self):
        input_data = tf.constant(["the   quick   brown   fox"])
        tokenizer = SimpleTokenizer()
        inputs = keras.Input(dtype="string", shape=())
        outputs = tokenizer.detokenize(tokenizer.tokenize(inputs))
        model = keras.Model(inputs, outputs)
        model_output = model(input_data)
        # There appears to be a bug with shape inference for ragged reduce_join.
        # The second dimension should be removed.
        self.assertAllEqual(model_output, [["the quick brown fox"]])

    def test_missing_tokenize_raises(self):
        with self.assertRaises(NotImplementedError):
            Tokenizer()(["the quick brown fox"])
