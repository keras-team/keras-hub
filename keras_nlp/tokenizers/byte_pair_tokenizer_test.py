# Copyright 2023 The KerasNLP Authors
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

import pytest
import tensorflow as tf

from keras_nlp.backend import keras
from keras_nlp.tests.test_case import TestCase
from keras_nlp.tokenizers.byte_pair_tokenizer import BytePairTokenizer

VOCAB_PATH = keras.utils.get_file(
    None,
    "https://storage.googleapis.com/keras-nlp/models/roberta_base/vocab.json",
)
MERGE_PATH = keras.utils.get_file(
    None,
    "https://storage.googleapis.com/keras-nlp/models/roberta_base/merges.txt",
)


@pytest.mark.large
class BytePairTokenizerTest(TestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = BytePairTokenizer(
            vocabulary=VOCAB_PATH, merges=MERGE_PATH
        )

    def test_tokenize_list_input(self):
        input_data = ["brown.", "black."]
        call_output = self.tokenizer(input_data)
        tokenize_output = self.tokenizer.tokenize(input_data)
        expected = [[31876, 4], [14178, 4]]
        self.assertAllEqual(call_output, expected)
        self.assertAllEqual(tokenize_output, expected)

        input_data = tf.convert_to_tensor(["brown.", "black."])
        encoded = self.tokenizer(input_data)
        self.assertAllEqual(encoded, expected)

    def test_tokenize_string_output(self):
        input_data = ["quick brown fox.", "slow black bear."]
        tokenizer = BytePairTokenizer(
            vocabulary=VOCAB_PATH, merges=MERGE_PATH, dtype="string"
        )
        call_output = tokenizer(input_data)
        expected = [
            ["quick", "Ġbrown", "Ġfox", "."],
            ["slow", "Ġblack", "Ġbear", "."],
        ]
        self.assertAllEqual(call_output, expected)

    def test_tokenize_with_special_tokens(self):
        vocab = {"sp": 0, "s": 1, "p": 2}
        merges = ["s p"]
        tokenizer = BytePairTokenizer(
            vocabulary=vocab,
            merges=merges,
            unsplittable_tokens=["s", "p"],
        )
        output = tokenizer("sp")
        self.assertAllEqual(output, [1, 2])

        # If not setting special tokens, "sp" is one token.
        tokenizer = BytePairTokenizer(
            vocabulary=vocab,
            merges=merges,
        )
        output = tokenizer("sp")
        self.assertAllEqual(output, [0])

    def test_tokenize_prefix_space(self):
        input_data = ["brown.", "black."]
        tokenizer = BytePairTokenizer(
            vocabulary=VOCAB_PATH,
            merges=MERGE_PATH,
            dtype="string",
            add_prefix_space=True,
        )
        call_output = tokenizer(input_data)

        expected = [["Ġbrown", "."], ["Ġblack", "."]]
        self.assertAllEqual(call_output, expected)

    def test_tokenize_scalar_input(self):
        input_data = "brown."
        encoded = self.tokenizer.tokenize(input_data)
        self.assertAllEqual(encoded, [31876, 4])

    def test_detokenize_scalar_input(self):
        input_data = ["quick brown fox."]
        encoded = self.tokenizer.tokenize(input_data)
        decoded = self.tokenizer.detokenize(encoded)
        self.assertAllEqual(input_data, decoded)

    def test_detokenize_list_input(self):
        input_data = ["quick brown fox.", "slow bear"]
        encoded = self.tokenizer.tokenize(input_data)
        decoded = self.tokenizer.detokenize(encoded)
        self.assertAllEqual(input_data, decoded)

    def test_error_id_out_of_vocabulary(self):
        with self.assertRaises(ValueError):
            self.tokenizer.id_to_token(self.tokenizer.vocabulary_size())
        with self.assertRaises(ValueError):
            self.tokenizer.id_to_token(-1)

    def test_whitespace_split(self):
        input_data = "\n\n\n  s"
        encoded = self.tokenizer(input_data)
        self.assertAllEqual(encoded, [50140, 50118, 1437, 579])

        input_data = "  \n\n\ns"
        encoded = self.tokenizer(input_data)
        self.assertAllEqual(encoded, [1437, 1437, 50140, 50118, 29])

    def test_special_whitespace(self):
        input_data = "\xa0 \xa0 \x3000 s"
        encoded = self.tokenizer(input_data)
        self.assertAllEqual(encoded, [50141, 50143, 12096, 579])

    def test_cjk_input(self):
        input_data = "素晴らしい！芭比Q啦～"
        # Black formats long list by one element per line, which is bad to read.
        expected = [36714, 20024, 21402, 37127, 27, 20024, 48945, 47918]
        expected += [47780, 43251, 4394, 10172, 36484, 27969, 12410, 37127]
        expected += [10965, 10674, 1864, 42393, 15722, 18164, 43251, 10809]
        expected += [17772]
        encoded = self.tokenizer(input_data)
        self.assertAllEqual(encoded, expected)

    def test_tokenize_with_tf_data(self):
        data = [
            "I am just a test string",
            "I am also a test string",
            "I am still a test string",
            "me too",
            "I am not a test string (joking)",
            "You guys should add punctuation!",
            "Period matters!",
        ]
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.batch(2).map(self.tokenizer)
        encoded = next(iter(ds))
        expected = [
            [100, 524, 95, 10, 1296, 6755],
            [100, 524, 67, 10, 1296, 6755],
        ]
        self.assertAllEqual(encoded, expected)

    def test_config(self):
        input_data = ["the quick brown whale."]
        cloned_tokenizer = BytePairTokenizer.from_config(
            self.tokenizer.get_config()
        )
        cloned_tokenizer.set_vocabulary_and_merges(
            self.tokenizer.vocabulary, self.tokenizer.merges
        )
        self.assertAllEqual(
            self.tokenizer(input_data),
            cloned_tokenizer(input_data),
        )
