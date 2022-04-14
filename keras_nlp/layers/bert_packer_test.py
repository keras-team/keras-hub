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
"""Tests for Transformer Decoder."""

import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers.bert_packer import BertPacker


class BertPackerTest(tf.test.TestCase):
    def test_trim_single_input_ints(self):
        input_data = tf.range(3, 10)
        packer = BertPacker(8, start_value=1, end_value=2)
        output = packer(input_data)
        self.assertAllEqual(output["tokens"], [1, 3, 4, 5, 6, 7, 8, 2])
        self.assertAllEqual(output["padding_mask"], [1, 1, 1, 1, 1, 1, 1, 1])
        self.assertAllEqual(output["segment_ids"], [0, 0, 0, 0, 0, 0, 0, 0])

    def test_trim_single_input_strings(self):
        input_data = tf.constant(["a", "b", "c", "d"])
        packer = BertPacker(5, start_value="[CLS]", end_value="[SEP]")
        output = packer(input_data)
        self.assertAllEqual(output["tokens"], ["[CLS]", "a", "b", "c", "[SEP]"])
        self.assertAllEqual(output["padding_mask"], [1, 1, 1, 1, 1])
        self.assertAllEqual(output["segment_ids"], [0, 0, 0, 0, 0])

    def test_trim_multiple_inputs_round_robin(self):
        seq1 = tf.constant(["a", "b", "c"])
        seq2 = tf.constant(["x", "y", "z"])
        packer = BertPacker(
            7, start_value="[CLS]", end_value="[SEP]", truncator="round_robin"
        )
        output = packer([seq1, seq2])
        self.assertAllEqual(
            output["tokens"], ["[CLS]", "a", "b", "[SEP]", "x", "y", "[SEP]"]
        )
        self.assertAllEqual(output["padding_mask"], [1, 1, 1, 1, 1, 1, 1])
        self.assertAllEqual(output["segment_ids"], [0, 0, 0, 0, 1, 1, 1])

    def test_trim_multiple_inputs_waterfall(self):
        seq1 = tf.constant(["a", "b", "c"])
        seq2 = tf.constant(["x", "y", "z"])
        packer = BertPacker(
            7, start_value="[CLS]", end_value="[SEP]", truncator="waterfall"
        )
        output = packer([seq1, seq2])
        self.assertAllEqual(
            output["tokens"], ["[CLS]", "a", "b", "c", "[SEP]", "x", "[SEP]"]
        )
        self.assertAllEqual(output["padding_mask"], [1, 1, 1, 1, 1, 1, 1])
        self.assertAllEqual(output["segment_ids"], [0, 0, 0, 0, 0, 1, 1])

    def test_trim_batched_inputs_round_robin(self):
        seq1 = tf.ragged.constant([["a", "b", "c"], ["a", "b", "c"]])
        seq2 = tf.ragged.constant([["x", "y", "z"], ["x", "y", "z"]])
        packer = BertPacker(
            7, start_value="[CLS]", end_value="[SEP]", truncator="round_robin"
        )
        output = packer([seq1, seq2])
        print(output["tokens"])
        self.assertAllEqual(
            output["tokens"],
            [
                ["[CLS]", "a", "b", "[SEP]", "x", "y", "[SEP]"],
                ["[CLS]", "a", "b", "[SEP]", "x", "y", "[SEP]"],
            ],
        )
        self.assertAllEqual(
            output["padding_mask"],
            [
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
            ],
        )
        self.assertAllEqual(
            output["segment_ids"],
            [
                [0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1],
            ],
        )

    def test_trim_batched_inputs_waterfall(self):
        seq1 = tf.ragged.constant([["a", "b", "c"], ["a", "b"]])
        seq2 = tf.ragged.constant([["x", "y", "z"], ["x", "y", "z"]])
        packer = BertPacker(
            7, start_value="[CLS]", end_value="[SEP]", truncator="waterfall"
        )
        output = packer([seq1, seq2])
        print(output["tokens"])
        self.assertAllEqual(
            output["tokens"],
            [
                ["[CLS]", "a", "b", "c", "[SEP]", "x", "[SEP]"],
                ["[CLS]", "a", "b", "[SEP]", "x", "y", "[SEP]"],
            ],
        )
        self.assertAllEqual(
            output["padding_mask"],
            [
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
            ],
        )
        self.assertAllEqual(
            output["segment_ids"],
            [
                [0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, 1, 1],
            ],
        )

    def test_pad_inputs(self):
        seq1 = tf.constant(["a"])
        seq2 = tf.constant(["x"])
        packer = BertPacker(
            6, start_value="[CLS]", end_value="[SEP]", pad_value="[PAD]"
        )
        output = packer([seq1, seq2])
        self.assertAllEqual(
            output["tokens"], ["[CLS]", "a", "[SEP]", "x", "[SEP]", "[PAD]"]
        )
        self.assertAllEqual(output["padding_mask"], [1, 1, 1, 1, 1, 0])
        self.assertAllEqual(output["segment_ids"], [0, 0, 0, 1, 1, 0])

    def test_pad_batched_inputs(self):
        seq1 = tf.ragged.constant([["a"], ["a"]])
        seq2 = tf.ragged.constant([["x"], ["x", "y"]])
        packer = BertPacker(
            7, start_value="[CLS]", end_value="[SEP]", pad_value="[PAD]"
        )
        output = packer([seq1, seq2])
        self.assertAllEqual(
            output["tokens"],
            [
                ["[CLS]", "a", "[SEP]", "x", "[SEP]", "[PAD]", "[PAD]"],
                ["[CLS]", "a", "[SEP]", "x", "y", "[SEP]", "[PAD]"],
            ],
        )
        self.assertAllEqual(
            output["padding_mask"],
            [
                [1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 0],
            ],
        )
        self.assertAllEqual(
            output["segment_ids"],
            [
                [0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 1, 0],
            ],
        )

    def test_config(self):
        seq1 = tf.ragged.constant([["a", "b", "c"], ["a", "b"]])
        seq2 = tf.ragged.constant([["x", "y", "z"], ["x", "y", "z"]])
        original_packer = BertPacker(
            7, start_value="[CLS]", end_value="[SEP]", truncator="waterfall"
        )
        cloned_packer = BertPacker.from_config(original_packer.get_config())
        self.assertAllEqual(
            original_packer([seq1, seq2])["tokens"],
            cloned_packer([seq1, seq2])["tokens"],
        )

    def test_saving(self):
        seq1 = tf.ragged.constant([["a", "b", "c"], ["a", "b"]])
        seq2 = tf.ragged.constant([["x", "y", "z"], ["x", "y", "z"]])
        packer = BertPacker(
            7, start_value="[CLS]", end_value="[SEP]", truncator="waterfall"
        )
        inputs = (
            keras.Input(dtype="string", ragged=True, shape=(None,)),
            keras.Input(dtype="string", ragged=True, shape=(None,)),
        )
        outputs = packer(inputs)
        model = keras.Model(inputs, outputs)
        model.save(self.get_temp_dir())
        restored_model = keras.models.load_model(self.get_temp_dir())
        self.assertAllEqual(
            model((seq1, seq2))["tokens"],
            restored_model((seq1, seq2))["tokens"],
        )
