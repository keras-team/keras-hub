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

import numpy as np

from keras_nlp.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_nlp.tests.test_case import TestCase


class MultiSegmentPackerTest(TestCase):
    def test_trim_single_input_ints(self):
        input_data = np.arange(3, 10)
        packer = MultiSegmentPacker(
            sequence_length=8, start_value=1, end_value=2
        )
        token_ids, segment_ids = packer(input_data)
        self.assertAllEqual(token_ids, [1, 3, 4, 5, 6, 7, 8, 2])
        self.assertAllEqual(segment_ids, [0, 0, 0, 0, 0, 0, 0, 0])

    def test_trim_single_input_strings(self):
        input_data = np.array(["a", "b", "c", "d"])
        packer = MultiSegmentPacker(
            sequence_length=5, start_value="[CLS]", end_value="[SEP]"
        )
        token_ids, segment_ids = packer(input_data)
        self.assertAllEqual(token_ids, ["[CLS]", "a", "b", "c", "[SEP]"])
        self.assertAllEqual(segment_ids, [0, 0, 0, 0, 0])

    def test_trim_multiple_inputs_round_robin(self):
        seq1 = ["a", "b", "c"]
        seq2 = ["x", "y", "z"]
        packer = MultiSegmentPacker(
            sequence_length=7,
            start_value="[CLS]",
            end_value="[SEP]",
            truncate="round_robin",
        )
        token_ids, segment_ids = packer([seq1, seq2])
        self.assertAllEqual(
            token_ids, ["[CLS]", "a", "b", "[SEP]", "x", "y", "[SEP]"]
        )
        self.assertAllEqual(segment_ids, [0, 0, 0, 0, 1, 1, 1])

    def test_trim_multiple_inputs_waterfall(self):
        seq1 = ["a", "b", "c"]
        seq2 = ["x", "y", "z"]
        packer = MultiSegmentPacker(
            sequence_length=7,
            start_value="[CLS]",
            end_value="[SEP]",
            truncate="waterfall",
        )
        token_ids, segment_ids = packer([seq1, seq2])
        self.assertAllEqual(
            token_ids, ["[CLS]", "a", "b", "c", "[SEP]", "x", "[SEP]"]
        )
        self.assertAllEqual(segment_ids, [0, 0, 0, 0, 0, 1, 1])

    def test_trim_batched_inputs_round_robin(self):
        seq1 = [["a", "b", "c"], ["a", "b", "c"]]
        seq2 = [["x", "y", "z"], ["x", "y", "z"]]
        packer = MultiSegmentPacker(
            sequence_length=7,
            start_value="[CLS]",
            end_value="[SEP]",
            truncate="round_robin",
        )
        token_ids, segment_ids = packer([seq1, seq2])
        self.assertAllEqual(
            token_ids,
            [
                ["[CLS]", "a", "b", "[SEP]", "x", "y", "[SEP]"],
                ["[CLS]", "a", "b", "[SEP]", "x", "y", "[SEP]"],
            ],
        )
        self.assertAllEqual(
            segment_ids,
            [
                [0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 1, 1, 1],
            ],
        )

    def test_trim_batched_inputs_waterfall(self):
        seq1 = [["a", "b", "c"], ["a", "b"]]
        seq2 = [["x", "y", "z"], ["x", "y", "z"]]
        packer = MultiSegmentPacker(
            sequence_length=7,
            start_value="[CLS]",
            end_value="[SEP]",
            truncate="waterfall",
        )
        token_ids, segment_ids = packer([seq1, seq2])
        self.assertAllEqual(
            token_ids,
            [
                ["[CLS]", "a", "b", "c", "[SEP]", "x", "[SEP]"],
                ["[CLS]", "a", "b", "[SEP]", "x", "y", "[SEP]"],
            ],
        )
        self.assertAllEqual(
            segment_ids,
            [
                [0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 1, 1, 1],
            ],
        )

    def test_pad_inputs(self):
        seq1 = ["a"]
        seq2 = ["x"]
        packer = MultiSegmentPacker(
            6, start_value="[CLS]", end_value="[SEP]", pad_value="[PAD]"
        )
        token_ids, segment_ids = packer([seq1, seq2])
        self.assertAllEqual(
            token_ids,
            ["[CLS]", "a", "[SEP]", "x", "[SEP]", "[PAD]"],
        )
        self.assertAllEqual(segment_ids, [0, 0, 0, 1, 1, 0])

    def test_pad_batched_inputs(self):
        seq1 = [["a"], ["a"]]
        seq2 = [["x"], ["x", "y"]]
        packer = MultiSegmentPacker(
            sequence_length=7,
            start_value="[CLS]",
            end_value="[SEP]",
            pad_value="[PAD]",
        )
        token_ids, segment_ids = packer([seq1, seq2])
        self.assertAllEqual(
            token_ids,
            [
                ["[CLS]", "a", "[SEP]", "x", "[SEP]", "[PAD]", "[PAD]"],
                ["[CLS]", "a", "[SEP]", "x", "y", "[SEP]", "[PAD]"],
            ],
        )
        self.assertAllEqual(
            segment_ids,
            [
                [0, 0, 0, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 1, 0],
            ],
        )

    def test_list_special_tokens(self):
        seq1 = [["a", "b"], ["a", "b"]]
        seq2 = [["x", "y"], ["x"]]
        packer = MultiSegmentPacker(
            8,
            start_value="<s>",
            end_value="</s>",
            sep_value=["</s>", "</s>"],
            pad_value="<pad>",
            truncate="round_robin",
        )
        token_ids, segment_ids = packer([seq1, seq2])
        self.assertAllEqual(
            token_ids,
            [
                ["<s>", "a", "b", "</s>", "</s>", "x", "y", "</s>"],
                ["<s>", "a", "b", "</s>", "</s>", "x", "</s>", "<pad>"],
            ],
        )
        self.assertAllEqual(
            segment_ids,
            [
                [0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 0],
            ],
        )

    def test_config(self):
        seq1 = [["a", "b", "c"], ["a", "b"]]
        seq2 = [["x", "y", "z"], ["x", "y", "z"]]
        original_packer = MultiSegmentPacker(
            sequence_length=7,
            start_value="[CLS]",
            end_value="[SEP]",
            truncate="waterfall",
        )
        cloned_packer = MultiSegmentPacker.from_config(
            original_packer.get_config()
        )
        token_ids, segment_ids = original_packer([seq1, seq2])
        cloned_token_ids, cloned_segment_ids = cloned_packer([seq1, seq2])
        self.assertAllEqual(token_ids, cloned_token_ids)
        self.assertAllEqual(segment_ids, cloned_segment_ids)
