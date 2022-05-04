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
"""Utility files for BERT scripts."""

import glob
import os
from absl import flags
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer(
    "max_predictions_per_seq",
    20,
    "Maximum number of masked LM predictions per sequence.",
)


def list_filenames_for_arg(arg_pattern):
    """List filenames from a comma separated list of files, dirs, and globs."""
    input_filenames = []
    for pattern in arg_pattern.split(","):
        pattern = os.path.expanduser(pattern)
        if os.path.isdir(pattern):
            pattern = os.path.join(pattern, "**", "*")
        for filename in glob.iglob(pattern, recursive=True):
            if not os.path.isdir(filename):
                input_filenames.append(filename)
    return input_filenames

def decode_record(record):
    """Decodes a record to a TensorFlow example."""
    seq_length = FLAGS.max_seq_length
    lm_length = FLAGS.max_predictions_per_seq
    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "masked_lm_positions": tf.io.FixedLenFeature([lm_length], tf.int64),
        "masked_lm_ids": tf.io.FixedLenFeature([lm_length], tf.int64),
        "masked_lm_weights": tf.io.FixedLenFeature([lm_length], tf.float32),
        "next_sentence_labels": tf.io.FixedLenFeature([1], tf.int64),
    }
    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    example = tf.io.parse_single_example(record, name_to_features)
    for name in list(example.keys()):
        value = example[name]
        if value.dtype == tf.int64:
            value = tf.cast(value, tf.int32)
        example[name] = value
    return example

def visualize_tfrecord(filepath):
    dataset = tf.data.TFRecordDataset(filepath)
    sample = next(iter(dataset))
    print(decode_record(sample))


