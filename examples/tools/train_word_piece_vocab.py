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
"""Create BERT wordpiece vocabularies.

This script will create wordpiece vocabularies suitable for pretraining BERT.

Usage:
python examples/tools/train_word_piece_vocabulary.py \
    --input_files ~/datasets/bert-sentence-split-data/ \
    --output_file vocab.txt
"""

import os
import sys

import tensorflow as tf
from absl import app
from absl import flags
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset

from examples.utils.scripting_utils import list_filenames_for_arg

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_files",
    None,
    "Comma seperated list of directories, files, or globs.",
)

flags.DEFINE_string(
    "output_file", None, "Output file for the computed vocabulary."
)

flags.DEFINE_bool(
    "do_lower_case",
    True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.",
)

flags.DEFINE_string(
    "reserved_tokens",
    "[PAD],[UNK],[CLS],[SEP],[MASK]",
    "Comma separated list of reserved tokens in the vocabulary.",
)

flags.DEFINE_integer("vocabulary_size", 30522, "Number of output files.")


def write_vocab_file(filepath, vocab):
    with open(filepath, "w") as file:
        for token in vocab:
            file.write(token + "\n")


def main(_):
    print(f"Reading input data from {FLAGS.input_files}")
    input_filenames = list_filenames_for_arg(FLAGS.input_files)
    if not input_filenames:
        print("No input files found. Check `input_files` flag.")
        sys.exit(1)

    print(f"Outputting to {FLAGS.output_file}")
    if os.path.exists(FLAGS.output_file):
        print(f"File {FLAGS.output_file} already exists.")
        sys.exit(1)

    with open(FLAGS.output_file, "w") as file:
        # TODO(mattdangerw): This is the slow and simple BERT vocabulary
        # learner from tf text, we should try the faster flume option.
        vocab = bert_vocab_from_dataset.bert_vocab_from_dataset(
            tf.data.TextLineDataset(input_filenames).batch(1000).prefetch(2),
            # The target vocabulary size
            vocab_size=FLAGS.vocabulary_size,
            # Reserved tokens that must be included in the vocabulary
            reserved_tokens=FLAGS.reserved_tokens.split(","),
            # Arguments for `text.BertTokenizer`
            bert_tokenizer_params={"lower_case": FLAGS.do_lower_case},
        )
        for token in vocab:
            file.write(token + "\n")


if __name__ == "__main__":
    flags.mark_flag_as_required("input_files")
    flags.mark_flag_as_required("output_file")
    app.run(main)
