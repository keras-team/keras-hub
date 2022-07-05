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
"""Split sentences from raw input documents using nltk.

A script to sentence split a raw dataset (e.g. wikipedia or bookscorpus) into
sentences for further preproessing for BERT. The output file format is the
format expected by `create_pretraining_data.py`, where each file contains one
line per sentence, with empty newlines between documents.

This script will run muliprocessed, and the number of concurrent process and
output file shards can be controlled with `--num_jobs` and `--num_shards`.

Usage:
python examples/tools/create_sentence_split_data.py \
    --input_files ~/datasets/wikipedia,~/datasets/bookscorpus \
    --output_directory ~/datasets/bert-sentence-split-data
"""

import contextlib
import multiprocessing
import os
import random
import sys

import nltk
from absl import app
from absl import flags
from tensorflow import keras

from examples.utils.scripting_utils import list_filenames_for_arg

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_files",
    None,
    "Comma seperated list of directories, files, or globs for input data.",
)

flags.DEFINE_string(
    "output_directory",
    None,
    "Directory for output data.",
)

flags.DEFINE_integer("num_jobs", None, "Number of file shards to use.")

flags.DEFINE_integer("num_shards", 500, "Number of file shards to use.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")


def parse_wiki_file(file):
    """Read documents from a wikipedia dump file."""
    documents = []
    in_article = False
    article_lines = []
    for line in file:
        line = line.strip()
        # Skip empty lines.
        if line == "":
            continue
        elif "<doc id=" in line:
            in_article = True
        elif "</doc>" in line:
            in_article = False
            # There are many wikipedia articles that are only titles (one
            # line) or or redirects (two lines), we will skip these.
            if len(article_lines) > 2:
                # Skip the title.
                documents.append(" ".join(article_lines[1:]))
            article_lines = []
        elif in_article:
            article_lines.append(line)
    return documents


def parse_text_file(file):
    """Read documents from a plain text file."""
    documents = []
    file_lines = []
    for line in file:
        line = line.strip()
        # Skip empty lines.
        if line == "":
            continue
        file_lines.append(line)
    documents.append(" ".join(file_lines))
    return documents


def read_file(filename):
    """Read documents from an input file."""
    with open(filename, mode="r") as file:
        firstline = file.readline()
        file.seek(0)
        # Very basic autodetection of file type.
        # Wikipedia dump files all start with a doc id tag.
        if "<doc id=" in firstline:
            return parse_wiki_file(file)
        return parse_text_file(file)


def process_file(filename):
    """Read documents from an input file and split into sentences with nltk."""
    split_documents = []
    for document in read_file(filename):
        sentences = nltk.tokenize.sent_tokenize(document)
        split_documents.append(sentences)
    return split_documents


def main(_):
    nltk.download("punkt")
    print(f"Reading input data from {FLAGS.input_files}")
    input_filenames = list_filenames_for_arg(FLAGS.input_files)
    if not input_filenames:
        print("No input files found. Check `input_files` flag.")
        sys.exit(1)

    # Randomize files so we aren't processing input directories sequentially.
    rng = random.Random(FLAGS.random_seed)
    rng.shuffle(input_filenames)

    # We will read and sentence split with multiprocessing, but write from
    # a single thread to balance our shard sizes well.
    pool = multiprocessing.Pool(FLAGS.num_jobs)

    print(f"Outputting to {FLAGS.output_directory}.")
    if not os.path.exists(FLAGS.output_directory):
        os.mkdir(FLAGS.output_directory)

    progbar = keras.utils.Progbar(len(input_filenames), unit_name="files")
    progbar.update(0)
    with contextlib.ExitStack() as stack:
        # Open all files.
        output_files = []
        for i in range(FLAGS.num_shards):
            path = os.path.join(FLAGS.output_directory, f"shard_{i}.txt")
            output_files.append(stack.enter_context(open(path, "w")))

        # Write documents to disk.
        total_files = 0
        total_documents = 0
        for documents in pool.imap_unordered(process_file, input_filenames):
            for document in documents:
                output_file = output_files[total_documents % FLAGS.num_shards]
                for sentence in document:
                    output_file.write(sentence + "\n")
                # Blank newline marks a new document.
                output_file.write("\n")
                total_documents += 1
            total_files += 1
            progbar.update(total_files)

    print("Done.")
    print(f"Read {total_files} files.")
    print(f"Processed {total_documents} documents.")


if __name__ == "__main__":
    flags.mark_flag_as_required("input_files")
    flags.mark_flag_as_required("output_directory")
    app.run(main)
