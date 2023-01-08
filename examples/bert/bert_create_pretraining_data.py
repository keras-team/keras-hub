# Copyright 2022 The KerasNLP Authors.
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
"""Create masked LM/next sentence masked_lm TF examples for BERT.

This script will create TFRecord files containing BERT training examples with
both word masking and next sentence prediction.

This script will load the entire dataset into memory to setup the next sentence
prediction task, so it is recommended to run this on shards of data at a time to
avoid memory issues.

By default, it will duplicate the input data 10 times with different masks and
sentence pairs, as will the original paper. So a 20gb source of wikipedia and
bookscorpus will result in a 400gb dataset.

This script is adapted from the original BERT respository:
https://github.com/google-research/bert/blob/master/create_pretraining_data.py

Usage:
python create_pretraining_data.py \
    --input_files ~/datasets/bert-sentence-split-data/shard_0.txt \
    --output_directory ~/datasets/bert-pretraining-data/shard_0.txt \
    --vocab_file vocab.txt
"""

import collections
import os
import random
import sys

import tensorflow as tf
import tensorflow_text as tf_text
from absl import app
from absl import flags

from examples.bert.bert_config import PREPROCESSING_CONFIG
from examples.utils.scripting_utils import list_filenames_for_arg

# Tokenization will happen with tensorflow and can easily OOM a GPU.
# Restrict the script to run CPU as GPU will not offer speedup here anyway.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_files",
    None,
    "Comma seperated list of directories, globs or files.",
)

flags.DEFINE_string(
    "output_file",
    None,
    "Output TF record file.",
)

flags.DEFINE_string(
    "vocab_file",
    None,
    "The vocabulary file for tokenization.",
)

flags.DEFINE_bool(
    "do_lower_case",
    True,
    "Whether to lower case the input text.",
)

flags.DEFINE_integer(
    "random_seed",
    12345,
    "Random seed for data generation.",
)


def convert_to_unicode(text):
    """Converts text to Unicode if it's not already, assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def printable_text(text):
    """Returns text encoded in a way suitable for print."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


# This tuple holds a complete training instance of data ready for serialization.
TrainingInstance = collections.namedtuple(
    "TrainingInstance",
    [
        "tokens",
        "segment_ids",
        "is_random_next",
        "masked_lm_positions",
        "masked_lm_labels",
    ],
)


def write_instance_to_example_files(
    instances, vocab, max_seq_length, max_predictions_per_seq, output_filename
):
    """Create TF example files from `TrainingInstance`s."""
    writer = tf.io.TFRecordWriter(output_filename)
    total_written = 0
    lookup = dict(zip(vocab, range(len(vocab))))
    for (inst_index, instance) in enumerate(instances):
        token_ids = [lookup[x] for x in instance.tokens]
        padding_mask = [1] * len(token_ids)
        segment_ids = list(instance.segment_ids)
        assert len(token_ids) <= max_seq_length

        while len(token_ids) < max_seq_length:
            token_ids.append(0)
            padding_mask.append(0)
            segment_ids.append(0)

        assert len(token_ids) == max_seq_length
        assert len(padding_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = [lookup[x] for x in instance.masked_lm_labels]
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        next_sentence_label = 1 if instance.is_random_next else 0

        features = collections.OrderedDict()
        features["token_ids"] = int_feature(token_ids)
        features["padding_mask"] = int_feature(padding_mask)
        features["segment_ids"] = int_feature(segment_ids)
        features["masked_lm_positions"] = int_feature(masked_lm_positions)
        features["masked_lm_ids"] = int_feature(masked_lm_ids)
        features["masked_lm_weights"] = float_feature(masked_lm_weights)
        features["next_sentence_labels"] = int_feature([next_sentence_label])

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features)
        )

        writer.write(tf_example.SerializeToString())
        total_written += 1

    writer.close()
    print(f"Wrote {total_written} total instances")


def int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))


def float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))


def create_training_instances(
    input_filenames,
    tokenizer,
    vocab,
    max_seq_length,
    dupe_factor,
    short_seq_prob,
    masked_lm_prob,
    max_predictions_per_seq,
    rng,
):
    """Create `TrainingInstance`s from raw text."""
    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    dataset = tf.data.TextLineDataset(input_filenames)
    dataset = dataset.map(
        lambda x: tokenizer.tokenize(x).flat_values,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    all_documents = []
    current_document = []
    for line in dataset.as_numpy_iterator():
        if line.size == 0 and current_document:
            all_documents.append(current_document)
            current_document = []
        else:
            line = [x.decode("utf-8") for x in line]
            if line:
                current_document.append(line)
    rng.shuffle(all_documents)

    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document(
                    all_documents,
                    document_index,
                    max_seq_length,
                    short_seq_prob,
                    masked_lm_prob,
                    max_predictions_per_seq,
                    vocab,
                    rng,
                )
            )
    rng.shuffle(instances)
    return instances


def create_instances_from_document(
    all_documents,
    document_index,
    max_seq_length,
    short_seq_prob,
    masked_lm_prob,
    max_predictions_per_seq,
    vocab_words,
    rng,
):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the
                # `A` (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for
                    # large corpora. However, just to be careful, we try to make
                    # sure that the random document is not the same as the
                    # document we're processing.
                    for _ in range(10):
                        random_document_index = rng.randint(
                            0, len(all_documents) - 1
                        )
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them
                    # back" so they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                (
                    tokens,
                    masked_lm_positions,
                    masked_lm_labels,
                ) = create_masked_lm_predictions(
                    tokens,
                    masked_lm_prob,
                    max_predictions_per_seq,
                    vocab_words,
                    rng,
                )
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels,
                )
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


MaskedLmInstance = collections.namedtuple(
    "MaskedLmInstance", ["index", "label"]
)


def create_masked_lm_predictions(
    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng
):
    """Creates the predictions for the masked LM objective."""

    # TODO(jbischof): replace with keras_nlp.layers.MaskedLMMaskGenerator
    # (Issue #166)

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(
        max_predictions_per_seq,
        max(1, int(round(len(tokens) * masked_lm_prob))),
    )

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[
                        rng.randint(0, len(vocab_words) - 1)
                    ]

            output_tokens[index] = masked_token

            masked_lms.append(
                MaskedLmInstance(index=index, label=tokens[index])
            )
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def main(_):
    print(f"Reading input data from {FLAGS.input_files}")
    input_filenames = list_filenames_for_arg(FLAGS.input_files)
    if not input_filenames:
        print("No input files found. Check `input_files` flag.")
        sys.exit(1)

    # Load the vocabulary.
    vocab = []
    with open(FLAGS.vocab_file, "r") as vocab_file:
        for line in vocab_file:
            vocab.append(line.strip())
    tokenizer = tf_text.BertTokenizer(
        FLAGS.vocab_file,
        lower_case=FLAGS.do_lower_case,
        token_out_type=tf.string,
    )

    rng = random.Random(FLAGS.random_seed)
    instances = create_training_instances(
        input_filenames,
        tokenizer,
        vocab,
        PREPROCESSING_CONFIG["max_seq_length"],
        PREPROCESSING_CONFIG["dupe_factor"],
        PREPROCESSING_CONFIG["short_seq_prob"],
        PREPROCESSING_CONFIG["masked_lm_prob"],
        PREPROCESSING_CONFIG["max_predictions_per_seq"],
        rng,
    )

    print(f"Outputting to {FLAGS.output_file}.")
    output_directory = os.path.dirname(FLAGS.output_file)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    write_instance_to_example_files(
        instances,
        vocab,
        PREPROCESSING_CONFIG["max_seq_length"],
        PREPROCESSING_CONFIG["max_predictions_per_seq"],
        FLAGS.output_file,
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("input_files")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    app.run(main)
