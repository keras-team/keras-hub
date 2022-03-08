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
import pathlib
import random
import re
import string

import tensorflow as tf
from tensorflow import keras


def download_data():
    text_file = keras.utils.get_file(
        fname="spa-eng.zip",
        origin=(
            "http://storage.googleapis.com/download.tensorflow.org/data/"
            + "spa-eng.zip"
        ),
        extract=True,
    )
    return pathlib.Path(text_file).parent / "spa-eng" / "spa.txt"


def read_data(filepath):
    with open(filepath) as f:
        lines = f.read().split("\n")[:-1]
        text_pairs = []
        for line in lines:
            eng, spa = line.split("\t")
            spa = "[start] " + spa + " [end]"
            text_pairs.append((eng, spa))
    return text_pairs


def split_train_val_test(text_pairs):
    random.shuffle(text_pairs)
    num_val_samples = int(0.15 * len(text_pairs))
    num_train_samples = len(text_pairs) - 2 * num_val_samples
    train_pairs = text_pairs[:num_train_samples]
    val_end_index = num_train_samples + num_val_samples
    val_pairs = text_pairs[num_train_samples:val_end_index]
    test_pairs = text_pairs[val_end_index:]
    return train_pairs, val_pairs, test_pairs


strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")


@keras.utils.register_keras_serializable()
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase,
        "[%s]" % re.escape(strip_chars),
        "",
    )


def prepare_tokenizer(train_pairs, sequence_length, vocab_size):
    """Preapare English and Spanish tokenizer."""
    eng_tokenizer = keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length,
    )
    spa_tokenizer = keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=sequence_length + 1,
        standardize=custom_standardization,
    )
    eng_texts, spa_texts = zip(*train_pairs)
    eng_tokenizer.adapt(eng_texts)
    spa_tokenizer.adapt(spa_texts)
    return eng_tokenizer, spa_tokenizer


def prepare_datasets(text_pairs, batch_size, eng_tokenizer, spa_tokenizer):
    """Transform raw text pairs to tf datasets."""
    eng_texts, spa_texts = zip(*text_pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)

    def format_dataset(eng, spa):
        """Format the dataset given input English and Spanish text.

        The output format is:
            x: a pair of English and Spanish sentence.
            y: The Spanish sentence in x shifts 1 token towards right, because
                we are predicting the next token.
        """
        eng = eng_tokenizer(eng)
        spa = spa_tokenizer(spa)
        return (
            {
                "encoder_inputs": eng,
                "decoder_inputs": spa[:, :-1],
            },
            spa[:, 1:],
            tf.cast((spa[:, 1:] != 0), tf.float32),  # mask as sample weights
        )

    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(2048).prefetch(tf.data.AUTOTUNE).cache()


def get_dataset_and_tokenizer(sequence_length, vocab_size, batch_size):
    """Main method to get the formatted machine translation dataset."""
    filepath = download_data()
    text_pairs = read_data(filepath)
    train_pairs, val_pairs, test_pairs = split_train_val_test(text_pairs)
    eng_tokenizer, spa_tokenizer = prepare_tokenizer(
        train_pairs, sequence_length, vocab_size
    )
    train_ds = prepare_datasets(
        train_pairs,
        batch_size,
        eng_tokenizer,
        spa_tokenizer,
    )
    val_ds = prepare_datasets(
        val_pairs,
        batch_size,
        eng_tokenizer,
        spa_tokenizer,
    )
    test_ds = prepare_datasets(
        test_pairs,
        batch_size,
        eng_tokenizer,
        spa_tokenizer,
    )
    return (train_ds, val_ds, test_ds), (eng_tokenizer, spa_tokenizer)
