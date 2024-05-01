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

import io

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "To use `keras_nlp`, please install Tensorflow: `pip install tensorflow`. "
        "The TensorFlow package is required for data preprocessing with any backend."
    )

try:
    import sentencepiece as spm
except ImportError:
    spm = None

from keras_nlp.src.api_export import keras_nlp_export


@keras_nlp_export("keras_nlp.tokenizers.compute_sentence_piece_proto")
def compute_sentence_piece_proto(
    data,
    vocabulary_size,
    model_type="unigram",
    proto_output_file=None,
    lowercase=False,
):
    r"""A utility to train a SentencePiece vocabulary.

    Trains a SentencePiece vocabulary from an input dataset or a list of
    filenames.

    If `data` is a list of filenames, the file format is required to be plain
    text files, and the text will be read in line by line during training.

    Args:
        data: A `tf.data.Dataset`, or a list of filenames.
        vocabulary_size: int. The maximum size of a vocabulary to be trained.
        model_type: str. The model algorithm must be one of
            `"unigram"`, `"bpe"`, `"word"` or `"char"`. Defaults to `"unigram"`.
        proto_output_file: str. If provided it will be used
            as model_file which is passed to model_writer.
            If `None`, the model_file will be `io.BytesIO` object.
            Defaults to `None`.
        lowercase: bool. If True, the input text will be
            lowercased before tokenization. Defaults to `False`.

    Returns:
        A `bytes` object with a serialized SentencePiece proto or
        `None` if proto_output_file if provided.

    Examples:

    Basic Usage (from Dataset).
    >>> inputs = tf.data.Dataset.from_tensor_slices(["Drifting Along"])
    >>> proto = keras_nlp.tokenizers.compute_sentence_piece_proto(inputs, vocabulary_size=15)
    >>> tokenizer = keras_nlp.tokenizers.SentencePieceTokenizer(proto=proto)
    >>> outputs = inputs.map(tokenizer)
    >>> for output in outputs:
    ...     print(output)
    tf.Tensor([ 4  8 12  5  9 14  5  6 13  4  7 10 11  6 13],
    shape=(15,), dtype=int32)

    Basic Usage (with files).
    ``` python
    with open("test.txt", "w+") as f: f.write("Drifting Along\n")
    inputs = ["test.txt"]
    proto = keras_nlp.tokenizers.compute_sentence_piece_proto(
         inputs, vocabulary_size=15, proto_output_file="model.spm")
    tokenizer = keras_nlp.tokenizers.SentencePieceTokenizer(proto="model.spm")
    ds = tf.data.Dataset.from_tensor_slices(["the quick brown fox."])
    ds = ds.map(tokenizer)
    ```

    Usage with lowercase
    >>> inputs = tf.data.Dataset.from_tensor_slices(["Drifting Along"])
    >>> proto = keras_nlp.tokenizers.compute_sentence_piece_proto(
    ...     inputs, vocabulary_size=15, lowercase=True)
    >>> tokenizer = keras_nlp.tokenizers.SentencePieceTokenizer(proto=proto)
    >>> outputs = inputs.map(tokenizer)
    >>> for output in outputs:
    ...     print(output)
    tf.Tensor([ 4  8 12  5  9 14  5  6 13  4  7 10 11  6 13],
    shape=(15,), dtype=int32)
    """

    if spm is None:
        raise ImportError(
            f"{compute_sentence_piece_proto.__name__} requires the "
            "`sentencepiece` package. Please install it with "
            "`pip install sentencepiece`."
        )

    if not isinstance(data, (list, tuple, tf.data.Dataset)):
        raise ValueError(
            "The `data` argument must be either `tf.data.Dataset` or `tuple` or `list`. "
            f"Received: type(data)={type(data)}."
        )

    if model_type not in ["unigram", "bpe", "word", "char"]:
        raise ValueError(
            "The `model_type` argument must be one of `unigram`, `bpe`, `word`"
            f"or `char`. Received: model_type={model_type}."
        )

    model_writer = (
        open(proto_output_file, "wb") if proto_output_file else io.BytesIO()
    )
    is_dataset = isinstance(data, tf.data.Dataset)
    if is_dataset:
        spm.SentencePieceTrainer.train(
            sentence_iterator=data.as_numpy_iterator(),
            model_writer=model_writer,
            vocab_size=vocabulary_size,
            model_type=model_type,
            normalization_rule_name="nmt_nfkc_cf" if lowercase else "nmt_nfkc",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )
    else:
        spm.SentencePieceTrainer.train(
            input=data,
            model_writer=model_writer,
            vocab_size=vocabulary_size,
            model_type=model_type,
            normalization_rule_name="nmt_nfkc_cf" if lowercase else "nmt_nfkc",
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )
    if proto_output_file:
        model_writer.close()
    else:
        return model_writer.getvalue()
