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

"""XLM-RoBERTa tokenizer."""

import copy
import os

import tensorflow as tf
from tensorflow import keras

from keras_nlp.models.xlm_roberta.xlm_roberta_presets import backbone_presets
from keras_nlp.tokenizers.sentence_piece_tokenizer import SentencePieceTokenizer
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring
from keras_nlp.utils.tf_utils import tensor_to_string_list


@keras.utils.register_keras_serializable(package="keras_nlp")
class XLMRobertaTokenizer(SentencePieceTokenizer):
    """XLM-RoBERTa tokenizer layer based on SentencePiece.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_nlp.tokenizers.SentencePieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    XLM-RoBERTa models and provides a `from_preset()` method to automatically
    download a matching vocabulary for a XLM-RoBERTa preset.

    The original fairseq implementation of XLM-RoBERTa modifies the indices of
    the SentencePiece tokenizer output. To preserve compatibility, we make the
    same changes, i.e., `"<s>"`, `"<pad>"`, `"</s>"` and `"<unk>"` are mapped to
    0, 1, 2, 3, respectively, and non-special token indices are shifted right
    by one.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        proto: Either a `string` path to a SentencePiece proto file, or a
            `bytes` object with a serialized SentencePiece proto. See the
            [SentencePiece repository](https://github.com/google/sentencepiece)
            for more details on the format.

    Examples:

    ```python
    def train_sentencepiece(ds, vocab_size):
        bytes_io = io.BytesIO()
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=ds.as_numpy_iterator(),
            model_writer=bytes_io,
            vocab_size=vocab_size,
            model_type="WORD",
            unk_id=0,
            bos_id=1,
            eos_id=2,
        )
        return bytes_io.getvalue()

    ds = tf.data.Dataset.from_tensor_slices(
        ["the quick brown fox", "the earth is round"]
    )

    proto = train_sentencepiece(ds, vocab_size=10)
    tokenizer = keras_nlp.models.XLMRobertaTokenizer(proto=proto)

    # Batched inputs.
    tokenizer(["the quick brown fox", "the earth is round"])

    # Unbatched inputs.
    tokenizer("the quick brown fox")

    # Detokenization.
    tokenizer.detokenize(tf.constant([[0, 4, 9, 5, 7, 2]]))
    ```
    """

    def __init__(self, proto, **kwargs):
        super().__init__(proto=proto, **kwargs)

        # List of special tokens.
        self._vocabulary_prefix = ["<s>", "<pad>", "</s>", "<unk>"]

        # IDs of special tokens.
        self.start_token_id = 0  # <s>
        self.pad_token_id = 1  # <pad>
        self.end_token_id = 2  # </s>
        self.unk_token_id = 3  # <unk>

    def vocabulary_size(self):
        """Get the size of the tokenizer vocabulary."""
        return super().vocabulary_size() + 1

    def get_vocabulary(self):
        """Get the size of the tokenizer vocabulary."""
        vocabulary = tensor_to_string_list(
            self._sentence_piece.id_to_string(
                tf.range(super().vocabulary_size())
            )
        )
        return self._vocabulary_prefix + vocabulary[3:]

    def id_to_token(self, id):
        """Convert an integer id to a string token."""
        if id < len(self._vocabulary_prefix):
            return self._vocabulary_prefix[id]

        return tensor_to_string_list(self._sentence_piece.id_to_string(id - 1))

    def token_to_id(self, token):
        """Convert a string token to an integer id."""
        if token in self._vocabulary_prefix:
            return self._vocabulary_prefix.index(token)

        return int(self._sentence_piece.string_to_id(token).numpy()) + 1

    def tokenize(self, inputs):
        tokens = super().tokenize(inputs)

        # Correct `unk_token_id` (0 -> 3). Note that we do not correct
        # `start_token_id` and `end_token_id`; they are dealt with in
        # `XLMRobertaPreprocessor`.
        tokens = tf.where(tf.equal(tokens, 0), self.unk_token_id - 1, tokens)

        # Shift the tokens IDs right by one.
        return tf.add(tokens, 1)

    def detokenize(self, inputs):
        if inputs.dtype == tf.string:
            return super().detokenize(inputs)

        # Shift the tokens IDs left by one.
        tokens = tf.subtract(inputs, 1)

        # Correct `unk_token_id`, `end_token_id`, `start_token_id`, respectively.
        # Note: The `pad_token_id` is taken as 0 (`unk_token_id`) since the
        # proto does not contain `pad_token_id`. This mapping of the pad token
        # is done automatically by the above subtraction.
        tokens = tf.where(tf.equal(tokens, self.unk_token_id - 1), 0, tokens)
        tokens = tf.where(tf.equal(tokens, self.end_token_id - 1), 2, tokens)
        tokens = tf.where(tf.equal(tokens, self.start_token_id - 1), 1, tokens)

        # Note: Even though we map `"<s>" and `"</s>"` to the correct IDs,
        # the `detokenize` method will return empty strings for these tokens.
        # This is a vagary of the `sentencepiece` library.
        return super().detokenize(tokens)

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    @classmethod
    @format_docstring(names=", ".join(backbone_presets))
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        """Instantiate an XLM-RoBERTa tokenizer from preset vocabulary.

        Args:
            preset: string. Must be one of {{names}}.

        Examples:
        ```python
        # Load a preset tokenizer.
        tokenizer = keras_nlp.models.XLMRobertaTokenizer.from_preset(
            "xlm_roberta_base",
        )

        # Tokenize some input.
        tokenizer("The quick brown fox tripped.")

        # Detokenize some input.
        tokenizer.detokenize([5, 6, 7, 8, 9])
        ```
        """
        if preset not in cls.presets:
            raise ValueError(
                "`preset` must be one of "
                f"""{", ".join(cls.presets)}. Received: {preset}."""
            )
        metadata = cls.presets[preset]

        spm_proto = keras.utils.get_file(
            "vocab.spm",
            metadata["spm_proto_url"],
            cache_subdir=os.path.join("models", preset),
            file_hash=metadata["spm_proto_hash"],
        )

        config = metadata["preprocessor_config"]
        config.update(
            {
                "proto": spm_proto,
            },
        )

        return cls.from_config({**config, **kwargs})
