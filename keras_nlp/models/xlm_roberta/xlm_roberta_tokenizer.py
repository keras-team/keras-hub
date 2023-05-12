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

"""XLM-RoBERTa tokenizer."""

import copy

import tensorflow as tf

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.models.xlm_roberta.xlm_roberta_presets import backbone_presets
from keras_nlp.tokenizers.sentence_piece_tokenizer import SentencePieceTokenizer
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.tf_utils import tensor_to_string_list


@keras_nlp_export("keras_nlp.models.XLMRobertaTokenizer")
class XLMRobertaTokenizer(SentencePieceTokenizer):
    """An XLM-RoBERTa tokenizer using SentencePiece subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_nlp.tokenizers.SentencePieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    XLM-RoBERTa models and provides a `from_preset()` method to automatically
    download a matching vocabulary for an XLM-RoBERTa preset.

    Note: If you are providing your own custom SentencePiece model, the original
    fairseq implementation of XLM-RoBERTa re-maps some token indices from the
    underlying sentencepiece output. To preserve compatibility, we do the same
    re-mapping here.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        proto: Either a `string` path to a SentencePiece proto file or a
            `bytes` object with a serialized SentencePiece proto. See the
            [SentencePiece repository](https://github.com/google/sentencepiece)
            for more details on the format.

    Examples:
    ```python
    tokenizer = keras_nlp.models.XLMRobertaTokenizer.from_preset(
        "xlm_roberta_base_multi",
    )

    # Unbatched inputs.
    tokenizer("the quick brown fox")

    # Batched inputs.
    tokenizer(["the quick brown fox", "الأرض كروية"])

    # Detokenization.
    tokenizer.detokenize(tokenizer("the quick brown fox"))

    # Custom vocabulary
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
        self.mask_token_id = self.vocabulary_size() - 1  # <mask>

    def vocabulary_size(self):
        """Get the size of the tokenizer vocabulary."""
        return super().vocabulary_size() + 2

    def get_vocabulary(self):
        """Get the size of the tokenizer vocabulary."""
        vocabulary = tensor_to_string_list(
            self._sentence_piece.id_to_string(
                tf.range(super().vocabulary_size())
            )
        )
        return self._vocabulary_prefix + vocabulary[3:] + ["<mask>"]

    def id_to_token(self, id):
        """Convert an integer id to a string token."""

        if id == self.mask_token_id:
            return "<mask>"

        if id < len(self._vocabulary_prefix):
            return self._vocabulary_prefix[id]

        return tensor_to_string_list(self._sentence_piece.id_to_string(id - 1))

    def token_to_id(self, token):
        """Convert a string token to an integer id."""

        if token in self._vocabulary_prefix:
            return self._vocabulary_prefix.index(token)

        spm_token_id = self._sentence_piece.string_to_id(token)

        # OOV token
        spm_unk_token_id = self._sentence_piece.string_to_id("<unk>")
        if spm_token_id == spm_unk_token_id:
            return self.unk_token_id

        return int(spm_token_id.numpy()) + 1

    def tokenize(self, inputs):
        tokens = super().tokenize(inputs)

        # Correct `unk_token_id` (0 -> 3). Note that we do not correct
        # `start_token_id` and `end_token_id`; they are dealt with in
        # `XLMRobertaPreprocessor`.
        tokens = tf.where(tf.equal(tokens, 0), self.unk_token_id - 1, tokens)

        # Shift the tokens IDs right by one.
        return tf.add(tokens, 1)

    def detokenize(self, ids):
        ids = tf.ragged.boolean_mask(ids, tf.not_equal(ids, self.mask_token_id))
        return super().detokenize(ids)

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)
