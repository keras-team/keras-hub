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

"""DeBERTa tokenizer."""

import copy

from tensorflow import keras

from keras_nlp.models.deberta_v3.deberta_v3_presets import backbone_presets
from keras_nlp.tokenizers.sentence_piece_tokenizer import SentencePieceTokenizer
from keras_nlp.utils.python_utils import classproperty


@keras.utils.register_keras_serializable(package="keras_nlp")
class DebertaV3Tokenizer(SentencePieceTokenizer):
    """DeBERTa tokenizer layer based on SentencePiece.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_nlp.tokenizers.SentencePieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    DeBERTa models and provides a `from_preset()` method to automatically
    download a matching vocabulary for a DeBERTa preset.

    This tokenizer does not provide truncation or padding of inputs. It can be
    combined with a `keras_nlp.models.DebertaV3Preprocessor` layer for input
    packing.

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
    tokenizer = keras_nlp.models.DebertaV3Tokenizer(proto="model.spm")

    # Batched inputs.
    tokenizer(["the quick brown fox", "the earth is round"])

    # Unbatched inputs.
    tokenizer("the quick brown fox")

    # Detokenization.
    tokenizer.detokenize(tf.constant([[1, 4, 9, 5, 7, 2]]))
    ```
    """

    def __init__(self, proto, mask_token_id=None, **kwargs):
        super().__init__(proto=proto, **kwargs)

        # Maintain a private copy of `mask_token_id` for config purposes.
        self._mask_token_id = mask_token_id

        # Maintain a private copy of the original vocabulary; the parent class's
        # `get_vocabulary()` function calls `self.vocabulary_size()`, which
        # throws up a segmentation fault.
        self._original_vocabulary = super().get_vocabulary()

        # Check for necessary special tokens.
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        pad_token = "[PAD]"
        mask_token = "[MASK]"

        in_vocab_special_tokens = [cls_token, pad_token, sep_token]
        if mask_token_id is None:
            in_vocab_special_tokens = in_vocab_special_tokens + [mask_token]

        for token in in_vocab_special_tokens:
            if token not in self._original_vocabulary:
                raise ValueError(
                    f"Cannot find token `'{token}'` in the provided "
                    f"`vocabulary`. Please provide `'{token}'` in your "
                    "`vocabulary` or use a pretrained `vocabulary` name."
                )

        self.cls_token_id = self.token_to_id(cls_token)
        self.sep_token_id = self.token_to_id(sep_token)
        self.pad_token_id = self.token_to_id(pad_token)
        self.mask_token_id = mask_token_id
        if mask_token_id is None:
            self.mask_token_id = self.token_to_id(mask_token)

    def vocabulary_size(self):
        vocabulary_size = super().vocabulary_size()

        # This is to avoid an error when `super.get_vocabulary()` is called
        # in `__init__()`.
        if not hasattr(self, "mask_token_id"):
            return vocabulary_size

        if self.mask_token_id >= vocabulary_size:
            return self.mask_token_id + 1
        return vocabulary_size

    def get_vocabulary(self):
        vocabulary = self._original_vocabulary
        if self.mask_token_id >= len(vocabulary):
            vocabulary = vocabulary + [None] * (
                self.mask_token_id - len(vocabulary) + 1
            )
        vocabulary[self.mask_token_id] = "[MASK]"
        return vocabulary

    def id_to_token(self, id):
        if id == self.mask_token_id:
            return "[MASK]"
        return super().id_to_token(id)

    def token_to_id(self, token):
        if token == "[MASK]":
            return self.mask_token_id
        return int(self._sentence_piece.string_to_id(token).numpy())

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mask_token_id": self._mask_token_id,
            }
        )
        return config

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)
