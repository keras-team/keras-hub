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

import tensorflow as tf
from tensorflow import keras

from keras_nlp.models.deberta_v3.deberta_v3_presets import backbone_presets
from keras_nlp.tokenizers.sentence_piece_tokenizer import SentencePieceTokenizer
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.tf_utils import tensor_to_string_list


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
        mask_token_id: The token ID (int) of the mask token (`[MASK]`). If
            `None`, either the SentencePiece vocabulary has the mask
            token or the mask token is not required by the user. This argument
            should be non-None only if the SentencePiece vocabulary does not
            have the mask token. Preset tokenizers will be loaded with the
            correct mask token ID by default.

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

        if (
            mask_token_id is not None
            and mask_token_id < super().vocabulary_size()
        ):
            raise ValueError(
                "`mask_token_id` must be greater than or equal to the "
                f"SentencePiece vocabulary size, {self._original_vocabulary_size}. "
                f"Received `mask_token_id = {mask_token_id}`."
            )

        # Check for necessary special tokens.
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        pad_token = "[PAD]"
        mask_token = "[MASK]"

        # We do not check for the presence of `mask_token`; this will be
        # handled in the corresponding MaskedLM processor.
        for token in [cls_token, pad_token, sep_token]:
            if token not in self._get_sentence_piece_vocabulary():
                raise ValueError(
                    f"Cannot find token `'{token}'` in the provided "
                    f"`vocabulary`. Please provide `'{token}'` in your "
                    "`vocabulary` or use a pretrained `vocabulary` name."
                )

        self.cls_token_id = self.token_to_id(cls_token)
        self.sep_token_id = self.token_to_id(sep_token)
        self.pad_token_id = self.token_to_id(pad_token)
        self.mask_token_id = mask_token_id
        if (
            mask_token_id is None
            and mask_token in self._get_sentence_piece_vocabulary()
        ):
            self.mask_token_id = self.token_to_id(mask_token)

    def _get_sentence_piece_vocabulary(self):
        return tensor_to_string_list(
            self._sentence_piece.id_to_string(
                tf.range(super().vocabulary_size())
            )
        )

    def vocabulary_size(self):
        if self._mask_token_id is None:
            return super().vocabulary_size()
        return self._mask_token_id + 1

    def get_vocabulary(self):
        original_vocabulary = self._get_sentence_piece_vocabulary()
        if self._mask_token_id is None:
            return original_vocabulary

        return (
            original_vocabulary
            + [None] * (self._mask_token_id - super().vocabulary_size())
            + ["[MASK]"]
        )

    def id_to_token(self, id):
        if self._mask_token_id is not None and id == self._mask_token_id:
            return "[MASK]"
        return super().id_to_token(id)

    def token_to_id(self, token):
        if self._mask_token_id is not None and token == "[MASK]":
            return self._mask_token_id
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
