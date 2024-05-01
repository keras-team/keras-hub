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


try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "To use `keras_nlp`, please install Tensorflow: `pip install tensorflow`. "
        "The TensorFlow package is required for data preprocessing with any backend."
    )

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)


@keras_nlp_export("keras_nlp.models.DebertaV3Tokenizer")
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

    Note: The mask token (`"[MASK]"`) is handled differently in this tokenizer.
    If the token is not present in the provided SentencePiece vocabulary, the
    token will be appended to the vocabulary. For example, if the vocabulary
    size is 100, the mask token will be assigned the ID 100.

    Args:
        proto: Either a `string` path to a SentencePiece proto file, or a
            `bytes` object with a serialized SentencePiece proto. See the
            [SentencePiece repository](https://github.com/google/sentencepiece)
            for more details on the format.

    Examples:

    ```python
    # Unbatched input.
    tokenizer = keras_nlp.models.DebertaV3Tokenizer.from_preset(
        "deberta_v3_base_en",
    )
    tokenizer("The quick brown fox jumped.")

    # Batched inputs.
    tokenizer(["the quick brown fox", "the earth is round"])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))

    # Custom vocabulary.
    bytes_io = io.BytesIO()
    ds = tf.data.Dataset.from_tensor_slices(["The quick brown fox jumped."])
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=ds.as_numpy_iterator(),
        model_writer=bytes_io,
        vocab_size=9,
        model_type="WORD",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="[PAD]",
        bos_piece="[CLS]",
        eos_piece="[SEP]",
        unk_piece="[UNK]",
    )
    tokenizer = keras_nlp.models.DebertaV3Tokenizer(
        proto=bytes_io.getvalue(),
    )
    tokenizer("The quick brown fox jumped.")
    ```
    """

    def __init__(self, proto, **kwargs):
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"

        super().__init__(proto=proto, **kwargs)

    def set_proto(self, proto):
        super().set_proto(proto)
        if proto is not None:
            for token in [self.cls_token, self.pad_token, self.sep_token]:
                if token not in super().get_vocabulary():
                    raise ValueError(
                        f"Cannot find token `'{token}'` in the provided "
                        f"`vocabulary`. Please provide `'{token}'` in your "
                        "`vocabulary` or use a pretrained `vocabulary` name."
                    )

            self.cls_token_id = self.token_to_id(self.cls_token)
            self.sep_token_id = self.token_to_id(self.sep_token)
            self.pad_token_id = self.token_to_id(self.pad_token)
            # If the mask token is not in the vocabulary, add it to the end of the
            # vocabulary.
            if self.mask_token in super().get_vocabulary():
                self.mask_token_id = super().token_to_id(self.mask_token)
            else:
                self.mask_token_id = super().vocabulary_size()
        else:
            self.cls_token_id = None
            self.sep_token_id = None
            self.pad_token_id = None
            self.mask_token_id = None

    def vocabulary_size(self):
        sentence_piece_size = super().vocabulary_size()
        if sentence_piece_size == self.mask_token_id:
            return sentence_piece_size + 1
        return sentence_piece_size

    def get_vocabulary(self):
        sentence_piece_vocabulary = super().get_vocabulary()
        if self.mask_token_id < super().vocabulary_size():
            return sentence_piece_vocabulary
        return sentence_piece_vocabulary + ["[MASK]"]

    def id_to_token(self, id):
        if id == self.mask_token_id:
            return "[MASK]"
        return super().id_to_token(id)

    def token_to_id(self, token):
        if token == "[MASK]":
            return self.mask_token_id
        return super().token_to_id(token)

    def detokenize(self, ids):
        ids = tf.ragged.boolean_mask(ids, tf.not_equal(ids, self.mask_token_id))
        return super().detokenize(ids)
