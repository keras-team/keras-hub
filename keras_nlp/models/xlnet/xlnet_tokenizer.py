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

"""XLNET tokenizer."""

import tensorflow as tf

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.tokenizers.sentence_piece_tokenizer import SentencePieceTokenizer

try:
    import unicodedata
except ImportError:
    unicodedata = None


@keras_nlp_export("keras_nlp.models.XLNetTokenizer")
class XLNetTokenizer(SentencePieceTokenizer):
    """XLNET tokenizer layer based on SentencePiece.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_nlp.tokenizers.SentencePieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    XLNET models and provides a `from_preset()` method to automatically
    download a matching vocabulary for a XLNET preset.

    This tokenizer does not provide truncation or padding of inputs. It can be
    combined with a `keras_nlp.models.XLNetPreprocessor` layer for input
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
    # Unbatched input.
    tokenizer = keras_nlp.models.XLNetTokenizer(proto="<path to SentencePiece proto file>")
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))

    # Custom vocabulary.
    bytes_io = io.BytesIO()
    ds = tf.data.Dataset.from_tensor_slices(["The quick brown fox jumped."])
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=vocab_data.as_numpy_iterator(),
        model_writer=bytes_io,
        vocab_size=14,
        model_type="WORD",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="<pad>",
        bos_piece="<s>",
        eos_piece="</s>",
        unk_piece="<unk>",
        user_defined_symbols=["<mask>", "<cls>", "<sep>"]
    )
    tokenizer = keras_nlp.models.XLNetTokenizer(
        proto=bytes_io.getvalue(),
    )
    tokenizer("The quick brown fox jumped.")
    ```
    """

    def __init__(self, proto, **kwargs):
        super().__init__(proto=proto, **kwargs)

        # Check for necessary special tokens.
        self.cls_token = "<cls>"
        self.sep_token = "<sep>"
        self.pad_token = "<pad>"
        self.mask_token = "<mask>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"

        for token in [
            self.cls_token,
            self.sep_token,
            self.pad_token,
            self.mask_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
        ]:
            if token not in self.get_vocabulary():
                raise ValueError(
                    f"Cannot find token `'{token}'` in the provided "
                    f"`vocabulary`. Please provide `'{token}'` in your "
                    "`vocabulary` or use a pretrained `vocabulary` name."
                )

        self.cls_token_id = self.token_to_id(self.cls_token)
        self.sep_token_id = self.token_to_id(self.sep_token)
        self.pad_token_id = self.token_to_id(self.pad_token)
        self.mask_token_id = self.token_to_id(self.mask_token)
        self.bos_token_id = self.token_to_id(self.bos_token)
        self.eos_token_id = self.token_to_id(self.eos_token)
        self.unk_token_id = self.token_to_id(self.unk_token)

    def preprocess_text(self, inputs):
        """Preprocesses the text. This method removes spaces and accents from the text."""

        # remove space
        outputs = " ".join(inputs.strip().split())
        outputs = outputs.replace("``", '"').replace("''", '"')

        # remove accents
        outputs = unicodedata.normalize("NFKD", outputs)
        outputs = "".join([c for c in outputs if not unicodedata.combining(c)])

        return outputs

    def tokenize(self, text):
        """Tokenize a string."""

        # check if there are multiple examples present or not
        is_batched = isinstance(text, list)
        if not is_batched:
            text = [text]

        tokenized_text = []
        for each_text in text:
            each_text = self.preprocess_text(each_text)
            pieces = [
                self.id_to_token(token_id)
                for token_id in super().tokenize(each_text)
            ]

            new_pieces = []
            for piece in pieces:
                if (
                    len(piece) > 1
                    and piece[-1] == str(",")
                    and piece[-2].isdigit()
                ):
                    cur_pieces = [
                        self.id_to_token(cur_piece_id)
                        for cur_piece_id in super().tokenize(
                            piece[:-1].replace("▁", "")
                        )
                    ]
                    if piece[0] != "▁" and cur_pieces[0][0] == "▁":
                        if len(cur_pieces[0]) == 1:
                            cur_pieces = cur_pieces[1:]
                        else:
                            cur_pieces[0] = cur_pieces[0][1:]
                    cur_pieces.append(piece[-1])
                    new_pieces.extend(cur_pieces)
                else:
                    new_pieces.append(piece)

            new_pieces = [
                self.token_to_id(new_piece_token)
                for new_piece_token in new_pieces
            ]
            # add sep_token and cls_token in the end.
            new_pieces.extend([self.sep_token_id, self.cls_token_id])

            tokenized_text.append(new_pieces)

        # if there are multiple examples present, then return a `tf.RaggedTensor`.
        if is_batched:
            return tf.ragged.constant(tokenized_text)

        return tokenized_text[0]

    def detokenize(self, inputs):
        """Detokenize the input_ids into text."""

        outputs = super().detokenize(inputs)

        outputs = tf.strings.regex_replace(outputs, self.cls_token, "")
        outputs = tf.strings.regex_replace(outputs, self.sep_token, "")
        outputs = tf.strings.regex_replace(outputs, self.mask_token, "")
        outputs = tf.strings.regex_replace(outputs, self.pad_token, "")

        return outputs
