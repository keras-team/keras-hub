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

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.tokenizers import WordPieceTokenizer


@keras_nlp_export("keras_nlp.models.ElectraTokenizer")
class ElectraTokenizer(WordPieceTokenizer):
    """A ELECTRA tokenizer using WordPiece subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_nlp.tokenizers.WordPieceTokenizer`.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        vocabulary: A list of strings or a string filename path. If
            passing a list, each element of the list should be a single word
            piece token string. If passing a filename, the file should be a
            plain text file containing a single word piece token per line.
        lowercase: If `True`, the input text will be first lowered before
            tokenization.
        special_tokens_in_strings: bool. A bool to indicate if the tokenizer
            should expect special tokens in input strings that should be
            tokenized and mapped correctly to their ids. Defaults to False.

    Examples:
    ```python
    # Custom Vocabulary.
    vocab = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    vocab += ["The", "quick", "brown", "fox", "jumped", "."]

    # Instantiate the tokenizer.
    tokenizer = keras_nlp.models.ElectraTokenizer(vocabulary=vocab)

    # Unbatched input.
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
    ```
    """

    def __init__(
        self,
        vocabulary,
        lowercase=False,
        special_tokens_in_strings=False,
        **kwargs,
    ):
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        super().__init__(
            vocabulary=vocabulary,
            lowercase=lowercase,
            special_tokens=[
                self.cls_token,
                self.sep_token,
                self.pad_token,
                self.mask_token,
            ],
            special_tokens_in_strings=special_tokens_in_strings,
            **kwargs,
        )

    def set_vocabulary(self, vocabulary):
        super().set_vocabulary(vocabulary)

        if vocabulary is not None:
            self.cls_token_id = self.token_to_id(self.cls_token)
            self.sep_token_id = self.token_to_id(self.sep_token)
            self.pad_token_id = self.token_to_id(self.pad_token)
            self.mask_token_id = self.token_to_id(self.mask_token)
        else:
            self.cls_token_id = None
            self.sep_token_id = None
            self.pad_token_id = None
            self.mask_token_id = None

    def get_config(self):
        config = super().get_config()
        del config["special_tokens"]  # Not configurable; set in __init__.
        return config
