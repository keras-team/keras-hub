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


from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)


@keras_nlp_export("keras_nlp.models.FNetTokenizer")
class FNetTokenizer(SentencePieceTokenizer):
    """FNet tokenizer layer based on SentencePiece.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_nlp.tokenizers.SentencePieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    FNet models and provides a `from_preset()` method to automatically
    download a matching vocabulary for a FNet preset.

    This tokenizer does not provide truncation or padding of inputs. It can be
    combined with a `keras_nlp.models.FNetPreprocessor` layer for input
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
    tokenizer = keras_nlp.models.FNetTokenizer.from_preset(
        "f_net_base_en",
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
    ```
    """

    def __init__(self, proto, **kwargs):
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = "<pad>"
        self.mask_token = "[MASK]"
        super().__init__(proto=proto, **kwargs)

    def set_proto(self, proto):
        super().set_proto(proto)
        if proto is not None:
            for token in [
                self.cls_token,
                self.sep_token,
                self.pad_token,
                self.mask_token,
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
        else:
            self.cls_token_id = None
            self.sep_token_id = None
            self.pad_token_id = None
            self.mask_token_id = None
