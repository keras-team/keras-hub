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


@keras_nlp_export("keras_nlp.models.T5Tokenizer")
class T5Tokenizer(SentencePieceTokenizer):
    """T5 tokenizer layer based on SentencePiece.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_nlp.tokenizers.SentencePieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    T5 models and provides a `from_preset()` method to automatically
    download a matching vocabulary for a T5 preset.

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
    bytes_io = io.BytesIO()
    ds = tf.data.Dataset.from_tensor_slices(["The quick brown fox jumped."])
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=ds.as_numpy_iterator(),
        model_writer=bytes_io,
        vocab_size=8,
        model_type="WORD",
        bos_id=-1,
        pad_id=0,
        eos_id=1,
        unk_id=2,
        pad_piece="<pad>",
        eos_piece="</s>",
        unk_piece="<unk>",
    )
    tokenizer = keras_nlp.models.T5Tokenizer(
        proto=bytes_io.getvalue(),
    )
    tokenizer("The quick brown fox jumped.")

    # Batched inputs.
    tokenizer(["the quick brown fox", "the earth is round"])

    # Unbatched inputs.
    tokenizer("the quick brown fox")

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
    ```
    """

    def __init__(self, proto, **kwargs):
        self.end_token = "</s>"
        self.pad_token = "<pad>"

        super().__init__(proto=proto, **kwargs)

    def set_proto(self, proto):
        super().set_proto(proto)
        if proto is not None:
            for token in [self.end_token, self.pad_token]:
                if token not in self.get_vocabulary():
                    raise ValueError(
                        f"Cannot find token `'{token}'` in the provided "
                        f"`vocabulary`. Please provide `'{token}'` in your "
                        "`vocabulary` or use a pretrained `vocabulary` name."
                    )
            self.end_token_id = self.token_to_id(self.end_token)
            self.pad_token_id = self.token_to_id(self.pad_token)
            # T5 uses the same start token as end token, i.e., "<\s>".
            self.start_token_id = self.end_token_id
        else:
            self.end_token_id = None
            self.pad_token_id = None
            self.start_token_id = None
