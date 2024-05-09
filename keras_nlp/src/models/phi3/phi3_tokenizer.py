# Copyright 2024 The KerasNLP Authors
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
import copy

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.phi3.phi3_presets import backbone_presets
from keras_nlp.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)
from keras_nlp.src.utils.python_utils import classproperty


@keras_nlp_export("keras_nlp.models.Phi3Tokenizer")
class Phi3Tokenizer(SentencePieceTokenizer):
    """Phi3 tokenizer layer based on SentencePiece.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_nlp.tokenizers.SentencePieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    Phi3 models and provides a `from_preset()` method to automatically
    download a matching vocabulary for a Phi3 preset.

    This tokenizer does not provide truncation or padding of inputs. It can be
    combined with a `keras_nlp.models.Phi3Preprocessor` layer for input
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
    tokenizer = keras_nlp.models.Phi3Tokenizer.from_preset(
        "phi3_mini_4k_instruct_en",
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
    ```
    """

    def __init__(self, proto, **kwargs):
        self.start_token = "<s>"
        self.end_token = "<|endoftext|>"
        super().__init__(proto=proto, **kwargs)

    def set_proto(self, proto):
        super().set_proto(proto)
        if proto is not None:
            for token in [self.start_token, self.end_token]:
                if token not in self.get_vocabulary():
                    raise ValueError(
                        f"Cannot find token `'{token}'` in the provided "
                        f"`vocabulary`. Please provide `'{token}'` in your "
                        "`vocabulary` or use a pretrained `vocabulary` name."
                    )
            self.start_token_id = self.token_to_id(self.start_token)
            self.end_token_id = self.token_to_id(self.end_token)
            # TODO: `pad_token` is `<|endoftext|>`, but setting it to `<unk>`
            # for now, because of the way sampler works. sampler will think that
            # `pad_token` is `end_token` and stop generation immediatly.
            self.pad_token_id = 0
        else:
            self.start_token_id = None
            self.end_token_id = None
            self.pad_token_id = None

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)
