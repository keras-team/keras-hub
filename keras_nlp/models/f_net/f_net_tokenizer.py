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

"""FNet tokenizer."""

import copy
import os

from tensorflow import keras

from keras_nlp.models.f_net.f_net_presets import backbone_presets
from keras_nlp.tokenizers.sentence_piece_tokenizer import SentencePieceTokenizer
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring


@keras.utils.register_keras_serializable(package="keras_nlp")
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
    tokenizer = keras_nlp.models.FNetTokenizer(proto="model.spm")

    # Batched inputs.
    tokenizer(["the quick brown fox", "the earth is round"])

    # Unbatched inputs.
    tokenizer("the quick brown fox")

    # Detokenization.
    tokenizer.detokenize(tf.constant([[2, 14, 2231, 886, 2385, 3]]))
    ```
    """

    def __init__(self, proto, **kwargs):
        super().__init__(proto=proto, **kwargs)

        # Check for necessary special tokens.
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        pad_token = "<pad>"
        for token in [cls_token, sep_token, pad_token]:
            if token not in self.get_vocabulary():
                raise ValueError(
                    f"Cannot find token `'{token}'` in the provided "
                    f"`vocabulary`. Please provide `'{token}'` in your "
                    "`vocabulary` or use a pretrained `vocabulary` name."
                )

        self.cls_token_id = self.token_to_id(cls_token)
        self.sep_token_id = self.token_to_id(sep_token)
        self.pad_token_id = self.token_to_id(pad_token)

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
        """Instantiate an FNet tokenizer from preset vocabulary.

        Args:
            preset: string. Must be one of {{names}}.

        Examples:
        ```python
        # Load a preset tokenizer.
        tokenizer = keras_nlp.models.FNetTokenizer.from_preset(
            "f_net_base_en",
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
