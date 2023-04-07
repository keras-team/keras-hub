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

"""RoBERTa tokenizer."""

import copy

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.models.roberta.roberta_presets import backbone_presets
from keras_nlp.tokenizers.byte_pair_tokenizer import BytePairTokenizer
from keras_nlp.utils.python_utils import classproperty


@keras_nlp_export("keras_nlp.models.RobertaTokenizer")
class RobertaTokenizer(BytePairTokenizer):
    """A RoBERTa tokenizer using Byte-Pair Encoding subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_nlp.tokenizers.BytePairTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by RoBERTa
    models and provides a `from_preset()` method to automatically download
    a matching vocabulary for a RoBERTa preset.

    This tokenizer does not provide truncation or padding of inputs. It can be
    combined with a `keras_nlp.models.RobertaPreprocessor` layer for input
    packing.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        vocabulary: A dictionary mapping tokens to integer ids, or file path
            to a json file containing the token to id mapping.
        merges: A list of merge rules or a string file path, If passing a file
            path. the file should have one merge rule per line. Every merge
            rule contains merge entities separated by a space.

    Examples:
    ```python
    # Unbatched input.
    tokenizer = keras_nlp.models.RobertaTokenizer.from_preset(
        "roberta_base_en",
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))

    # Custom vocabulary.
    # Note: 'Ġ' is space
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<mask>": 3}
    vocab = {**vocab, "a": 4, "Ġquick": 5, "Ġfox": 6}
    merges = ["Ġ q", "u i", "c k", "ui ck", "Ġq uick"]
    merges += ["Ġ f", "o x", "Ġf ox"]
    tokenizer = keras_nlp.models.RobertaTokenizer(
        vocabulary=vocab,
        merges=merges
    )
    tokenizer(["a quick fox", "a fox quick"])
    ```
    """

    def __init__(
        self,
        vocabulary,
        merges,
        **kwargs,
    ):
        # Special tokens.
        start_token = "<s>"
        pad_token = "<pad>"
        end_token = "</s>"
        mask_token = "<mask>"

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            unsplittable_tokens=[start_token, pad_token, end_token, mask_token],
            **kwargs,
        )

        # Check whether special tokens are present in the vocabulary.
        for token in [start_token, pad_token, end_token, mask_token]:
            if token not in self.get_vocabulary():
                raise ValueError(
                    f"Cannot find token `'{token}'` in the provided "
                    f"`vocabulary`. Please provide `'{token}'` in your "
                    "`vocabulary` or use a pretrained `vocabulary` name."
                )

        self.start_token_id = self.token_to_id(start_token)
        self.pad_token_id = self.token_to_id(pad_token)
        self.end_token_id = self.token_to_id(end_token)
        self.mask_token_id = self.token_to_id(mask_token)

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    def get_config(self):
        config = super().get_config()
        # In the constructor, we pass the list of special tokens to the
        # `unsplittable_tokens` arg of the superclass' constructor. Hence, we
        # delete it from the config here.
        del config["unsplittable_tokens"]
        return config
