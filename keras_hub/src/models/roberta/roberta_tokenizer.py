# Copyright 2024 The KerasHub Authors
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


from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.roberta.roberta_backbone import RobertaBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.RobertaTokenizer",
        "keras_hub.models.RobertaTokenizer",
    ]
)
class RobertaTokenizer(BytePairTokenizer):
    """A RoBERTa tokenizer using Byte-Pair Encoding subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.BytePairTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by RoBERTa
    models and provides a `from_preset()` method to automatically download
    a matching vocabulary for a RoBERTa preset.

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
    tokenizer = keras_hub.models.RobertaTokenizer.from_preset(
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
    tokenizer = keras_hub.models.RobertaTokenizer(
        vocabulary=vocab,
        merges=merges
    )
    tokenizer(["a quick fox", "a fox quick"])
    ```
    """

    backbone_cls = RobertaBackbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        **kwargs,
    ):
        self._add_special_token("<s>", "start_token")
        self._add_special_token("</s>", "end_token")
        self._add_special_token("<pad>", "pad_token")
        self._add_special_token("<mask>", "mask_token")
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )
