# Copyright 2024 The KerasHub Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""GptOss tokenizer."""

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.gpt_oss.gpt_oss_backbone import GptOssBackbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.GptOssTokenizer",
        "keras_hub.models.GptOssTokenizer",
    ]
)
class GptOssTokenizer(BytePairTokenizer):
    """A GptOss tokenizer using BytePair encoding.

    Tokenizer is a subclass of `keras_hub.tokenizers.BytePairTokenizer`.
    It uses a BytePair encoding model to tokenize strings. It also adds special
    tokens for the start and end of a sequence.

    Args:
        vocabulary: string or dict, maps token to integer ids. If it is a
            string, it should be the file path to a json file.
        merges: string or list, contains the merge rule. If it is a string,
            it should be the file path to merge rules.
    """

    backbone_cls = GptOssBackbone

    def __init__(
            self, 
            vocabulary=None, 
            merges=None,
            **kwargs
        ):
        """Initializes the GptOssTokenizer.

        Args:
            vocabulary: string or dict, maps token to integer ids.
            merges: string or list, contains the merge rule.
            **kwargs: Additional keyword arguments.
        """
        self._add_special_token("<|startoftext|>", "start_token")
        self._add_special_token("<|endoftext|>", "end_token")
        self.pad_token_id = 0
        super().__init__(vocabulary=vocabulary, merges=merges, **kwargs)
