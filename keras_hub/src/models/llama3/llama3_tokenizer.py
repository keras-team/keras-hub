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
from keras_hub.src.models.llama3.llama3_backbone import Llama3Backbone
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_hub_export(
    [
        "keras_hub.tokenizers.Llama3Tokenizer",
        "keras_hub.models.Llama3Tokenizer",
    ]
)
class Llama3Tokenizer(BytePairTokenizer):
    backbone_cls = Llama3Backbone

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        **kwargs,
    ):
        self._add_special_token("<|begin_of_text|>", "start_token")
        self._add_special_token("<|end_of_text|>", "end_token")
        self.pad_token_id = 0
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )
