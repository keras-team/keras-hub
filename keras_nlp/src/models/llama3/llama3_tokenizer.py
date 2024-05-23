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
from keras_nlp.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


@keras_nlp_export("keras_nlp.models.Llama3Tokenizer")
class Llama3Tokenizer(BytePairTokenizer):
    def __init__(
        self,
        vocabulary=None,
        merges=None,
        **kwargs,
    ):
        self.start_token = "<|begin_of_text|>"
        self.end_token = "<|end_of_text|>"

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            unsplittable_tokens=[self.start_token, self.end_token],
            **kwargs,
        )

    def set_vocabulary_and_merges(self, vocabulary, merges):
        super().set_vocabulary_and_merges(vocabulary, merges)

        if vocabulary is not None:
            # Check for necessary special tokens.
            if self.end_token not in self.get_vocabulary():
                raise ValueError(
                    f"Cannot find token `'{self.end_token}'` in the provided "
                    f"`vocabulary`. Please provide `'{self.end_token}'` in "
                    "your `vocabulary` or use a pretrained `vocabulary` name."
                )

            self.start_token_id = self.token_to_id(self.start_token)
            self.end_token_id = self.token_to_id(self.end_token)
            self.pad_token_id = 0
        else:
            self.end_token_id = None
            self.start_token_id = None
            self.pad_token_id = None

    def get_config(self):
        config = super().get_config()
        # In the constructor, we pass the list of special tokens to the
        # `unsplittable_tokens` arg of the superclass' constructor. Hence, we
        # delete it from the config here.
        del config["unsplittable_tokens"]
        return config
