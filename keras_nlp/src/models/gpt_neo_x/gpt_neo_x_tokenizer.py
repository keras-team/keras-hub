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


@keras_nlp_export("keras_nlp.models.GPTNeoXTokenizer")
class GPTNeoXTokenizer(BytePairTokenizer):
    """A GPTNeoX tokenizer using Byte-Pair Encoding subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_nlp.tokenizers.BytePairTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by GPTNeoX
    models and provides a `from_preset()` method to automatically download
    a matching vocabulary for a GPTNeoX preset.

    This tokenizer does not provide truncation or padding of inputs.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        vocabulary: string or dict, maps token to integer ids. If it is a
            string, it should be the file path to a json file.
        merges: string or list, contains the merge rule. If it is a string,
            it should be the file path to merge rules. The merge rule file
            should have one merge rule per line. Every merge rule contains
            merge entities separated by a space.
    """

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        **kwargs,
    ):
        # GPTNeoX uses the same start as end token, i.e., "<|endoftext|>".
        self.end_token = self.start_token = "<|endoftext|>"

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            unsplittable_tokens=[self.end_token],
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

            self.end_token_id = self.token_to_id(self.end_token)
            self.start_token_id = self.end_token_id
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
