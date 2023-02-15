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

"""BART tokenizer."""

import copy

from tensorflow import keras

from keras_nlp.models.bart.bart_presets import backbone_presets
from keras_nlp.tokenizers.byte_pair_tokenizer import BytePairTokenizer
from keras_nlp.utils.python_utils import classproperty


@keras.utils.register_keras_serializable(package="keras_nlp")
class BartTokenizer(BytePairTokenizer):
    """A BART tokenizer using Byte-Pair Encoding subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_nlp.tokenizers.BytePairTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by BART
    models and provides a `from_preset()` method to automatically download
    a matching vocabulary for a BART preset.

    This tokenizer does not provide truncation or padding of inputs. It can be
    combined with a `keras_nlp.models.BartPreprocessor` layer for input
    packing.

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

    Examples:

    Batched inputs.
    >>> vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<mask>": 3}
    >>> vocab = {**vocab, "a": 4, "Ġquick": 5, "Ġfox": 6}
    >>> merges = ["Ġ q", "u i", "c k", "ui ck", "Ġq uick"]
    >>> merges += ["Ġ f", "o x", "Ġf ox"]
    >>> tokenizer = keras_nlp.models.RobertaTokenizer(
    ...     vocabulary=vocab, merges=merges
    ... )
    >>> tokenizer(["a quick fox", "a fox quick"])
    <tf.RaggedTensor [[4, 5, 6], [4, 6, 5]]>

    Unbatched input.
    >>> vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<mask>": 3}
    >>> vocab = {**vocab, "a": 4, "Ġquick": 5, "Ġfox": 6}
    >>> merges = ["Ġ q", "u i", "c k", "ui ck", "Ġq uick"]
    >>> merges += ["Ġ f", "o x", "Ġf ox"]
    >>> tokenizer = keras_nlp.models.RobertaTokenizer(
    ...     vocabulary=vocab, merges=merges
    ... )
    >>> tokenizer("a quick fox")
    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([4, 5, 6], dtype=int32)>

    Detokenization.
    >>> vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<mask>": 3}
    >>> vocab = {**vocab, "a": 4, "Ġquick": 5, "Ġfox": 6}
    >>> merges = ["Ġ q", "u i", "c k", "ui ck", "Ġq uick"]
    >>> merges += ["Ġ f", "o x", "Ġf ox"]
    >>> tokenizer = keras_nlp.models.RobertaTokenizer(
    ...     vocabulary=vocab, merges=merges
    ... )
    >>> tokenizer.detokenize(tokenizer("a quick fox")).numpy().decode('utf-8')
    'a quick fox'
    """

    def __init__(
        self,
        vocabulary,
        merges,
        **kwargs,
    ):
        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            **kwargs,
        )

        # Check for necessary special tokens.
        start_token = "<s>"
        pad_token = "<pad>"
        end_token = "</s>"
        for token in [start_token, pad_token, end_token]:
            if token not in self.get_vocabulary():
                raise ValueError(
                    f"Cannot find token `'{token}'` in the provided "
                    f"`vocabulary`. Please provide `'{token}'` in your "
                    "`vocabulary` or use a pretrained `vocabulary` name."
                )

        self.start_token_id = self.token_to_id(start_token)
        self.pad_token_id = self.token_to_id(pad_token)
        self.end_token_id = self.token_to_id(end_token)

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)
