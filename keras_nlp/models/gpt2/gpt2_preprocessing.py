# Copyright 2022 The KerasNLP Authors
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
"""GPT-2 preprocessing layers."""

import copy
import os

from tensorflow import keras

from keras_nlp.models.gpt2.gpt2_presets import backbone_presets
from keras_nlp.tokenizers.byte_pair_tokenizer import BytePairTokenizer
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring


@keras.utils.register_keras_serializable(package="keras_nlp")
class GPT2Tokenizer(BytePairTokenizer):
    """A GPT-2 tokenizer using Byte-Pair Encoding subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_nlp.tokenizers.BytePairTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by GPT-2
    models and provides a `from_preset()` method to automatically download
    a matching vocabulary for a GPT-2 preset.

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

    Examples:

    Batched inputs.
    >>> vocab = {"<|endoftext|>": 0, "reful":1, "gent": 2, "Ġafter": 3}
    >>> vocab = {**vocab, **{"noon": 4, "Ġsun": 5}}
    >>> merges = ["Ġ a", "Ġ s", "r e", "f u", "g e", "n t", "e r", "n o", "o n"]
    >>> merges += ["i g", "h t", "Ġs u", "Ġa f", "ge nt", "no on", "re fu"]
    >>> merges += ["Ġsu n", "Ġaf t", "refu l", "Ġaft er"] # Ġ for whitespace
    >>> inputs = [" afternoon sun", "refulgent sun"]
    >>> tokenizer = keras_nlp.models.GPT2Tokenizer(
    ...     vocabulary=vocab,
    ...     merges=merges,
    ... )
    >>> tokenizer(inputs)
    <tf.RaggedTensor [[3, 4, 5], [1, 2, 5]]>

    Unbatched input.
    >>> vocab = {"<|endoftext|>": 0, "Ġafter": 1, "noon": 2, "Ġsun": 3}
    >>> merges = ["Ġ a", "Ġ s", "e r", "n o", "o n", "i g", "h t", "Ġs u"]
    >>> merges += ["Ġa f", "no on", "Ġsu n", "Ġaf t", "Ġaft er"]
    >>> inputs = " afternoon sun"
    >>> tokenizer = keras_nlp.models.GPT2Tokenizer(
    ...     vocabulary=vocab,
    ...     merges=merges,
    ... )
    >>> tokenizer(inputs)
    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 3], dtype=int32)>

    Detokenization.
    >>> vocab = {"<|endoftext|>": 0, "Ġafter": 1, "noon": 2, "Ġsun": 3}
    >>> merges = ["Ġ a", "Ġ s", "e r", "n o", "o n", "i g", "h t", "Ġs u"]
    >>> merges += ["Ġa f", "no on", "Ġsu n", "Ġaf t", "Ġaft er"]
    >>> inputs = " afternoon sun"
    >>> tokenizer = keras_nlp.models.GPT2Tokenizer(
    ...     vocabulary=vocab,
    ...     merges=merges,
    ... )
    >>> tokenizer.detokenize(tokenizer.tokenize(inputs)).numpy().decode('utf-8')
    ' afternoon sun'
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
        end_token = "<|endoftext|>"

        if end_token not in self.get_vocabulary():
            raise ValueError(
                f"Cannot find token `'{end_token}'` in the provided "
                f"`vocabulary`. Please provide `'{end_token}'` in your "
                "`vocabulary` or use a pretrained `vocabulary` name."
            )

        self.end_token_id = self.token_to_id(end_token)

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
        """Instantiate a GPT-2 tokenizer from preset vocabulary and merge rules.

        Args:
            preset: string. Must be one of {{names}}.

        Examples:
        ```python
        # Load a preset tokenizer.
        tokenizer = keras_nlp.models.GPT2Tokenizer.from_preset(
            "gpt2_base",
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

        vocabulary = keras.utils.get_file(
            "vocab.json",
            metadata["vocabulary_url"],
            cache_subdir=os.path.join("models", preset),
            file_hash=metadata["vocabulary_hash"],
        )
        merges = keras.utils.get_file(
            "merges.txt",
            metadata["merges_url"],
            cache_subdir=os.path.join("models", preset),
            file_hash=metadata["merges_hash"],
        )

        config = metadata["preprocessor_config"]
        config.update(
            {
                "vocabulary": vocabulary,
                "merges": merges,
            },
        )

        return cls.from_config({**config, **kwargs})
