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

"""RoBERTa tokenizer."""

import copy
import os

from tensorflow import keras

from keras_nlp.models.roberta.roberta_presets import backbone_presets
from keras_nlp.tokenizers.byte_pair_tokenizer import BytePairTokenizer
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring


@keras.utils.register_keras_serializable(package="keras_nlp")
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
        vocabulary: string or dict, maps token to integer ids. If it is a
            string, it should be the file path to a json file.
        merges: string or list, contains the merge rule. If it is a string,
            it should be the file path to merge rules. The merge rule file
            should have one merge rule per line. Every merge rule contains
            merge entities separated by a space.

    Examples:

    Batched inputs.
    >>> vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "reful": 3, "gent": 4}
    >>> vocab = {**vocab, **{"Ġafter": 5, "noon": 6, "Ġsun": 7}}
    >>> merges = ["Ġ a", "Ġ s", "r e", "f u", "g e", "n t"]
    >>> merges += ["e r", "n o", "o n", "i g", "h t"]
    >>> merges += ["Ġs u", "Ġa f", "ge nt", "no on", "re fu"]
    >>> merges += ["Ġsu n", "Ġaf t", "refu l", "Ġaft er"]
    >>> inputs = [" afternoon sun", "refulgent sun"]
    >>> tokenizer = keras_nlp.models.RobertaTokenizer(
    ...     vocabulary=vocab,
    ...     merges=merges,
    ... )
    >>> tokenizer(inputs)
    <tf.RaggedTensor [[5, 6, 7], [3, 4, 7]]>

    Unbatched input.
    >>> vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "Ġafter": 3, "noon": 4, "Ġsun": 5}
    >>> merges = ["Ġ a", "Ġ s", "e r", "n o", "o n", "i g", "h t", "Ġs u"]
    >>> merges += ["Ġa f", "no on", "Ġsu n", "Ġaf t", "Ġaft er"]
    >>> inputs = " afternoon sun"
    >>> tokenizer = keras_nlp.models.RobertaTokenizer(
    ...     vocabulary=vocab,
    ...     merges=merges,
    ... )
    >>> tokenizer(inputs)
    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([3, 4, 5], dtype=int32)>

    Detokenization.
    >>> vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "Ġafter": 3, "noon": 4, "Ġsun": 5}
    >>> merges = ["Ġ a", "Ġ s", "e r", "n o", "o n", "i g", "h t", "Ġs u"]
    >>> merges += ["Ġa f", "no on", "Ġsu n", "Ġaf t", "Ġaft er"]
    >>> inputs = " afternoon sun"
    >>> tokenizer = keras_nlp.models.RobertaTokenizer(
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

    @classmethod
    @format_docstring(names=", ".join(backbone_presets))
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        """Instantiate a RoBERTa tokenizer from preset vocabulary and merge rules.

        Args:
            preset: string. Must be one of {{names}}.

        Examples:
        ```python
        # Load a preset tokenizer.
        tokenizer = keras_nlp.models.RobertaTokenizer.from_preset(
            "roberta_base",
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
