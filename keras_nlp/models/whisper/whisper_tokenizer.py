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
"""Whisper tokenizer."""

import json

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.tokenizers.byte_pair_tokenizer import BytePairTokenizer


def _load_dict(dict_or_path):
    if isinstance(dict_or_path, str):
        with open(dict_or_path, "r") as f:
            dict_or_path = json.load(f)
    return dict_or_path


@keras_nlp_export("keras_nlp.models.WhisperTokenizer")
class WhisperTokenizer(BytePairTokenizer):
    """Whisper text tokenizer using Byte-Pair Encoding subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_nlp.tokenizers.BytePairTokenizer`.
    This tokenizer does not provide truncation or padding of inputs.

    Args:
        vocabulary: string or dict, maps token to integer ids. If it is a
            string, it should be the file path to a json file.
        merges: string or list, contains the merge rule. If it is a string,
            it should be the file path to merge rules. The merge rule file
            should have one merge rule per line. Every merge rule contains
            merge entities separated by a space.
        special_tokens: string or dict, maps special tokens to integer IDs. If
            it is a string, it should be the path to a JSON file.
        language_tokens: string or dict, maps language tokens to integer IDs. If
            non-None or empty, the tokenizer will be assumed to be a
            multilingual tokenizer.

    Examples:

    Batched inputs.
    >>> vocab = {"reful":0, "gent": 1, "Ġafter": 2, "noon": 3, "Ġsun": 4}
    >>> merges = ["Ġ a", "Ġ s", "r e", "f u", "g e", "n t", "e r", "n o", "o n"]
    >>> merges += ["i g", "h t", "Ġs u", "Ġa f", "ge nt", "no on", "re fu"]
    >>> merges += ["Ġsu n", "Ġaf t", "refu l", "Ġaft er"] # Ġ for whitespace
    >>> special_tokens = {"<|startoftranscript|>": 5, "<|endoftext|>": 6}
    >>> special_tokens = {**special_tokens, "<|notimestamps|>": 7, "<|transcribe|>": 8}
    >>> special_tokens = {**special_tokens, "<|translate|>": 9}
    >>> inputs = [" afternoon sun", "refulgent sun"]
    >>> tokenizer = keras_nlp.models.WhisperTokenizer(
    ...     vocabulary=vocab,
    ...     merges=merges,
    ...     special_tokens=special_tokens,
    ... )
    >>> tokenizer(inputs)
    <tf.RaggedTensor [[2, 3, 4], [0, 1, 4]]>

    Unbatched input.
    >>> vocab = {"Ġafter": 0, "noon": 1, "Ġsun": 2}
    >>> merges = ["Ġ a", "Ġ s", "e r", "n o", "o n", "i g", "h t", "Ġs u"]
    >>> merges += ["Ġa f", "no on", "Ġsu n", "Ġaf t", "Ġaft er"]
    >>> special_tokens = {"<|startoftranscript|>": 3, "<|endoftext|>": 4}
    >>> special_tokens = {**special_tokens, "<|notimestamps|>": 5, "<|transcribe|>": 6}
    >>> special_tokens = {**special_tokens, "<|translate|>": 7}
    >>> inputs = " afternoon sun"
    >>> tokenizer = keras_nlp.models.WhisperTokenizer(
    ...     vocabulary=vocab,
    ...     merges=merges,
    ...     special_tokens=special_tokens,
    ... )
    >>> tokenizer(inputs)
    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([0, 1, 2], dtype=int32)>

    Detokenization.
    >>> vocab = {"<|endoftext|>": 0, "Ġafter": 1, "noon": 2, "Ġsun": 3}
    >>> merges = ["Ġ a", "Ġ s", "e r", "n o", "o n", "i g", "h t", "Ġs u"]
    >>> merges += ["Ġa f", "no on", "Ġsu n", "Ġaf t", "Ġaft er"]
    >>> special_tokens = {"<|startoftranscript|>": 4, "<|endoftext|>": 5}
    >>> special_tokens = {**special_tokens, "<|notimestamps|>": 6, "<|transcribe|>": 7}
    >>> special_tokens = {**special_tokens, "<|translate|>": 8}
    >>> inputs = " afternoon sun"
    >>> tokenizer = keras_nlp.models.WhisperTokenizer(
    ...     vocabulary=vocab,
    ...     merges=merges,
    ...     special_tokens=special_tokens,
    ... )
    >>> tokenizer.detokenize(tokenizer.tokenize(inputs)).numpy().decode('utf-8')
    ' afternoon sun'
    """

    def __init__(
        self,
        vocabulary,
        merges,
        special_tokens,
        language_tokens=None,
        **kwargs,
    ):
        vocabulary = _load_dict(vocabulary)

        # Necessary special tokens.
        bos_token = "<|startoftranscript|>"
        eos_token = "<|endoftext|>"

        if language_tokens:
            # Multilingual tokenizer.
            # TODO: The pad token for the multilingual tokenizer is actually
            # "", but it errors out (OOM). After BPE is fixed, we can update
            # this to "". For now, we will use `"<endoftext>"`.
            pad_token = "<|endoftext|>"
            language_tokens = _load_dict(language_tokens)

            # Add language tokens to the vocabulary. This makes detokenization
            # easier for us.
            vocabulary = {
                **vocabulary,
                **language_tokens,
            }
        else:
            # English tokenizer.
            pad_token = "<|endoftext|>"

        no_timestamps_token = "<|notimestamps|>"
        # Task special tokens.
        translate_token = "<|translate|>"
        transcribe_token = "<|transcribe|>"

        special_tokens = _load_dict(special_tokens)
        for token in [
            bos_token,
            eos_token,
            pad_token,
            no_timestamps_token,
            translate_token,
            transcribe_token,
        ]:
            if token not in special_tokens:
                raise ValueError(
                    f"Cannot find token `'{token}'` in the provided "
                    f"`special_tokens`. Please provide `'{token}'` in your "
                    "`special_tokens`."
                )
            # Add special tokens to `vocabulary` for easy detokenization.
            vocabulary[token] = special_tokens[token]

        self.bos_token_id = special_tokens[bos_token]
        self.eos_token_id = special_tokens[eos_token]
        self.pad_token_id = special_tokens[pad_token]
        self.no_timestamps_token_id = special_tokens[no_timestamps_token]
        self.translate_token_id = special_tokens[translate_token]
        self.transcribe_token_id = special_tokens[transcribe_token]

        unsplittable_tokens = list(special_tokens.keys())
        if language_tokens:
            unsplittable_tokens += list(language_tokens.keys())

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            unsplittable_tokens=unsplittable_tokens,
            **kwargs,
        )

        self.special_tokens = special_tokens
        self.language_tokens = language_tokens

    def get_config(self):
        config = super().get_config()

        # In the constructor, we pass the list of special tokens to the
        # `unsplittable_tokens` arg of the superclass' constructor. Hence, we
        # delete it from the config here.
        del config["unsplittable_tokens"]

        config.update(
            {
                "special_tokens": self.special_tokens,
                "language_tokens": self.language_tokens,
            }
        )
        return config
