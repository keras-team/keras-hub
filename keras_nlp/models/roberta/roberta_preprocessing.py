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

"""RoBERTa preprocessing layers."""

import copy
import os

import tensorflow as tf
import tensorflow_text as tf_text
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


@keras.utils.register_keras_serializable(package="keras_nlp")
class RobertaPreprocessor(keras.layers.Layer):
    """RoBERTa preprocessing layer which tokenizes and packs inputs.

    This preprocessing layer will do three things:

    - Tokenize any number of inputs using `tokenizer`.
    - Pack the inputs together with the appropriate `"<s>"`, `"</s>"` and
      `"<pad>"` tokens, i.e., adding a single `"<s>"` at the start of the
      entire sequence, `"</s></s>"` at the end of each segment, save the last
      and a `"</s>"` at the end of the entire sequence.
    - Construct a dictionary with keys `"token_ids"` and `"padding_mask"` that
      can be passed directly to a RoBERTa model.

    This layer will accept either a tuple of (possibly batched) inputs, or a
    single input tensor. If a single tensor is passed, it will be packed
    equivalently to a tuple with a single element.

    Args:
        tokenizer: A `keras_nlp.models.RobertaTokenizer` instance.
        sequence_length: The length of the packed inputs.
        truncate: string. The algorithm to truncate a list of batched segments
            to fit within `sequence_length`. The value can be either
            `round_robin` or `waterfall`:
                - `"round_robin"`: Available space is assigned one token at a
                    time in a round-robin fashion to the inputs that still need
                    some, until the limit is reached.
                - `"waterfall"`: The allocation of the budget is done using a
                    "waterfall" algorithm that allocates quota in a
                    left-to-right manner and fills up the buckets until we run
                    out of budget. It supports an arbitrary number of segments.

    Examples:
    ```python
    vocab = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "reful": 3,
        "gent": 4,
        "Ġafter": 5,
        "noon": 6,
        "Ġsun": 7,
        "Ġbright": 8,
        "Ġnight": 9,
        "Ġmoon": 10,
    }
    merges = ["Ġ a", "Ġ m", "Ġ s", "Ġ b", "Ġ n", "r e", "f u", "g e", "n t"]
    merges += ["e r", "n o", "o n", "i g", "h t"]
    merges += ["Ġs u", "Ġa f", "Ġm o", "Ġb r","ge nt", "no on", "re fu", "ig ht"]
    merges += ["Ġn ight", "Ġsu n", "Ġaf t", "Ġmo on", "Ġbr ight", "refu l", "Ġaft er"]

    tokenizer = keras_nlp.models.RobertaTokenizer(
        vocabulary=vocab,
        merges=merges,
    )
    preprocessor = keras_nlp.models.RobertaPreprocessor(
        tokenizer=tokenizer,
        sequence_length=20,
    )

    # Tokenize and pack a single sentence directly.
    preprocessor(" afternoon sun")

    # Tokenize and pack multiples sentences directly.
    preprocessor((" afternoon sun", "refulgent sun"))

    # Map a dataset to preprocess a single sentence at a time.
    features = [" afternoon sun", "refulgent sun"]
    labels = [0, 1]
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = ds.map(
        lambda x, y: (preprocessor(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Map a dataset to preprocess multiple sentences.
    first_sentences = [" afternoon sun", "refulgent sun"]
    second_sentences = [" night moon", " bright moon"]
    labels = [1, 1]
    ds = tf.data.Dataset.from_tensor_slices(
        (
            (first_sentences, second_sentences), labels
        )
    )
    ds = ds.map(
        lambda x, y: (preprocessor(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ```
    """

    def __init__(
        self,
        tokenizer,
        sequence_length=512,
        truncate="round_robin",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._tokenizer = tokenizer

        self.packer = RobertaMultiSegmentPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            truncate=truncate,
            sequence_length=sequence_length,
        )

    @property
    def tokenizer(self):
        """The `keras_nlp.models.RobertaTokenizer` used to tokenize strings."""
        return self._tokenizer

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "tokenizer": keras.layers.serialize(self.tokenizer),
                "sequence_length": self.packer.sequence_length,
                "truncate": self.packer.truncate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if "tokenizer" in config:
            config["tokenizer"] = keras.layers.deserialize(config["tokenizer"])
        return cls(**config)

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        inputs = [self.tokenizer(x) for x in inputs]
        token_ids = self.packer(inputs)
        return {
            "token_ids": token_ids,
            "padding_mask": token_ids != self.tokenizer.pad_token_id,
        }

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    @classmethod
    @format_docstring(names=", ".join(backbone_presets))
    def from_preset(
        cls,
        preset,
        sequence_length=None,
        truncate="round_robin",
        **kwargs,
    ):
        """Instantiate RoBERTa preprocessor from preset architecture.

        Args:
            preset: string. Must be one of {{names}}.
            sequence_length: int, optional. The length of the packed inputs.
                Must be equal to or smaller than the `max_sequence_length` of
                the preset. If left as default, the `max_sequence_length` of
                the preset will be used.
            truncate: string. The algorithm to truncate a list of batched
                segments to fit within `sequence_length`. The value can be
                either `round_robin` or `waterfall`:
                    - `"round_robin"`: Available space is assigned one token at
                        a time in a round-robin fashion to the inputs that still
                        need some, until the limit is reached.
                    - `"waterfall"`: The allocation of the budget is done using
                        a "waterfall" algorithm that allocates quota in a
                        left-to-right manner and fills up the buckets until we
                        run out of budget. It supports an arbitrary number of
                        segments.

        Examples:
        ```python
        # Load preprocessor from preset
        preprocessor = keras_nlp.models.RobertPreprocessor.from_preset(
            "roberta_base",
        )
        preprocessor("The quick brown fox jumped.")

        # Override sequence_length
        preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
            "roberta_base",
            sequence_length=64
        )
        preprocessor("The quick brown fox jumped.")
        ```
        """
        if preset not in cls.presets:
            raise ValueError(
                "`preset` must be one of "
                f"""{", ".join(cls.presets)}. Received: {preset}."""
            )

        tokenizer = RobertaTokenizer.from_preset(preset)

        # Use model's `max_sequence_length` if `sequence_length` unspecified;
        # otherwise check that `sequence_length` not too long.
        metadata = cls.presets[preset]
        max_sequence_length = metadata["config"]["max_sequence_length"]
        if sequence_length is not None:
            if sequence_length > max_sequence_length:
                raise ValueError(
                    f"`sequence_length` cannot be longer than `{preset}` "
                    f"preset's `max_sequence_length` of {max_sequence_length}. "
                    f"Received: {sequence_length}."
                )
        else:
            sequence_length = max_sequence_length

        return cls(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            truncate=truncate,
            **kwargs,
        )


# TODO: This is a temporary, unexported layer until we find a way to make the
# `MultiSegmentPacker` layer more generic.
@keras.utils.register_keras_serializable(package="keras_nlp")
class RobertaMultiSegmentPacker(keras.layers.Layer):
    """Packs multiple sequences into a single fixed width model input.

    This layer packs multiple input sequences into a single fixed width sequence
    containing start and end delimiters, forming a dense input suitable for a
    classification task for RoBERTa.

    Takes as input a list or tuple of token segments. The layer will process
    inputs as follows:
     - Truncate all input segments to fit within `sequence_length` according to
       the `truncate` strategy.
     - Concatenate all input segments, adding a single `start_value` at the
       start of the entire sequence, `[end_value, end_value]` at the end of
       each segment save the last, and a single `end_value` at the end of the
       entire sequence.
     - Pad the resulting sequence to `sequence_length` using `pad_tokens`.

    Input should be either a `tf.RaggedTensor` or a dense `tf.Tensor`, and
    either rank-1 or rank-2.

    Please refer to the arguments of `keras_nlp.layers.MultiSegmentPacker` for
    more details.
    """

    def __init__(
        self,
        sequence_length,
        start_value,
        end_value,
        pad_value=None,
        truncate="round_robin",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        if truncate not in ("round_robin", "waterfall"):
            raise ValueError(
                "Only 'round_robin' and 'waterfall' algorithms are "
                "supported. Received %s" % truncate
            )
        self.truncate = truncate
        self.start_value = start_value
        self.end_value = end_value
        self.pad_value = pad_value

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "start_value": self.start_value,
                "end_value": self.end_value,
                "pad_value": self.pad_value,
                "truncate": self.truncate,
            }
        )
        return config

    def _trim_inputs(self, inputs):
        """Trim inputs to desired length."""
        # Special tokens include the start token at the beginning of the
        # sequence, two `end_value` at the end of every segment save the last,
        # and the `end_value` at the end of the sequence.
        num_special_tokens = 2 * len(inputs)
        if self.truncate == "round_robin":
            return tf_text.RoundRobinTrimmer(
                self.sequence_length - num_special_tokens
            ).trim(inputs)
        elif self.truncate == "waterfall":
            return tf_text.WaterfallTrimmer(
                self.sequence_length - num_special_tokens
            ).trim(inputs)
        else:
            raise ValueError("Unsupported truncate: %s" % self.truncate)

    def _combine_inputs(self, segments):
        """Combine inputs with start and end values added."""
        dtype = segments[0].dtype
        batch_size = segments[0].nrows()

        start_value = tf.convert_to_tensor(self.start_value, dtype=dtype)
        end_value = tf.convert_to_tensor(self.end_value, dtype=dtype)

        start_column = tf.fill((batch_size, 1), start_value)
        end_column = tf.fill((batch_size, 1), end_value)

        segments_to_combine = []
        for i, seg in enumerate(segments):
            segments_to_combine.append(start_column if i == 0 else end_column)
            segments_to_combine.append(seg)
            segments_to_combine.append(end_column)

        token_ids = tf.concat(segments_to_combine, 1)
        return token_ids

    def call(self, inputs):
        def to_ragged(x):
            return tf.RaggedTensor.from_tensor(x[tf.newaxis, :])

        # If rank 1, add a batch dim.
        rank_1 = inputs[0].shape.rank == 1
        if rank_1:
            inputs = [to_ragged(x) for x in inputs]

        segments = self._trim_inputs(inputs)
        token_ids = self._combine_inputs(segments)
        # Pad to dense tensor output.
        shape = tf.cast([-1, self.sequence_length], tf.int64)
        token_ids = token_ids.to_tensor(
            shape=shape, default_value=self.pad_value
        )
        # Remove the batch dim if added.
        if rank_1:
            token_ids = tf.squeeze(token_ids, 0)

        return token_ids
