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
"""XLM-RoBERTa model configurable class and preconfigured versions."""

import os
from collections import defaultdict

import tensorflow as tf
from tensorflow import keras

from keras_nlp.models import roberta
from keras_nlp.models.roberta import RobertaMultiSegmentPacker
from keras_nlp.tokenizers import SentencePieceTokenizer

checkpoints = {
    "xlm_roberta_base": {
        "model": "XLMRobertaBase",
        "vocabulary": "common_crawl",
        "description": (
            "Base size of XLM-RoBERTa with 277M parameters. Trained on "
            "the CommonCrawl dataset (100 languages)."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/xlm_roberta_base/model.h5",
        "weights_hash": "2eb6fcda5a42f0a88056213ba3d93906",
    },
    "xlm_roberta_large": {
        "model": "XLMRobertaLarge",
        "vocabulary": "common_crawl",
        "description": (
            "Large size of XLM-RoBERTa with 558M parameters. Trained on "
            "the CommonCrawl dataset (100 languages)."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/xlm_roberta_base/model.h5",
        "weights_hash": "276211827174b71751f2ce3a89da503a",
    },
}

# Index checkpoints by arch compatibility.
checkpoints_per_arch = defaultdict(set)
for arch, metadata in checkpoints.items():
    checkpoints_per_arch[metadata["model"]].add(arch)


def compatible_checkpoints(arch):
    """Returns a list of compatible checkpoints per arch"""
    return checkpoints_per_arch[arch]


vocabularies = {
    "common_crawl": {
        "description": (
            "The BPE SentencePiece vocabulary for XLM-RoBERTa models trained on "
            "the CommonCrawl dataset."
        ),
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/xlm_roberta_base/vocab.spm",
        "vocabulary_hash": "bf25eb5120ad92ef5c7d8596b5dc4046",
        "vocabulary_size": 250002,
    }
}


def _handle_pretrained_tokenizer_arguments(vocabulary):
    """Look up pretrained defaults for tokenizer arguments.

    This helper will validate the `vocabulary` argument, and
    fully resolve it in the case we are loading pretrained weights.
    """
    if isinstance(vocabulary, str) and vocabulary in vocabularies:
        metadata = vocabularies[vocabulary]
        vocabulary = keras.utils.get_file(
            "vocab.spm",
            metadata["vocabulary_url"],
            cache_subdir=os.path.join("models", "xlm_roberta", vocabulary),
            file_hash=metadata["vocabulary_hash"],
        )

    return vocabulary


def _handle_pretrained_model_arguments(
    xlm_roberta_variant, weights, vocabulary_size
):
    """Look up pretrained defaults for model arguments.

    This helper will validate the `weights` and `vocabulary_size` arguments, and
    fully resolve them in the case we are loading pretrained weights.
    """
    if (vocabulary_size is None and weights is None) or (
        vocabulary_size and weights
    ):
        raise ValueError(
            "One of `vocabulary_size` or `weights` must be specified "
            "(but not both). "
            f"Received: weights={weights}, "
            f"vocabulary_size={vocabulary_size}"
        )

    if weights:
        arch_checkpoints = compatible_checkpoints(xlm_roberta_variant)
        if weights not in arch_checkpoints:
            raise ValueError(
                "`weights` must be one of "
                f"""{", ".join(arch_checkpoints)}. """
                f"Received: {weights}"
            )
        metadata = checkpoints[weights]
        vocabulary = metadata["vocabulary"]
        vocabulary_size = vocabularies[vocabulary]["vocabulary_size"]

        weights = keras.utils.get_file(
            "model.h5",
            metadata["weights_url"],
            cache_subdir=os.path.join("models", weights),
            file_hash=metadata["weights_hash"],
        )

    return weights, vocabulary_size


class XLMRobertaCustom(roberta.RobertaCustom):
    """XLM-RoBERTa encoder with a customizable set of hyperparameters.

    This network implements a bi-directional Transformer-based encoder as
    described in
    ["Unsupervised Cross-lingual Representation Learning at Scale"](https://arxiv.org/abs/1911.02116).
    It includes the embedding lookups and transformer layers, but does not
    include the masked language modeling head used during pretraining.

    This class gives a fully configurable XLM-R model with any number of
    layers, heads, and embedding dimensions. The graph of XLM-R is
    exactly the same as RoBERTa's. For specific XLM-R architectures
    defined in the paper, see, for example, `keras_nlp.models.XLMRobertaBase`.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The size of the transformer encoding layer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        dropout: float. Dropout probability for the Transformer encoder.
        max_sequence_length: int. The maximum sequence length this encoder can
            consume. The sequence length of the input must be less than
            `max_sequence_length`.
        name: string, optional. Name of the model.
        trainable: boolean, optional. If the model's variables should be
            trainable.

    Example usage:
    ```python
    # Randomly initialized XLM-R model
    model = keras_nlp.models.XLMRobertaCustom(
        vocabulary_size=50265,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12
    )

    # Call encoder on the inputs.
    input_data = {
        "token_ids": tf.random.uniform(
            shape=(1, 12), dtype=tf.int64, maxval=50265),
        "padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)),
    }
    output = model(input_data)
    ```
    """

    pass


PREPROCESSOR_DOCSTRING = """XLM-RoBERTa preprocessor with pretrained vocabularies.

This preprocessing layer will do three things:

 - Tokenize any number of inputs using a
   `keras_nlp.tokenizers.SentencePieceTokenizer`.
 - Pack the inputs together with the appropriate `"<s>"`, `"</s>"` and `"<pad>"`
   tokens, i.e., adding a single `"<s>"` at the start of the entire sequence,
   `"</s></s>"` at the end of each segment, save the last and a `"</s>"` at the
   end of the entire sequence.
 - Construct a dictionary of with keys `"token_ids"`, `"padding_mask"`, that can
   be passed directly to a XLM-RoBERTa model.

Note that the original fairseq implementation modifies the indices of the
SentencePiece tokenizer output. To preserve compatibility, we make the same
changes, i.e., `"<s>"`, `"<pad>"`, `"</s>"` and `"<unk>"` are mapped to
1, 2, 3, 4, respectively, and non-special tokens' indices are shifted right by
one. Keep this in mind if generating your own vocabulary for tokenization.

This layer will accept either a tuple of (possibly batched) inputs, or a single
input tensor. If a single tensor is passed, it will be packed equivalently to
a tuple with a single element.

The SentencePiece tokenizer can be accessed via the `tokenizer` property on this
layer, and can be used directly for custom packing on inputs.

Args:
    proto: Either a `string` path to a SentencePiece proto file, a
        `bytes` object with a serialized SentencePiece proto, or the name of a
        pretrained vocabulary. For a pretrained vocabulary, `proto` should be
        one of {names}.
    sequence_length: The length of the packed inputs. Only used if
        `pack_inputs` is True.
    truncate: The algorithm to truncate a list of batched segments to fit
        within `sequence_length`. Only used if
        `pack_inputs` is True. The value can be either `round_robin` or
        `waterfall`:
            - `"round_robin"`: Available space is assigned one token at a
                time in a round-robin fashion to the inputs that still need
                some, until the limit is reached.
            - `"waterfall"`: The allocation of the budget is done using a
                "waterfall" algorithm that allocates quota in a
                left-to-right manner and fills up the buckets until we run
                out of budget. It supports an arbitrary number of segments.

Examples:
```python
preprocessor = keras_nlp.models.XLMRobertaPreprocessor(proto="model.spm")

# Tokenize and pack a single sentence directly.
preprocessor("The quick brown fox jumped.")

# Tokenize and pack a multiple sentence directly.
preprocessor(("The quick brown fox jumped.", "Call me Ishmael."))

# Map a dataset to preprocess a single sentence.
features = ["The quick brown fox jumped.", "I forgot my homework."]
labels = [0, 1]
ds = tf.data.Dataset.from_tensor_slices((features, labels))
ds = ds.map(
    lambda x, y: (preprocessor(x), y),
    num_parallel_calls=tf.data.AUTOTUNE,
)

# Map a dataset to preprocess a multiple sentences.
first_sentences = ["The quick brown fox jumped.", "Call me Ishmael."]
second_sentences = ["The fox tripped.", "Oh look, a whale."]
labels = [1, 1]
ds = tf.data.Dataset.from_tensor_slices(((first_sentences, second_sentences), labels))
ds = ds.map(
    lambda x, y: (preprocessor(x), y),
    num_parallel_calls=tf.data.AUTOTUNE,
)
```
"""


class XLMRobertaPreprocessor(keras.layers.Layer):
    def __init__(
        self,
        proto,
        sequence_length=512,
        truncate="round_robin",
        **kwargs,
    ):
        super().__init__(**kwargs)

        proto = _handle_pretrained_tokenizer_arguments(proto)
        self.tokenizer = SentencePieceTokenizer(proto=proto)

        # Check for necessary special tokens.
        start_token_id = 0
        pad_token_id = 1
        end_token_id = 2

        self.packer = RobertaMultiSegmentPacker(
            start_value=start_token_id,
            end_value=end_token_id,
            pad_value=pad_token_id,
            truncate=truncate,
            sequence_length=sequence_length,
        )

        self.pad_token_id = pad_token_id

    def vocabulary_size(self):
        """Returns the vocabulary size of the tokenizer."""
        return self.tokenizer.vocabulary_size()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "proto": self.tokenizer.proto,
                "sequence_length": self.packer.sequence_length,
                "truncate": self.packer.truncate,
            }
        )
        return config

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        def _tokenize(x):
            tokenized = self.tokenizer(x)
            dtype = tokenized.dtype

            # In the official SPM proto file, `[unk]`'s ID is 0. Replace that
            # with 2. This will be changed to 3 (`[unk]`'s ID is 3 in the
            # official implementation) after adding by 1.
            tokenized = tf.where(
                tf.equal(tokenized, tf.constant(0, dtype=dtype)),
                tf.constant(2, dtype=dtype),
                tokenized,
            )
            # Shift the tokens IDs by one.
            tokenized = tf.add(tokenized, tf.constant(1, dtype=dtype))
            return tokenized

        inputs = [_tokenize(x) for x in inputs]
        token_ids = self.packer(inputs)
        return {
            "token_ids": token_ids,
            "padding_mask": token_ids != self.pad_token_id,
        }


setattr(
    XLMRobertaPreprocessor,
    "__doc__",
    PREPROCESSOR_DOCSTRING.format(names=", ".join(vocabularies)),
)


MODEL_DOCSTRING = """XLM-RoBERTa "{type}" architecture.

    This network implements a bi-directional Transformer-based encoder as
    described in
    ["Unsupervised Cross-lingual Representation Learning at Scale"](https://arxiv.org/abs/1911.02116).
    It includes the embedding lookups and transformer layers, but does not
    include the masked language modeling head used during pretraining.

    Args:
        vocabulary_size: int, optional. The size of the token vocabulary.
        name: string, optional. Name of the model.
        trainable: boolean, optional. If the model's variables should be
            trainable.

    Example usage:
    ```python
    # Randomly initialized XLMRoberta{type} encoder
    model = keras_nlp.models.XLMRoberta{type}(weights=None, vocabulary_size=10000)

    # Call encoder on the inputs.
    input_data = {
        "token_ids": tf.random.uniform(
            shape=(1, 512), dtype=tf.int64, maxval=model.vocabulary_size),
        "padding_mask": tf.ones((1, 512)),
    }
    output = model(input_data)
    ```
"""


def XLMRobertaBase(
    weights="xlm_roberta_base", vocabulary_size=None, name=None, trainable=True
):
    weights, vocabulary_size = _handle_pretrained_model_arguments(
        "XLMRobertaBase", weights, vocabulary_size
    )

    model = XLMRobertaCustom(
        vocabulary_size=vocabulary_size,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        dropout=0.1,
        max_sequence_length=512,
        name=name,
        trainable=trainable,
    )

    if weights:
        model.load_weights(weights)

    return model


def XLMRobertaLarge(
    weights="xlm_roberta_large", vocabulary_size=None, name=None, trainable=True
):
    weights, vocabulary_size = _handle_pretrained_model_arguments(
        "XLMRobertaLarge", weights, vocabulary_size
    )

    model = XLMRobertaCustom(
        vocabulary_size=vocabulary_size,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        intermediate_dim=4096,
        dropout=0.1,
        max_sequence_length=512,
        name=name,
        trainable=trainable,
    )

    if weights:
        model.load_weights(weights)

    return model
