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

"""BERT model configurable class, preconfigured versions, and task heads."""

import os

import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers.multi_segment_packer import MultiSegmentPacker
from keras_nlp.layers.position_embedding import PositionEmbedding
from keras_nlp.layers.transformer_encoder import TransformerEncoder
from keras_nlp.tokenizers.word_piece_tokenizer import WordPieceTokenizer


def _bert_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


# Metadata for loading pretrained model weights.
checkpoints = {
    "bert_tiny": {
        "uncased_en": {
            "description": (
                "Tiny size of BERT where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_tiny_uncased_en/model.h5",
            "weights_hash": "c2b29fcbf8f814a0812e4ab89ef5c068",
        }
    },
    "bert_small": {
        "uncased_en": {
            "description": (
                "Small size of BERT where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_small_uncased_en/model.h5",
            "weights_hash": "08632c9479b034f342ba2c2b7afba5f7",
        }
    },
    "bert_medium": {
        "uncased_en": {
            "description": (
                "Medium size of BERT where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_medium_uncased_en/model.h5",
            "weights_hash": "bb990e1184ec6b6185450c73833cd661",
        }
    },
    "bert_base": {
        "uncased_en": {
            "description": (
                "Base size of Bert where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_uncased_en/model.h5",
            "weights_hash": "9b2b2139f221988759ac9cdd17050b31",
        },
        "cased_en": {
            "description": (
                "Base size of Bert where case is maintained. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_cased_en/model.h5",
            "weights_hash": "f94a6cb012e18f4fb8ec92abb91864e9",
        },
        "zh": {
            "description": ("Base size of BERT. Trained on Chinese Wikipedia."),
            "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_zh/model.h5",
            "weights_hash": "79afa421e386076e62ab42dad555ab0c",
        },
        "multi_cased": {
            "description": (
                "Base size of BERT. Trained on Wikipedias of 104 languages."
            ),
            "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_multi_cased/model.h5",
            "weights_hash": "b0631cec0a1f2513c6cfd75ba29c33aa",
        },
    },
    "bert_large": {
        "uncased_en": {
            "description": (
                "Large size of BERT where all input is lowercased. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_large_uncased_en/model.h5",
            "weights_hash": "cc5cacc9565ef400ee4376105f40ddae",
        },
        "cased_en": {
            "description": (
                "Large size of BERT where case is maintained. "
                "Trained on English Wikipedia + BooksCorpus."
            ),
            "weights_url": "https://storage.googleapis.com/keras-nlp/models/bert_large_cased_en/model.h5",
            "weights_hash": "8b8ab82290bbf4f8db87d4f100648890",
        },
    },
}


# Metadata for loading pretrained tokenizer vocabularies.
# We need the vocabulary_size hardcoded so we can instantiate a Bert network
# with the right embedding size without downloading the matching vocabulary.
# TODO(mattdangerw): Update our bucket structure so the vocabularies are
# stored in an independent way, rather than reading from the base model.
vocabularies = {
    "uncased_en": {
        "description": (
            "The vocabulary for BERT models trained on "
            "English Wikipedia + BooksCorpus where case is discarded."
        ),
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_uncased_en/vocab.txt",
        "vocabulary_hash": "64800d5d8528ce344256daf115d4965e",
        "vocabulary_size": 30522,
        "lowercase": True,
    },
    "cased_en": {
        "description": (
            "The vocabulary for BERT models trained on "
            "English Wikipedia + BooksCorpus where case is maintained."
        ),
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_cased_en/vocab.txt",
        "vocabulary_hash": "bb6ca9b42e790e5cd986bbb16444d0e0",
        "vocabulary_size": 28996,
        "lowercase": False,
    },
    "zh": {
        "description": (
            "The vocabulary for BERT models trained on Chinese Wikipedia."
        ),
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_zh/vocab.txt",
        "vocabulary_hash": "3b5b76c4aef48ecf8cb3abaafe960f09",
        "vocabulary_size": 21128,
        "lowercase": False,
    },
    "multi_cased": {
        "description": (
            "The vocabulary for BERT models trained on trained on Wikipedias "
            "of 104 languages."
        ),
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/bert_base_multi_cased/vocab.txt",
        "vocabulary_hash": "d9d865138d17f1958502ed060ecfeeb6",
        "vocabulary_size": 119547,
        "lowercase": False,
    },
}


def _handle_pretrained_tokenizer_arguments(vocabulary, lowercase):
    """Look up pretrained defaults for tokenizer arguments.

    This helper will validate the `vocabulary` and `lowercase` arguments, and
    fully resolve them in the case we are loading pretrained weights.
    """

    if isinstance(vocabulary, str) and vocabulary in vocabularies:
        metadata = vocabularies[vocabulary]
        vocabulary = keras.utils.get_file(
            "vocab.txt",
            metadata["vocabulary_url"],
            cache_subdir=os.path.join("models", "bert", vocabulary),
            file_hash=metadata["vocabulary_hash"],
        )
        lowercase = metadata["lowercase"]

    return vocabulary, lowercase


def _handle_pretrained_model_arguments(bert_variant, weights, vocabulary_size):
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
        if weights not in checkpoints[bert_variant]:
            raise ValueError(
                "`weights` must be one of "
                f"""{", ".join(checkpoints[bert_variant])}. """
                f"Received: {weights}"
            )
        metadata = checkpoints[bert_variant][weights]
        vocabulary_size = vocabularies[weights]["vocabulary_size"]

        # TODO(jbischof): consider changing format from `h5` to
        # `tf.train.Checkpoint` once
        # https://github.com/keras-team/keras/issues/16946 is resolved.
        weights = keras.utils.get_file(
            "model.h5",
            metadata["weights_url"],
            cache_subdir=os.path.join("models", "bert", weights, bert_variant),
            file_hash=metadata["weights_hash"],
        )

    return weights, vocabulary_size


class BertCustom(keras.Model):
    """BERT encoder network with custom hyperparmeters.

    This network implements a bi-directional Transformer-based encoder as
    described in ["BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding"](https://arxiv.org/abs/1810.04805). It includes the
    embedding lookups and transformer layers, but not the masked language model
    or classification task networks.

    This class gives a fully customizable BERT model with any number of layers,
    heads, and embedding dimensions. For specific specific BERT architectures
    defined in the paper, see for example `keras_nlp.models.BertBase`.

    Args:
        vocabulary_size: Int. The size of the token vocabulary.
        num_layers: Int. The number of transformer layers.
        num_heads: Int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: Int. The size of the transformer encoding and pooler layers.
        intermediate_dim: Int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        dropout: Float. Dropout probability for the Transformer encoder.
        max_sequence_length: Int. The maximum sequence length that this encoder
            can consume. If None, `max_sequence_length` uses the value from
            sequence length. This determines the variable shape for positional
            embeddings.
        num_segments: Int. The number of types that the 'segment_ids' input can
            take.
        name: String, optional. Name of the model.
        trainable: Boolean, optional. If the model's variables should be
            trainable.

    Examples:
    ```python
    # Randomly initialized BERT encoder
    model = keras_nlp.models.BertCustom(
        vocabulary_size=30522,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12,
        name="encoder",
    )

    # Call encoder on the inputs
    input_data = {
        "token_ids": tf.random.uniform(
            shape=(1, 12), dtype=tf.int64, maxval=model.vocabulary_size
        ),
        "segment_ids": tf.constant(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
        "padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
    }
    output = model(input_data)
    ```
    """

    # TODO(jbischof): consider changing `intermediate_dim` and `hidden_dim` to
    # less confusing name here and in TransformerEncoder (`feed_forward_dim`?)

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout=0.1,
        max_sequence_length=512,
        num_segments=2,
        name=None,
        trainable=True,
    ):

        # Index of classification token in the vocabulary
        cls_token_index = 0
        # Inputs
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        segment_id_input = keras.Input(
            shape=(None,), dtype="int32", name="segment_ids"
        )
        padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # Embed tokens, positions, and segment ids.
        token_embedding_layer = keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=_bert_kernel_initializer(),
            name="token_embedding",
        )
        token_embedding = token_embedding_layer(token_id_input)
        position_embedding = PositionEmbedding(
            initializer=_bert_kernel_initializer(),
            sequence_length=max_sequence_length,
            name="position_embedding",
        )(token_embedding)
        segment_embedding = keras.layers.Embedding(
            input_dim=num_segments,
            output_dim=hidden_dim,
            embeddings_initializer=_bert_kernel_initializer(),
            name="segment_embedding",
        )(segment_id_input)

        # Sum, normailze and apply dropout to embeddings.
        x = keras.layers.Add()(
            (token_embedding, position_embedding, segment_embedding)
        )
        x = keras.layers.LayerNormalization(
            name="embeddings_layer_norm",
            axis=-1,
            epsilon=1e-12,
            dtype=tf.float32,
        )(x)
        x = keras.layers.Dropout(
            dropout,
            name="embeddings_dropout",
        )(x)

        # Apply successive transformer encoder blocks.
        for i in range(num_layers):
            x = TransformerEncoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                activation=lambda x: keras.activations.gelu(
                    x, approximate=True
                ),
                dropout=dropout,
                kernel_initializer=_bert_kernel_initializer(),
                name=f"transformer_layer_{i}",
            )(x, padding_mask=padding_mask)

        # Construct the two BERT outputs. The pooled output is a dense layer on
        # top of the [CLS] token.
        sequence_output = x
        pooled_output = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=_bert_kernel_initializer(),
            activation="tanh",
            name="pooled_dense",
        )(x[:, cls_token_index, :])

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "segment_ids": segment_id_input,
                "padding_mask": padding_mask,
            },
            outputs={
                "sequence_output": sequence_output,
                "pooled_output": pooled_output,
            },
            name=name,
            trainable=trainable,
        )
        # All references to `self` below this line
        self.token_embedding = token_embedding_layer
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_sequence_length = max_sequence_length
        self.num_segments = num_segments
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.cls_token_index = cls_token_index

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "max_sequence_length": self.max_sequence_length,
                "num_segments": self.num_segments,
                "dropout": self.dropout,
                "cls_token_index": self.cls_token_index,
            }
        )
        return config


PREPROCESSOR_DOCSTRING = """BERT preprocessor with pretrained vocabularies.

This preprocessing layer will do three things:

 - Tokenize any number of inputs using a
   `keras_nlp.tokenizers.WordPieceTokenizer`.
 - Pack the inputs together using a `keras_nlp.layers.MultiSegmentPacker`.
   with the appropriate `"[CLS]"`, `"[SEP]"` and `"[PAD]"` tokens.
 - Construct a dictionary of with keys `"token_ids"`, `"segment_ids"`,
   `"padding_mask"`, that can be passed directly to a BERT model.

This layer will accept either a tuple of (possibly batched) inputs, or a single
input tensor. If a single tensor is passed, it will be packed equivalently to
a tuple with a single element.

The WordPiece tokenizer can be accessed via the `tokenizer` property on this
layer, and can be used directly for custom packing on inputs.

Args:
    vocabulary: One of a list of vocabulary terms, a vocabulary filename, or
        the name of the pretrained vocabulary. For a pretrained vocabulary,
        `vocabulary` should be one of {names}, and should match the `weights`
        parameter of any pretrained BERT model.
    lowercase: If `True`, input will be lowercase before tokenization. If
        `vocabulary` is set to a pretrained vocabulary, this parameter will
        be inferred.
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
preprocessor = keras_nlp.models.BertPreprocessor(vocabulary="uncased_en")

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


class BertPreprocessor(keras.layers.Layer):
    def __init__(
        self,
        vocabulary="uncased_en",
        lowercase=False,
        sequence_length=512,
        truncate="round_robin",
        **kwargs,
    ):
        super().__init__(**kwargs)

        vocabulary, lowercase = _handle_pretrained_tokenizer_arguments(
            vocabulary, lowercase
        )

        self.tokenizer = WordPieceTokenizer(
            vocabulary=vocabulary,
            lowercase=lowercase,
        )

        # Check for necessary special tokens.
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        pad_token = "[PAD]"
        for token in [cls_token, pad_token, sep_token]:
            if token not in self.tokenizer.get_vocabulary():
                raise ValueError(
                    f"Cannot find token `'{token}'` in the provided "
                    f"`vocabulary`. Please provide `'{token}'` in your "
                    "`vocabulary` or use a pretrained `vocabulary` name."
                )

        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.token_to_id(cls_token),
            end_value=self.tokenizer.token_to_id(sep_token),
            pad_value=self.tokenizer.token_to_id(pad_token),
            truncate=truncate,
            sequence_length=sequence_length,
        )

    def vocabulary_size(self):
        """Returns the vocabulary size of the tokenizer."""
        return self.tokenizer.vocabulary_size()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary": self.tokenizer.vocabulary,
                "lowercase": self.tokenizer.lowercase,
                "sequence_length": self.packer.sequence_length,
                "trucator": self.packer.trucator,
            }
        )
        return config

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        inputs = [self.tokenizer(x) for x in inputs]
        token_ids, segment_ids = self.packer(inputs)
        return {
            "token_ids": token_ids,
            "segment_ids": segment_ids,
            "padding_mask": token_ids != 0,
        }


setattr(
    BertPreprocessor,
    "__doc__",
    PREPROCESSOR_DOCSTRING.format(names=", ".join(vocabularies)),
)


class BertClassifier(keras.Model):
    """BERT encoder model with a classification head.

    Args:
        base_model: A `keras_nlp.models.BertCustom` to encode inputs.
        num_classes: Int. Number of classes to predict.
        name: String, optional. Name of the model.
        trainable: Boolean, optional. If the model's variables should be
            trainable.

    Examples:
    ```python
    # Randomly initialized BERT encoder
    model = keras_nlp.models.BertCustom(
        vocabulary_size=30522,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12
    )

    # Call classifier on the inputs.
    input_data = {
        "token_ids": tf.random.uniform(
            shape=(1, 12), dtype=tf.int64, maxval=model.vocabulary_size
        ),
        "segment_ids": tf.constant(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
        "padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
    }
    classifier = bert.BertClassifier(model, 4, name="classifier")
    logits = classifier(input_data)
    ```
    """

    def __init__(
        self,
        base_model,
        num_classes,
        name=None,
        trainable=True,
    ):
        inputs = base_model.input
        pooled = base_model(inputs)["pooled_output"]
        outputs = keras.layers.Dense(
            num_classes,
            kernel_initializer=_bert_kernel_initializer(),
            name="logits",
        )(pooled)
        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs=inputs, outputs=outputs, name=name, trainable=trainable
        )
        # All references to `self` below this line
        self.base_model = base_model
        self.num_classes = num_classes


MODEL_DOCSTRING = """Bert "{type}" architecture.

    This network implements a bi-directional Transformer-based encoder as
    described in ["BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding"](https://arxiv.org/abs/1810.04805). It includes the
    embedding lookups and transformer layers, but not the masked language model
    or classification task networks.

    Args:
        weights: String, optional. Name of pretrained model to load weights.
            Should be one of {names}.
            If None, model is randomly initialized. Either `weights` or
            `vocabulary_size` must be specified, but not both.
        vocabulary_size: Int, optional. The size of the token vocabulary. Either
            `weights` or `vocabularly_size` must be specified, but not both.
        name: String, optional. Name of the model.
        trainable: Boolean, optional. If the model's variables should be
            trainable.

    Examples:
    ```python
    # Randomly initialized Bert{type} encoder
    model = keras_nlp.models.Bert{type}(vocabulary_size=10000)

    # Call encoder on the inputs.
    input_data = {{
        "token_ids": tf.random.uniform(
            shape=(1, 512), dtype=tf.int64, maxval=model.vocabulary_size
        ),
        "segment_ids": tf.constant([0] * 200 + [1] * 312, shape=(1, 512)),
        "padding_mask": tf.constant([1] * 512, shape=(1, 512)),
    }}
    output = model(input_data)

    # Load a pretrained model
    model = keras_nlp.models.Bert{type}(weights="uncased_en")
    # Call encoder on the inputs.
    output = model(input_data)
    ```
"""


def BertTiny(weights=None, vocabulary_size=None, name=None, trainable=True):
    weights, vocabulary_size = _handle_pretrained_model_arguments(
        "bert_tiny", weights, vocabulary_size
    )

    model = BertCustom(
        vocabulary_size=vocabulary_size,
        num_layers=2,
        num_heads=2,
        hidden_dim=128,
        intermediate_dim=512,
        dropout=0.1,
        max_sequence_length=512,
        name=name,
        trainable=trainable,
    )

    if weights:
        model.load_weights(weights)

    # TODO(jbischof): attach the tokenizer or create separate tokenizer class.
    # This comment applies to other variants as well.
    return model


def BertSmall(weights=None, vocabulary_size=None, name=None, trainable=True):
    weights, vocabulary_size = _handle_pretrained_model_arguments(
        "bert_small", weights, vocabulary_size
    )

    model = BertCustom(
        vocabulary_size=vocabulary_size,
        num_layers=4,
        num_heads=8,
        hidden_dim=512,
        intermediate_dim=2048,
        dropout=0.1,
        max_sequence_length=512,
        name=name,
        trainable=trainable,
    )

    if weights:
        model.load_weights(weights)

    return model


def BertMedium(weights=None, vocabulary_size=None, name=None, trainable=True):
    weights, vocabulary_size = _handle_pretrained_model_arguments(
        "bert_medium", weights, vocabulary_size
    )

    model = BertCustom(
        vocabulary_size=vocabulary_size,
        num_layers=8,
        num_heads=8,
        hidden_dim=512,
        intermediate_dim=2048,
        dropout=0.1,
        max_sequence_length=512,
        name=name,
        trainable=trainable,
    )

    if weights:
        model.load_weights(weights)

    return model


def BertBase(weights=None, vocabulary_size=None, name=None, trainable=True):
    weights, vocabulary_size = _handle_pretrained_model_arguments(
        "bert_base", weights, vocabulary_size
    )

    model = BertCustom(
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

    if weights is not None:
        model.load_weights(weights)

    return model


def BertLarge(weights=None, vocabulary_size=None, name=None, trainable=True):
    weights, vocabulary_size = _handle_pretrained_model_arguments(
        "bert_large", weights, vocabulary_size
    )

    model = BertCustom(
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

    if weights is not None:
        model.load_weights(weights)

    return model


setattr(
    BertTiny,
    "__doc__",
    MODEL_DOCSTRING.format(
        type="Tiny", names=", ".join(checkpoints["bert_tiny"])
    ),
)

setattr(
    BertSmall,
    "__doc__",
    MODEL_DOCSTRING.format(
        type="Small", names=", ".join(checkpoints["bert_small"])
    ),
)

setattr(
    BertMedium,
    "__doc__",
    MODEL_DOCSTRING.format(
        type="Medium", names=", ".join(checkpoints["bert_medium"])
    ),
)

setattr(
    BertBase,
    "__doc__",
    MODEL_DOCSTRING.format(
        type="Base", names=", ".join(checkpoints["bert_base"])
    ),
)
setattr(
    BertLarge,
    "__doc__",
    MODEL_DOCSTRING.format(
        type="Large", names=", ".join(checkpoints["bert_large"])
    ),
)
