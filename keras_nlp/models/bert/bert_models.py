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

"""BERT backbone models."""

import os

import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers.position_embedding import PositionEmbedding
from keras_nlp.layers.transformer_encoder import TransformerEncoder
from keras_nlp.models.bert.bert_checkpoints import checkpoints
from keras_nlp.models.bert.bert_checkpoints import compatible_checkpoints
from keras_nlp.models.bert.bert_checkpoints import vocabularies


def bert_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


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
        arch_checkpoints = compatible_checkpoints(bert_variant)
        if weights not in arch_checkpoints:
            raise ValueError(
                "`weights` must be one of "
                f"""{", ".join(arch_checkpoints)}. """
                f"Received: {weights}"
            )
        metadata = checkpoints[weights]
        vocabulary = metadata["vocabulary"]
        vocabulary_size = vocabularies[vocabulary]["vocabulary_size"]

        # TODO(jbischof): consider changing format from `h5` to
        # `tf.train.Checkpoint` once
        # https://github.com/keras-team/keras/issues/16946 is resolved.
        weights = keras.utils.get_file(
            "model.h5",
            metadata["weights_url"],
            cache_subdir=os.path.join("models", weights),
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
    heads, and embedding dimensions. For specific BERT architectures defined in
    the paper, see, for example, `keras_nlp.models.BertBase`.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The size of the transformer encoding and pooler layers.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        dropout: float. Dropout probability for the Transformer encoder.
        max_sequence_length: int. The maximum sequence length that this encoder
            can consume. If None, `max_sequence_length` uses the value from
            sequence length. This determines the variable shape for positional
            embeddings.
        num_segments: int. The number of types that the 'segment_ids' input can
            take.
        name: string, optional. Name of the model.
        trainable: boolean, optional. If the model's variables should be
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
            embeddings_initializer=bert_kernel_initializer(),
            name="token_embedding",
        )
        token_embedding = token_embedding_layer(token_id_input)
        position_embedding = PositionEmbedding(
            initializer=bert_kernel_initializer(),
            sequence_length=max_sequence_length,
            name="position_embedding",
        )(token_embedding)
        segment_embedding = keras.layers.Embedding(
            input_dim=num_segments,
            output_dim=hidden_dim,
            embeddings_initializer=bert_kernel_initializer(),
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
                kernel_initializer=bert_kernel_initializer(),
                name=f"transformer_layer_{i}",
            )(x, padding_mask=padding_mask)

        # Construct the two BERT outputs. The pooled output is a dense layer on
        # top of the [CLS] token.
        sequence_output = x
        pooled_output = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=bert_kernel_initializer(),
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


MODEL_DOCSTRING = """Bert "{type}" architecture.

    This network implements a bi-directional Transformer-based encoder as
    described in ["BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding"](https://arxiv.org/abs/1810.04805). It includes the
    embedding lookups and transformer layers, but not the masked language model
    or classification task networks.

    Args:
        weights: string, optional. Name of pretrained model to load weights.
            Should be one of {names}.
            If None, model is randomly initialized. Either `weights` or
            `vocabulary_size` must be specified, but not both.
        vocabulary_size: Int, optional. The size of the token vocabulary. Either
            `weights` or `vocabulary_size` must be specified, but not both.
        name: string, optional. Name of the model.
        trainable: boolean, optional. If the model's variables should be
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
    model = keras_nlp.models.Bert{type}(weights="bert_base_uncased_en")
    # Call encoder on the inputs.
    output = model(input_data)
    ```
"""


def BertTiny(weights=None, vocabulary_size=None, name=None, trainable=True):
    weights, vocabulary_size = _handle_pretrained_model_arguments(
        "BertTiny", weights, vocabulary_size
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

    return model


def BertSmall(weights=None, vocabulary_size=None, name=None, trainable=True):
    weights, vocabulary_size = _handle_pretrained_model_arguments(
        "BertSmall", weights, vocabulary_size
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
        "BertMedium", weights, vocabulary_size
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
        "BertBase", weights, vocabulary_size
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
        "BertLarge", weights, vocabulary_size
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


def model_class_by_name(classname):
    """Return model class given the class name."""
    return {
        "BertTiny": BertTiny,
        "BertSmall": BertSmall,
        "BertMedium": BertMedium,
        "BertBase": BertBase,
        "BertLarge": BertLarge,
    }[classname]


setattr(
    BertTiny,
    "__doc__",
    MODEL_DOCSTRING.format(
        type="Tiny", names=", ".join(compatible_checkpoints("BertTiny"))
    ),
)

setattr(
    BertSmall,
    "__doc__",
    MODEL_DOCSTRING.format(
        type="Small", names=", ".join(compatible_checkpoints("BertSmall"))
    ),
)

setattr(
    BertMedium,
    "__doc__",
    MODEL_DOCSTRING.format(
        type="Medium", names=", ".join(compatible_checkpoints("BertMedium"))
    ),
)

setattr(
    BertBase,
    "__doc__",
    MODEL_DOCSTRING.format(
        type="Base", names=", ".join(compatible_checkpoints("BertBase"))
    ),
)
setattr(
    BertLarge,
    "__doc__",
    MODEL_DOCSTRING.format(
        type="Large", names=", ".join(compatible_checkpoints("BertLarge"))
    ),
)
