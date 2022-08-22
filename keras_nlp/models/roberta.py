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

"""RoBerta model configurable class, preconfigured versions, and task heads."""

import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers import TokenAndPositionEmbedding
from keras_nlp.layers import TransformerEncoder


def _roberta_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


class Roberta(keras.Model):
    """Robustly optimized Bert pretraining approach.

    This network implements a bi-directional Transformer-based encoder as
    described in ["RoBERTa: A Robustly Optimized BERT Pretraining Approach"](https://arxiv.org/abs/1907.11692).
    It includes the embedding lookups and transformer layers, but not include
    the masked language model network.

    This class gives a fully configurable Roberta model with any number of
    layers, heads, and embedding dimensions. For specific roberta architectures
    defined in the paper, see for example `keras_nlp.models.RobertaBase`.

    Args:
        vocabulary_size: Int. The size of the token vocabulary.
        num_layers: Int. The number of transformer layers.
        num_heads: Int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: Int. The size of the transformer encoding layer.
        intermediate_dim: Int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        dropout: Float. Dropout probability for the Transformer encoder.
        max_sequence_length: Int. The maximum sequence length that this encoder
            can consume. If None, `max_sequence_length` uses the value from
            sequence length. This determines the variable shape for positional
            embeddings.
        bos_token_index: Int. Index of <s> token in the vocabulary. Equivalent
            to [CLS] in BERT.
        name: String, optional. Name of the model.
        trainable: Boolean, optional. If the model's variables should be
            trainable.

    Example usage:
    ```python
    # Randomly initialized Roberta encoder
    encoder = keras_nlp.models.Roberta(
        vocabulary_size=50265,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12
    )

    # Call encoder on the inputs.
    input_data = {
        "input_ids": tf.random.uniform(
            shape=(1, 12), dtype=tf.int64, maxval=30522),
        "input_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)),
    }
    output = encoder(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout=0.1,
        max_sequence_length=512,
        bos_token_index=0,
        name=None,
        trainable=True,
    ):

        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="input_ids"
        )
        input_mask = keras.Input(
            shape=(None,), dtype="int32", name="input_mask"
        )

        # Embed tokens and positions.
        token_and_position_embedding_layer = TokenAndPositionEmbedding(
            vocabulary_size=vocabulary_size,
            sequence_length=max_sequence_length,
            embedding_dim=hidden_dim,
            embeddings_initializer=_roberta_kernel_initializer(),
            name="token_and_position_embeddings",
        )
        embedding = token_and_position_embedding_layer(token_id_input)

        # Sum, normailze and apply dropout to embeddings.
        x = keras.layers.LayerNormalization(
            name="embeddings_layer_norm",
            axis=-1,
            epsilon=1e-5,  # Original paper uses this epsilon value
            dtype=tf.float32,
        )(embedding)
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
                kernel_initializer=_roberta_kernel_initializer(),
                name=f"""transformer_layer_{i}""",
            )(x, padding_mask=input_mask)

        # Construct the two RoBerta outputs.
        output = x

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs={
                "input_ids": token_id_input,
                # "segment_ids": segment_id_input,
                "input_mask": input_mask,
            },
            outputs=output,
            name=name,
            trainable=trainable,
        )
        # All references to `self` below this line
        self.token_embedding = (
            token_and_position_embedding_layer.token_embedding
        )
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_sequence_length = max_sequence_length
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        # BOS token '<s>' is equivalent to '[CLS]' from BERT
        self.bos_token_index = bos_token_index

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
                "dropout": self.dropout,
                "bos_token_index": self.bos_token_index,
            }
        )
        return config


# TODO: add RobertaMLM, different from BERT.


class RobertaClassifier(keras.Model):
    """Roberta encoder model with a classification head.

    Args:
        base_model: A `keras_nlp.models.Roberta` to encode inputs.
        num_classes: Int. Number of classes to predict.
        hidden_dim: Int. The size of the pooler layer.
        bos_token_index: Int. Index of <s> token in the vocabulary. Equivalent
            to [CLS] in BERT.
        name: String, optional. Name of the model.
        trainable: Boolean, optional. If the model's variables should be
            trainable.

    Example usage:
    ```
    python
    # Randomly initialized Roberta encoder
    encoder = keras_nlp.models.Roberta(
        vocabulary_size=50265,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12
    )

    # Call classifier on the inputs.
    input_data = {
        "input_ids": tf.random.uniform(
            shape=(1, 12), dtype=tf.int64, maxval=50265),
        "input_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)),
    }
    classifier = keras_nlp.models.RobertaClassifier(encoder, 4)
    logits = classifier(input_data)
    ```
    """

    def __init__(
        self,
        base_model,
        num_classes,
        hidden_dim=None,
        bos_token_index=0,
        dropout=0.1,
        name=None,
        trainable=True,
    ):
        inputs = base_model.input
        if hidden_dim is None:
            hidden_dim = base_model.hidden_dim

        x = base_model(inputs)[:, bos_token_index, :]
        x = keras.layers.Dropout(dropout, name="pooler_dropout_1")(x)
        x = keras.layers.Dense(
            hidden_dim, activation="tanh", name="pooled_dense"
        )(x)
        x = keras.layers.Dropout(dropout, name="pooler_dropout_2")(x)
        outputs = keras.layers.Dense(
            num_classes,
            kernel_initializer=_roberta_kernel_initializer(),
            name="logits",
        )(x)

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs=inputs, outputs=outputs, name=name, trainable=trainable
        )
        # All references to `self` below this line
        self.base_model = base_model
        self.num_classes = num_classes


def RobertaBase(name=None, trainable=True):
    """RoBERTa implementation using "Base" architecture.

    This network implements a bi-directional Transformer-based encoder as
    described in ["RoBERTa: A Robustly Optimized BERT Pretraining
    Approach"](https://arxiv.org/abs/1907.11692). It includes the
    embedding lookups and transformer layers, but not the masked language model
    or classification task networks.

    Args:
        name: String, optional. Name of the model.
        trainable: Boolean, optional. If the model's variables should be
            trainable.

    Example usage:
    ```python
    # Randomly initialized RobertaBase encoder
    encoder = keras_nlp.models.RobertaBase()

    # Call encoder on the inputs.
    input_data = {
        "input_ids": tf.random.uniform(
            shape=(1, 512), dtype=tf.int64, maxval=encoder.vocabulary_size),
        "input_mask": tf.constant(
            [1] * 512, shape=(1, 512)),
    }
    output = encoder(input_data)
    ```
    """

    model = Roberta(
        vocabulary_size=50265,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        dropout=0.1,
        max_sequence_length=512,
        name=name,
        trainable=trainable,
    )

    return model
