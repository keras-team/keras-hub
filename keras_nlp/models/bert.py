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

"""Bert model configurable class, preconfigured versions, and task heads."""

import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers import PositionEmbedding
from keras_nlp.layers import TransformerEncoder


def _bert_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


class BertCustom(keras.Model):
    """Bi-directional Transformer-based encoder network.

    This network implements a bi-directional Transformer-based encoder as
    described in ["BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding"](https://arxiv.org/abs/1810.04805). It includes the
    embedding lookups and transformer layers, but not the masked language model
    or classification task networks.

    This class gives a fully customizable Bert model with any number of layers,
    heads, and embedding dimensions. For specific specific bert architectures
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
        cls_token_index: Int. Index of [CLS] token in the vocabulary.
        name: String, optional. Name of the model.
        trainable: Boolean, optional. If the model's variables should be
            trainable.

    Example usage:
    ```python
    # Randomly initialized Bert encoder
    encoder = keras_nlp.models.BertCustom(
        vocabulary_size=30522,
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
        "segment_ids": tf.constant(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)),
        "input_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)),
    }
    output = encoder(input_data)
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
        cls_token_index=0,
        name=None,
        trainable=True,
    ):

        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="input_ids"
        )
        segment_id_input = keras.Input(
            shape=(None,), dtype="int32", name="segment_ids"
        )
        input_mask = keras.Input(
            shape=(None,), dtype="int32", name="input_mask"
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
                name=f"""transformer_layer_{i}""",
            )(x, padding_mask=input_mask)

        # Construct the two Bert outputs. The pooled output is a dense layer on
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
                "input_ids": token_id_input,
                "segment_ids": segment_id_input,
                "input_mask": input_mask,
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


class BertClassifier(keras.Model):
    """Bert encoder model with a classification head.

    Args:
        base_model: A `keras_nlp.models.BertCustom` to encode inputs.
        num_classes: Int. Number of classes to predict.
        name: String, optional. Name of the model.
        trainable: Boolean, optional. If the model's variables should be
            trainable.

    Example usage:
    ```
    python
    # Randomly initialized Bert encoder
    encoder = keras_nlp.models.BertCustom(
        vocabulary_size=30522,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12
    )

     # Call classifier on the inputs.
    input_data = {
        "input_ids": tf.random.uniform(
            shape=(1, 12), dtype=tf.int64, maxval=30522),
        "segment_ids": tf.constant(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)),
        "input_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)),
    }
    classifier = keras_nlp.models.BertClassifier(encoder, 4)
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


def BertBase(name=None, trainable=True):
    """Bi-directional Transformer-based encoder network (Bert) using "Base"
    architecture.

    This network implements a bi-directional Transformer-based encoder as
    described in ["BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding"](https://arxiv.org/abs/1810.04805). It includes the
    embedding lookups and transformer layers, but not the masked language model
    or classification task networks.

    Args:
        name: String, optional. Name of the model.
        trainable: Boolean, optional. If the model's variables should be
            trainable.

    Example usage:
    ```python
    # Randomly initialized BertBase encoder
    encoder = keras_nlp.models.BertBase()

     # Call encoder on the inputs.
    input_data = {
        "input_ids": tf.random.uniform(
            shape=(1, 512), dtype=tf.int64, maxval=encoder.vocabulary_size),
        "segment_ids": tf.constant(
            [0] * 200 + [1] * 312, shape=(1, 512)),
        "input_mask": tf.constant(
            [1] * 512, shape=(1, 512)),
    }
    output = encoder(input_data)
    ```
    """

    model = BertCustom(
        vocabulary_size=30522,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        dropout=0.1,
        max_sequence_length=512,
        name=name,
        trainable=trainable,
    )

    # TODO(jbischof): add some documentation or magic to load our checkpoints
    # TODO(jbischof): attach the tokenizer
    return model
