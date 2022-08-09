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
"""Bert model and layer implementations."""

import tensorflow as tf
from tensorflow import keras

import keras_nlp.layers

# isort: off
# TODO(bischof): decide what to export or whether we are using these decorators
from tensorflow.python.util.tf_export import keras_export

CLS_INDEX = 0
TOKEN_EMBEDDING_LAYER_NAME = "token_embedding"


class Bert(keras.Model):
    """Bi-directional Transformer-based encoder network.

    This network implements a bi-directional Transformer-based encoder as
    described in "BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding" (https://arxiv.org/abs/1810.04805). It includes the
    embedding lookups and transformer layers, but not the masked language model
    or classification task networks.

    The default values for this object are taken from the BERT-Base
    implementation in "BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding".

    Args:
        vocab_size: The size of the token vocabulary.
        num_layers: The number of transformer layers.
        hidden_size: The size of the transformer hidden layers.
        num_attention_heads: The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        inner_size: The output dimension of the first Dense layer in a two-layer
            feedforward network for each transformer.
        inner_activation: The activation for the first Dense layer in a
            two-layer feedforward network for each transformer.
        initializer_range: The initialzer range to use for a truncated normal
            initializer.
        dropout: Dropout probability for the Transformer encoder.
        max_sequence_length: The maximum sequence length that this encoder can
            consume. If None, max_sequence_length uses the value from sequence
            length. This determines the variable shape for positional
            embeddings.
        type_vocab_size: The number of types that the 'segment_ids' input can
            take.
        norm_first: Whether to normalize inputs to attention and intermediate
            dense layers. If set False, output of attention and intermediate
            dense layers is normalized.
    """
    def __init__(
        self,
        vocab_size,
        num_layers,
        hidden_size,
        num_attention_heads,
        inner_size,
        inner_activation="gelu",
        initializer_range=0.02,
        dropout=0.1,
        max_sequence_length=512,
        type_vocab_size=2,
        **kwargs
    ):

        # Create lambda functions from input params
        inner_activation_fn = keras.activations.get(inner_activation)
        initializer_fn = keras.initializers.TruncatedNormal(
            stddev=initializer_range
        )

        # Functional version of model
        token_id_input = keras.Input(shape=(None,), dtype="int32", name="input_ids")
        segment_id_input = keras.Input(
            shape=(None,), dtype="int32", name="segment_ids"
        )
        input_mask = keras.Input(shape=(None,), dtype="int32", name="input_mask")

        # Embed tokens, positions, and segment ids.
        token_embedding = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=hidden_size,
            name=TOKEN_EMBEDDING_LAYER_NAME,
        )(token_id_input)
        position_embedding = keras_nlp.layers.PositionEmbedding(
            initializer=initializer_fn,
            sequence_length=max_sequence_length,
            name="position_embedding",
        )(token_embedding)
        segment_embedding = keras.layers.Embedding(
            input_dim=type_vocab_size,
            output_dim=hidden_size,
            name="segment_embedding",
        )(segment_id_input)

        # Sum, normailze and apply dropout to embeddings.
        x = keras.layers.Add(
            name="embedding_sum",
        )((token_embedding, position_embedding, segment_embedding))
        x = keras.layers.LayerNormalization(
            name="embeddings/layer_norm",
            axis=-1,
            epsilon=1e-12,
            dtype=tf.float32,
        )(x)
        x = keras.layers.Dropout(
            dropout,
            name="embedding_dropout",
        )(x)

        # Apply successive transformer encoder blocks.
        for i in range(num_layers):
            x = keras_nlp.layers.TransformerEncoder(
                num_heads=num_attention_heads,
                intermediate_dim=inner_size,
                activation=inner_activation_fn,
                dropout=dropout,
                kernel_initializer=initializer_fn,
                name="transformer/layer_%d" % i,
            )(x, padding_mask=input_mask)

        # Construct the two BERT outputs, and apply a dense to the pooled output.
        sequence_output = x
        pooled_output = keras.layers.Dense(
            hidden_size,
            activation="tanh",
            name="pooled_dense",
        )(x[:, CLS_INDEX, :])

        # Instantiate using Functional API Model constructor
        super(Bert, self).__init__(
            inputs={
                "input_ids": token_id_input,
                "segment_ids": segment_id_input,
                "input_mask": input_mask,
            },
            outputs={
                "sequence_output": sequence_output,
                "pooled_output": pooled_output,
            },
            **kwargs)
        # All references to `self` below this line
        self.inner_activation_fn = inner_activation_fn
        self.initializer_fn = initializer_fn
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.max_sequence_length = max_sequence_length
        self.type_vocab_size = type_vocab_size
        self.inner_size = inner_size
        self.inner_activation = keras.activations.get(inner_activation)
        self.initializer_range = initializer_range
        self.dropout = dropout
    
    def get_embedding_table(self):
        return self.get_layer(TOKEN_EMBEDDING_LAYER_NAME).embeddings

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_attention_heads": self.num_attention_heads,
                "max_sequence_length": self.max_sequence_length,
                "type_vocab_size": self.type_vocab_size,
                "inner_size": self.inner_size,
                "inner_activation": keras.activations.serialize(
                    self.inner_activation
                ),
                "dropout": self.dropout,
                "initializer_range": self.initializer_range,
            }
        )
        return config


class MaskedLMHead(keras.layers.Layer):
    """Masked language model network head for BERT.

    This layer implements a masked language model based on the provided
    transformer based encoder. It assumes that the encoder network being passed
    has a "get_embedding_table()" method.

    Example:
    ```python
    encoder=modeling.networks.BertEncoder(...)
    lm_layer=MaskedLMHead(embedding_table=encoder.get_embedding_table())
    ```

    Args:
        embedding_table: The embedding table from encoder network.
        inner_activation: The activation, if any, for the inner dense layer.
        initializer: The initializer for the dense layer. Defaults to a Glorot
            uniform initializer.
        output: The output style for this layer. Can be either 'logits' or
            'predictions'.
    """

    def __init__(
        self,
        embedding_table,
        inner_activation="gelu",
        initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_table = embedding_table
        self.inner_activation = keras.activations.get(inner_activation)
        self.initializer = initializer

    def build(self, input_shape):
        self._vocab_size, hidden_size = self.embedding_table.shape
        self.dense = keras.layers.Dense(
            hidden_size,
            activation=self.inner_activation,
            kernel_initializer=self.initializer,
            name="transform/dense",
        )
        self.layer_norm = keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-12, name="transform/LayerNorm"
        )
        self.bias = self.add_weight(
            "output_bias/bias",
            shape=(self._vocab_size,),
            initializer="zeros",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, sequence_data, masked_positions):
        masked_lm_input = self._gather_indexes(sequence_data, masked_positions)
        lm_data = self.dense(masked_lm_input)
        lm_data = self.layer_norm(lm_data)
        lm_data = tf.matmul(lm_data, self.embedding_table, transpose_b=True)
        logits = tf.nn.bias_add(lm_data, self.bias)
        masked_positions_length = (
            masked_positions.shape.as_list()[1] or tf.shape(masked_positions)[1]
        )
        return tf.reshape(
            logits, [-1, masked_positions_length, self._vocab_size]
        )

    def _gather_indexes(self, sequence_tensor, positions):
        """Gathers the vectors at the specific positions, for performance.

        Args:
            sequence_tensor: Sequence output of shape
                (`batch_size`, `seq_length`, `hidden_size`) where `hidden_size`
                is number of hidden units.
            positions: Positions ids of tokens in sequence to mask for
                pretraining of with dimension (batch_size, num_predictions)
                where `num_predictions` is maximum number of tokens to mask out
                and predict per each sequence.

        Returns:
            Masked out sequence tensor of shape (batch_size * num_predictions,
            `hidden_size`).
        """
        sequence_shape = tf.shape(sequence_tensor)
        batch_size, seq_length = sequence_shape[0], sequence_shape[1]
        width = sequence_tensor.shape.as_list()[2] or sequence_shape[2]

        flat_offsets = tf.reshape(
            tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1]
        )
        flat_positions = tf.reshape(positions + flat_offsets, [-1])
        flat_sequence_tensor = tf.reshape(
            sequence_tensor, [batch_size * seq_length, width]
        )
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)

        return output_tensor


class BertLanguageModel(keras.Model):
    """
    MLM + NSP model with BertEncoder.
    """

    def __init__(self, encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        # TODO(jbischof): replace with keras_nlp.layers.MLMHead
        self.masked_lm_head = MaskedLMHead(
            embedding_table=encoder.get_embedding_table(),
            initializer=encoder.initializer_fn,
        )
        self.next_sentence_head = keras.layers.Dense(
            encoder.type_vocab_size,
            kernel_initializer=encoder.initializer_fn,
        )
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.lm_loss_tracker = keras.metrics.Mean(name="lm_loss")
        self.nsp_loss_tracker = keras.metrics.Mean(name="nsp_loss")
        self.lm_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="lm_accuracy"
        )
        self.nsp_accuracy = keras.metrics.SparseCategoricalAccuracy(
            name="nsp_accuracy"
        )

    def call(self, data):
        encoder_output = self.encoder(
            {
                "input_ids": data["input_ids"],
                "input_mask": data["input_mask"],
                "segment_ids": data["segment_ids"],
            }
        )
        sequence_output, pooled_output = encoder_output["sequence_output"], encoder_output["pooled_output"]
        lm_preds = self.masked_lm_head(
            sequence_output, data["masked_lm_positions"]
        )
        nsp_preds = self.next_sentence_head(pooled_output)
        return lm_preds, nsp_preds

    def train_step(self, data):
        with tf.GradientTape() as tape:
            lm_preds, nsp_preds = self(data, training=True)
            lm_labels = data["masked_lm_ids"]
            lm_weights = data["masked_lm_weights"]
            nsp_labels = data["next_sentence_labels"]

            lm_loss = keras.losses.sparse_categorical_crossentropy(
                lm_labels, lm_preds, from_logits=True
            )
            lm_weights_summed = tf.reduce_sum(lm_weights, -1)
            lm_loss = tf.reduce_sum(lm_loss * lm_weights, -1)
            lm_loss = tf.math.divide_no_nan(lm_loss, lm_weights_summed)
            nsp_loss = keras.losses.sparse_categorical_crossentropy(
                nsp_labels, nsp_preds, from_logits=True
            )
            nsp_loss = tf.reduce_mean(nsp_loss)
            loss = lm_loss + nsp_loss

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics
        self.loss_tracker.update_state(loss)
        self.lm_loss_tracker.update_state(lm_loss)
        self.nsp_loss_tracker.update_state(nsp_loss)
        self.lm_accuracy.update_state(lm_labels, lm_preds, lm_weights)
        self.nsp_accuracy.update_state(nsp_labels, nsp_preds)
        return {m.name: m.result() for m in self.metrics}


class BertClassifier(keras.Model):
    """Classifier model with BertEncoder."""

    def __init__(self, encoder, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.num_classes = num_classes
        self._logit_layer = keras.layers.Dense(
            num_classes,
            kernel_initializer=encoder.initializer_fn,
            name="logits",
        )

    def call(self, inputs):
        # Ignore the sequence output, use the pooled output.
        _, pooled_output = self.bert_model(inputs)
        return self._logit_layer(pooled_output)


def BertBase(weights=None):
    """Factory for BertEncoder using "Base" architecture."""

    model = Bert(
        vocab_size=30522,
        num_layers=12,
        hidden_size=768,
        num_attention_heads=12,
        inner_size=3072,
        inner_activation="gelu",
        initializer_range=0.02,
        dropout=0.1,
    )

    # TODO(bischof): add some documentation or magic to load our checkpoints
    # Note: This is pure Keras and also intended to work with user checkpoints
    if weights is not None:
        model.load_weights(weights)

    # TODO(bischof): attach the tokenizer
    return model
