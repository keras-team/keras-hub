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

import json
import sys

import tensorflow as tf
from absl import app
from absl import flags
from tensorflow import keras

from examples.bert.bert_model import BertModel
from examples.bert.bert_utils import list_filenames_for_arg

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_files",
    None,
    "Comma seperated list of directories, files, or globs for input data.",
)

flags.DEFINE_string(
    "saved_model_output", None, "Output directory to save the model to."
)

flags.DEFINE_string(
    "bert_config_file",
    None,
    "The json config file for the bert model parameters.",
)

flags.DEFINE_string(
    "vocab_file",
    None,
    "The vocabulary file that the BERT model was trained on.",
)

flags.DEFINE_integer("epochs", 10, "The number of training epochs.")

flags.DEFINE_integer("batch_size", 256, "The training batch size.")

flags.DEFINE_float("learning_rate", 1e-4, "The initial learning rate for Adam.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer(
    "max_predictions_per_seq",
    20,
    "Maximum number of masked LM predictions per sequence.",
)


class ClassificationHead(tf.keras.layers.Layer):
    """Pooling head for sentence-level classification tasks.

    Args:
        inner_dim: The dimensionality of inner projection layer. If 0 or `None`
            then only the output projection layer is created.
        num_classes: Number of output classes.
        cls_token_idx: The index inside the sequence to pool.
        activation: Dense layer activation.
        dropout_rate: Dropout probability.
        initializer: Initializer for dense layer kernels.
        **kwargs: Keyword arguments.
    """

    def __init__(
        self,
        inner_dim,
        num_classes,
        cls_token_idx=0,
        activation="tanh",
        dropout_rate=0.0,
        initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.inner_dim = inner_dim
        self.num_classes = num_classes
        self.activation = keras.activations.get(activation)
        self.initializer = keras.initializers.get(initializer)
        self.cls_token_idx = cls_token_idx

        if self.inner_dim:
            self.dense = keras.layers.Dense(
                units=self.inner_dim,
                activation=self.activation,
                kernel_initializer=self.initializer,
                name="pooler_dense",
            )
        self.dropout = keras.layers.Dropout(rate=self.dropout_rate)

        self.out_proj = keras.layers.Dense(
            units=num_classes,
            kernel_initializer=self.initializer,
            name="logits",
        )

    def call(self, features: tf.Tensor):
        """Implements call().

        Args:
            features: a rank-3 Tensor when self.inner_dim is specified,
                otherwise it is a rank-2 Tensor.

        Returns:
            a Tensor shape= [batch size, num classes].
        """
        if not self.inner_dim:
            x = features
        else:
            x = features[:, self.cls_token_idx, :]  # take <CLS> token.
            x = self.dense(x)

        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    def get_config(self):
        config = {
            "cls_token_idx": self.cls_token_idx,
            "dropout_rate": self.dropout_rate,
            "num_classes": self.num_classes,
            "inner_dim": self.inner_dim,
            "activation": tf.keras.activations.serialize(self.activation),
            "initializer": tf.keras.initializers.serialize(self.initializer),
        }
        config.update(super(ClassificationHead, self).get_config())
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
        activation: The activation, if any, for the dense layer.
        initializer: The initializer for the dense layer. Defaults to a Glorot
            uniform initializer.
        output: The output style for this layer. Can be either 'logits' or
            'predictions'.
    """

    def __init__(self, embedding_table, **kwargs):
        super().__init__(**kwargs)
        self.embedding_table = embedding_table

    def build(self, input_shape):
        self._vocab_size, hidden_size = self.embedding_table.shape
        self.dense = keras.layers.Dense(
            hidden_size, activation=None, name="transform/dense"
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


class BertPretrainer(keras.Model):
    def __init__(self, bert_model, **kwargs):
        super().__init__(**kwargs)
        self.bert_model = bert_model
        self.masked_lm_head = MaskedLMHead(bert_model.get_embedding_table())
        self.next_sentence_head = ClassificationHead(
            inner_dim=768, num_classes=2, dropout_rate=0.1
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
        outputs = self.bert_model(
            {
                "input_ids": data["input_ids"],
                "input_mask": data["input_mask"],
                "segment_ids": data["segment_ids"],
            }
        )
        lm_preds = self.masked_lm_head(outputs, data["masked_lm_positions"])
        nsp_preds = self.next_sentence_head(outputs)
        return lm_preds, nsp_preds

    def train_step(self, data):
        # TODO(mattdangerw): Add metrics (e.g nsp, lm accuracy).
        with tf.GradientTape() as tape:
            lm_preds, nsp_preds = self(data, training=True)
            lm_loss = keras.metrics.sparse_categorical_crossentropy(
                data["masked_lm_ids"], lm_preds, from_logits=True
            )
            lm_weights = data["masked_lm_weights"]
            lm_weights_summed = tf.reduce_sum(lm_weights, -1)
            lm_loss = tf.reduce_sum(lm_loss * lm_weights, -1)
            lm_loss = tf.math.divide_no_nan(lm_loss, lm_weights_summed)
            nsp_loss = keras.metrics.sparse_categorical_crossentropy(
                data["next_sentence_labels"], nsp_preds, from_logits=True
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
        self.lm_accuracy.update_state(data["masked_lm_ids"], lm_preds)
        self.nsp_accuracy.update_state(data["next_sentence_labels"], nsp_preds)
        return {m.name: m.result() for m in self.metrics}


def decode_record(record):
    """Decodes a record to a TensorFlow example."""
    seq_length = FLAGS.max_seq_length
    lm_length = FLAGS.max_predictions_per_seq
    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "masked_lm_positions": tf.io.FixedLenFeature([lm_length], tf.int64),
        "masked_lm_ids": tf.io.FixedLenFeature([lm_length], tf.int64),
        "masked_lm_weights": tf.io.FixedLenFeature([lm_length], tf.float32),
        "next_sentence_labels": tf.io.FixedLenFeature([1], tf.int64),
    }
    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    example = tf.io.parse_single_example(record, name_to_features)
    for name in list(example.keys()):
        value = example[name]
        if value.dtype == tf.int64:
            value = tf.cast(value, tf.int32)
        example[name] = value
    return example


def main(_):
    print(f"Reading input data from {FLAGS.input_files}")
    input_filenames = list_filenames_for_arg(FLAGS.input_files)
    if not input_filenames:
        print("No input files found. Check `input_files` flag.")
        sys.exit(1)

    vocab = []
    with open(FLAGS.vocab_file, "r") as vocab_file:
        for line in vocab_file:
            vocab.append(line.strip())

    with open(FLAGS.bert_config_file, "r") as bert_config_file:
        bert_config = json.loads(bert_config_file.read())

    # Decode and batch data.
    dataset = tf.data.TFRecordDataset(input_filenames)
    dataset = dataset.map(
        lambda record: decode_record(record),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)

    # Create a BERT model the input config.
    model = BertModel(
        vocab_size=len(vocab),
        **bert_config,
    )
    # Make sure model has been called.
    model(model.inputs)
    model.summary()

    # Wrap with pretraining heads and call fit.
    pretraining_model = BertPretrainer(model)
    pretraining_model.compile(
        # TODO(mattdangerw): Add AdamW and a learning rate schedule.
        optimizer=keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    )
    # TODO(mattdangerw): Add TPU strategy support.
    pretraining_model.fit(dataset, epochs=FLAGS.epochs)

    print(f"Saving to {FLAGS.saved_model_output}")
    model.save(FLAGS.saved_model_output)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_files")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("saved_model_output")
    app.run(main)
