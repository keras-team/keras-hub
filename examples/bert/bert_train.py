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

import datetime
import sys

import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from tensorflow import keras

import keras_nlp
from examples.bert.bert_config import MODEL_CONFIGS
from examples.bert.bert_config import PREPROCESSING_CONFIG
from examples.bert.bert_config import TRAINING_CONFIG

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_directory",
    None,
    "The directory of training data. It can be a local disk path, or the URL "
    "of Google cloud storage bucket.",
)

flags.DEFINE_string(
    "saved_model_output",
    None,
    "Output directory to save the model to.",
)

flags.DEFINE_string(
    "checkpoint_save_directory",
    None,
    "Output directory to save checkpoints to.",
)

flags.DEFINE_bool(
    "skip_restore",
    False,
    "Skip restoring from checkpoint if True",
)

flags.DEFINE_bool(
    "tpu_name",
    None,
    "The TPU to connect to. If None, TPU will not be used.",
)

flags.DEFINE_bool(
    "enable_cloud_logging",
    False,
    "If True, the script will use cloud logging.",
)

flags.DEFINE_string(
    "tensorboard_log_path",
    None,
    "The path to save tensorboard log to.",
)

flags.DEFINE_string(
    "model_size",
    "tiny",
    "One of: tiny, mini, small, medium, base, or large.",
)

flags.DEFINE_string(
    "vocab_file",
    None,
    "The vocabulary file for tokenization.",
)

flags.DEFINE_integer(
    "num_train_steps",
    None,
    "Override the pre-configured number of train steps..",
)


class MaskedLMHead(keras.layers.Layer):
    """Masked language model network head for BERT.

    This layer implements a masked language model based on the provided
    transformer based encoder. It assumes that the encoder network being passed
    has a "get_embedding_table()" method.

    Example:
    ```python
    encoder = keras_nlp.models.BertBackbone(
        vocabulary_size=30552,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12,
    )
    lm_layer = MaskedLMHead(embedding_table=encoder.get_embedding_table())
    ```

    Args:
        embedding_table: The embedding table from encoder network.
        intermediate_activation: The activation, if any, for the inner dense
            layer.
        initializer: The initializer for the dense layer. Defaults to a Glorot
            uniform initializer.
        output: The output style for this layer. Can be either 'logits' or
            'predictions'.
    """

    def __init__(
        self,
        embedding_table,
        intermediate_activation="gelu",
        initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_table = embedding_table
        self.intermediate_activation = keras.activations.get(
            intermediate_activation
        )
        self.initializer = initializer

    def build(self, input_shape):
        self._vocab_size, hidden_dim = self.embedding_table.shape
        self.dense = keras.layers.Dense(
            hidden_dim,
            activation=self.intermediate_activation,
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
                (`batch_size`, `seq_length`, `hidden_dim`) where `hidden_dim`
                is number of hidden units.
            positions: Positions ids of tokens in sequence to mask for
                pretraining of with dimension (batch_size, num_predictions)
                where `num_predictions` is maximum number of tokens to mask out
                and predict per each sequence.

        Returns:
            Masked out sequence tensor of shape (batch_size * num_predictions,
            `hidden_dim`).
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


class BertPretrainingModel(keras.Model):
    """MaskedLM + NSP model with Bert encoder."""

    def __init__(self, encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        # TODO(jbischof): replace with keras_nlp.layers.MaskedLMHead (Issue #166)
        self.masked_lm_head = MaskedLMHead(
            embedding_table=encoder.token_embedding.embeddings,
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            name="mlm_layer",
        )
        self.next_sentence_head = keras.layers.Dense(
            encoder.num_segments,
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            name="nsp_layer",
        )

    def call(self, data):
        encoder_output = self.encoder(
            {
                "token_ids": data["token_ids"],
                "segment_ids": data["segment_ids"],
                "padding_mask": data["padding_mask"],
            }
        )
        sequence_output, pooled_output = (
            encoder_output["sequence_output"],
            encoder_output["pooled_output"],
        )
        lm_preds = self.masked_lm_head(
            sequence_output, data["masked_lm_positions"]
        )
        nsp_preds = self.next_sentence_head(pooled_output)
        return {"mlm": lm_preds, "nsp": nsp_preds}


class LinearDecayWithWarmup(keras.optimizers.schedules.LearningRateSchedule):
    """
    A learning rate schedule with linear warmup and decay.

    This schedule implements a linear warmup for the first `num_warmup_steps`
    and a linear ramp down until `num_train_steps`.
    """

    def __init__(self, learning_rate, num_warmup_steps, num_train_steps):
        self.learning_rate = learning_rate
        self.warmup_steps = num_warmup_steps
        self.train_steps = num_train_steps

    def __call__(self, step):
        peak_lr = tf.cast(self.learning_rate, dtype=tf.float32)
        warmup = tf.cast(self.warmup_steps, dtype=tf.float32)
        training = tf.cast(self.train_steps, dtype=tf.float32)
        step = tf.cast(step, dtype=tf.float32)

        is_warmup = step < warmup

        # Linear Warmup will be implemented if current step is less than
        # `num_warmup_steps` else Linear Decay will be implemented.
        return tf.cond(
            is_warmup,
            lambda: peak_lr * (step / warmup),
            lambda: tf.math.maximum(
                0.0, peak_lr * (training - step) / (training - warmup)
            ),
        )

    def get_config(self):
        return {
            "learning_rate": self.learning_rate,
            "num_warmup_steps": self.warmup_steps,
            "num_train_steps": self.train_steps,
        }


def decode_record(record):
    """Decodes a record to a TensorFlow example."""
    seq_length = PREPROCESSING_CONFIG["max_seq_length"]
    lm_length = PREPROCESSING_CONFIG["max_predictions_per_seq"]
    name_to_features = {
        "token_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
        "padding_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
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

    inputs = {
        "token_ids": example["token_ids"],
        "padding_mask": example["padding_mask"],
        "segment_ids": example["segment_ids"],
        "masked_lm_positions": example["masked_lm_positions"],
    }
    labels = {
        "mlm": example["masked_lm_ids"],
        "nsp": example["next_sentence_labels"],
    }
    sample_weights = {"mlm": example["masked_lm_weights"], "nsp": tf.ones((1,))}
    sample = (inputs, labels, sample_weights)
    return sample


def get_checkpoint_callback():
    if tf.io.gfile.exists(FLAGS.checkpoint_save_directory):
        if not tf.io.gfile.isdir(FLAGS.checkpoint_save_directory):
            raise ValueError(
                "`checkpoint_save_directory` should be a directory, "
                f"but {FLAGS.checkpoint_save_directory} is not a "
                "directory. Please set `checkpoint_save_directory` as "
                "a directory."
            )

        elif FLAGS.skip_restore:
            # Clear up the directory if users want to skip restoring.
            tf.io.gfile.rmtree(FLAGS.checkpoint_save_directory)
    checkpoint_path = FLAGS.checkpoint_save_directory
    return keras.callbacks.BackupAndRestore(
        backup_dir=checkpoint_path,
    )


def get_tensorboard_callback():
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = FLAGS.tensorboard_log_path + timestamp
    return keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


def main(_):
    if FLAGS.enable_cloud_logging:
        # If the job is on cloud, we will use cloud logging.
        import google.cloud.logging

        keras.utils.disable_interactive_logging()
        client = google.cloud.logging.Client()
        client.setup_logging()

    logging.info(f"Reading input data from {FLAGS.input_directory}")
    if not tf.io.gfile.isdir(FLAGS.input_directory):
        raise ValueError(
            "`input_directory` should be a directory, "
            f"but {FLAGS.input_directory} is not a directory. Please "
            "set `input_directory` flag as a directory."
        )
    files = tf.io.gfile.listdir(FLAGS.input_directory)
    input_filenames = [FLAGS.input_directory + "/" + file for file in files]

    if not input_filenames:
        logging.info("No input files found. Check `input_directory` flag.")
        sys.exit(1)

    vocab = []
    with tf.io.gfile.GFile(FLAGS.vocab_file) as vocab_file:
        for line in vocab_file:
            vocab.append(line.strip())

    model_config = MODEL_CONFIGS[FLAGS.model_size]

    if FLAGS.tpu_name is None:
        # Use default strategy if not using TPU.
        strategy = tf.distribute.get_strategy()
    else:
        # Connect to TPU and create TPU strategy.
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(
            tpu=FLAGS.tpu_name
        )
        strategy = tf.distribute.TPUStrategy(resolver)

    # Decode and batch data.
    dataset = tf.data.TFRecordDataset(input_filenames)
    dataset = dataset.map(
        lambda record: decode_record(record),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    dataset = dataset.batch(TRAINING_CONFIG["batch_size"], drop_remainder=True)
    dataset = dataset.repeat()

    with strategy.scope():
        # Create a Bert model the input config.
        encoder = keras_nlp.models.BertBackbone(
            vocabulary_size=len(vocab), **model_config
        )
        # Make sure model has been called.
        encoder(encoder.inputs)
        encoder.summary()

        # Allow overriding train steps from the command line for quick testing.
        if FLAGS.num_train_steps is not None:
            num_train_steps = FLAGS.num_train_steps
        else:
            num_train_steps = TRAINING_CONFIG["num_train_steps"]
        num_warmup_steps = int(
            num_train_steps * TRAINING_CONFIG["warmup_percentage"]
        )
        learning_rate_schedule = LinearDecayWithWarmup(
            learning_rate=TRAINING_CONFIG["learning_rate"],
            num_warmup_steps=num_warmup_steps,
            num_train_steps=num_train_steps,
        )
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate_schedule)

        lm_loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            name="lm_loss",
        )
        nsp_loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            name="nsp_loss",
        )

        lm_accuracy = keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        nsp_accuracy = keras.metrics.SparseCategoricalAccuracy(name="accuracy")

        pretraining_model = BertPretrainingModel(encoder)
        pretraining_model.compile(
            optimizer=optimizer,
            loss={"mlm": lm_loss, "nsp": nsp_loss},
            weighted_metrics={"mlm": lm_accuracy, "nsp": nsp_accuracy},
        )

    epochs = TRAINING_CONFIG["epochs"]
    steps_per_epoch = num_train_steps // epochs

    callbacks = []
    if FLAGS.checkpoint_save_directory:
        callbacks.append(get_checkpoint_callback())
    if FLAGS.tensorboard_log_path:
        callbacks.append(get_tensorboard_callback())

    pretraining_model.fit(
        dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
    )

    model_path = FLAGS.saved_model_output
    logging.info(f"Saving to {FLAGS.saved_model_output}")
    encoder.save(model_path)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_directory")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("saved_model_output")
    app.run(main)
