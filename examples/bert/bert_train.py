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

from examples.bert.bert_config import MODEL_CONFIGS
from examples.bert.bert_config import PREPROCESSING_CONFIG
from examples.bert.bert_config import TRAINING_CONFIG
from keras_nlp.applications.bert import Bert
from keras_nlp.applications.bert import BertLanguageModel

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
        # Create a BERT model the input config.
        encoder = Bert(vocab_size=len(vocab), **model_config)
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

        language_model = BertLanguageModel(encoder)
        language_model.compile(
            optimizer=optimizer,
        )

    epochs = TRAINING_CONFIG["epochs"]
    steps_per_epoch = num_train_steps // epochs

    callbacks = []
    if FLAGS.checkpoint_save_directory:
        callbacks.append(get_checkpoint_callback())
    if FLAGS.tensorboard_log_path:
        callbacks.append(get_tensorboard_callback())

    language_model.fit(
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
