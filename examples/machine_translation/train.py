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

import tensorflow as tf
from absl import app
from absl import flags

from examples.machine_translation.data import get_dataset_and_tokenizer
from examples.machine_translation.model import TranslationModel

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_epochs", 1, "Number of epochs to train.")
flags.DEFINE_integer("steps_per_epoch", None, "Number of steps per epoch.")
flags.DEFINE_integer("num_encoders", 2, "Number of Transformer encoder layers.")
flags.DEFINE_integer("num_decoders", 2, "Number of Transformer decoder layers.")
flags.DEFINE_integer("batch_size", 64, "The training batch size.")
flags.DEFINE_float("learning_rate", 0.001, "The initial learning rate.")
flags.DEFINE_integer("model_dim", 64, "Embedding size.")
flags.DEFINE_integer(
    "intermediate_dim",
    128,
    "Intermediate dimension (feedforward network) of transformer.",
)
flags.DEFINE_integer(
    "num_heads",
    8,
    "Number of head of the multihead attention.",
)
flags.DEFINE_integer(
    "sequence_length",
    20,
    "Input and output sequence length.",
)
flags.DEFINE_integer(
    "vocab_size",
    15000,
    "Vocabulary size, required by tokenizer.",
)

flags.DEFINE_string(
    "saved_model_path",
    "saved_models/machine_translation_model",
    "The path to saved model",
)


def run_training(model, train_ds, val_ds):
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=FLAGS.learning_rate,
        decay_steps=20,
        decay_rate=0.98,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )
    metrics = tf.keras.metrics.SparseCategoricalAccuracy()
    model.compile(optimizer=optimizer, metrics=[metrics], loss=loss_fn)
    model.fit(
        train_ds,
        epochs=FLAGS.num_epochs,
        validation_data=val_ds,
        steps_per_epoch=FLAGS.steps_per_epoch,
    )


def main(_):
    (train_ds, val_ds, test_ds), (
        eng_tokenizer,
        spa_tokenizer,
    ) = get_dataset_and_tokenizer(
        FLAGS.sequence_length, FLAGS.vocab_size, FLAGS.batch_size
    )
    english_vocab_size = eng_tokenizer.vocabulary_size()
    spanish_vocab_size = spa_tokenizer.vocabulary_size()
    model = TranslationModel(
        encoder_tokenizer=eng_tokenizer,
        decoder_tokenizer=spa_tokenizer,
        num_encoders=FLAGS.num_encoders,
        num_decoders=FLAGS.num_decoders,
        num_heads=FLAGS.num_heads,
        transformer_intermediate_dim=FLAGS.intermediate_dim,
        encoder_vocab_size=english_vocab_size,
        decoder_vocab_size=spanish_vocab_size,
        embed_dim=FLAGS.model_dim,
        sequence_length=FLAGS.sequence_length,
    )

    run_training(model, train_ds, val_ds)

    print(f"Saving to {FLAGS.saved_model_path}")
    model.save(FLAGS.saved_model_path)

    print(f"Successfully saved model to {FLAGS.saved_model_path}")


if __name__ == "__main__":
    app.run(main)
