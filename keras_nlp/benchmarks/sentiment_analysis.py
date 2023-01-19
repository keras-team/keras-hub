# Copyright 2023 The KerasNLP Authors
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
import inspect
import time

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app
from absl import flags
from tensorflow import keras

import keras_nlp

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "model",
    None,
    "The name of the classifier such as BertClassifier.",
)
flags.DEFINE_string(
    "preset",
    None,
    "The name of a preset, e.g. bert_base_multi.",
)

flags.DEFINE_string(
    "mixed_precision_policy",
    "mixed_float16",
    "The global precision policy to use. E.g. 'mixed_float16' or 'float32'.",
)

flags.DEFINE_float("learning_rate", 5e-5, "The learning rate.")
flags.DEFINE_integer("num_epochs", 1, "The number of epochs.")
flags.DEFINE_integer("batch_size", 16, "The batch size.")

tfds.disable_progress_bar()

BUFFER_SIZE = 10000


def create_imdb_dataset():
    dataset, info = tfds.load(
        "imdb_reviews", as_supervised=True, with_info=True
    )
    train_dataset, test_dataset = dataset["train"], dataset["test"]

    train_dataset = (
        train_dataset.shuffle(BUFFER_SIZE)
        .batch(FLAGS.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # We split the test data evenly into validation and test sets.
    test_dataset_size = info.splits["test"].num_examples // 2

    val_dataset = (
        test_dataset.take(test_dataset_size)
        .batch(FLAGS.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_dataset = (
        test_dataset.skip(test_dataset_size)
        .batch(FLAGS.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_dataset, val_dataset, test_dataset


def create_model():
    for name, symbol in keras_nlp.models.__dict__.items():
        if inspect.isclass(symbol) and issubclass(symbol, keras.Model):
            if FLAGS.model and name != FLAGS.model:
                continue
            if not hasattr(symbol, "from_preset"):
                continue
            for preset in symbol.presets:
                if FLAGS.preset and preset != FLAGS.preset:
                    continue
                model = symbol.from_preset(preset)
                print(f"Using model {name} with preset {preset}")
                return model

    raise ValueError(f"Model {FLAGS.model} or preset {FLAGS.preset} not found.")


def train_model(
    model: keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
):
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(5e-5),
        metrics=keras.metrics.SparseCategoricalAccuracy(),
        jit_compile=True,
    )

    model.fit(
        train_dataset,
        epochs=FLAGS.num_epochs,
        validation_data=validation_dataset,
        verbose=2,
    )

    return model


def evaluate_model(model: keras.Model, test_dataset: tf.data.Dataset):
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Test loss: {loss}")
    print(f"Test accuracy: {accuracy}")


def main(_):
    keras.mixed_precision.set_global_policy(FLAGS.mixed_precision_policy)

    # Start time
    start_time = time.time()

    train_dataset, validation_dataset, test_dataset = create_imdb_dataset()
    model = create_model()
    model = train_model(model, train_dataset, validation_dataset)
    evaluate_model(model, test_dataset)

    # End time
    end_time = time.time()
    print(f"Total wall time: {end_time - start_time}")


if __name__ == "__main__":
    flags.mark_flag_as_required("model")
    app.run(main)
