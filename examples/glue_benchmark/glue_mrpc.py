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
import os

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app
from absl import flags
from tensorflow import keras

import keras_nlp

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
tf.random.set_seed(seed)

flags.DEFINE_string("model", None, "The Model you want to train and evaluate.")
flags.DEFINE_bool(
    "freeze_backbone",
    True,
    "If you want to freeze the backbone and only train the downstream layers or not.",
)
flags.DEFINE_string(
    "preset",
    None,
    "The model preset(eg. For bert it is 'bert_base_en', 'bert_tiny_en_uncased')",
)
flags.DEFINE_float(
    "learning_rate", 0.005, "The learning_rate for the optimizer."
)
flags.DEFINE_integer("epochs", 2, "No of Epochs.")
flags.DEFINE_integer("batch_size", 8, "Batch Size.")


FLAGS = flags.FLAGS


def load_data():
    """
    Load GLUE dataset.
    Load GLUE dataset, and convert the dictionary format to (features, label),
    where features is a tuple of all input sentences.
    """

    feature_names = ("sentence1", "sentence2")

    def split_features(x):
        # GLUE comes with dictonary data, we convert it to a uniform format
        # (features, label), where features is a tuple consisting of all
        # features.
        features = tuple([x[name] for name in feature_names])
        label = x["label"]
        return (features, label)

    train_ds, test_ds, validation_ds = tfds.load(
        "glue/mrpc",
        split=["train", "test", "validation"],
    )

    train_ds = train_ds.map(split_features, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(split_features, num_parallel_calls=tf.data.AUTOTUNE)
    validation_ds = validation_ds.map(
        split_features, num_parallel_calls=tf.data.AUTOTUNE
    )
    return train_ds, test_ds, validation_ds


def load_model_and_preprocessor(model, preset):
    for name, symbol in keras_nlp.models.__dict__.items():
        if inspect.isclass(symbol) and issubclass(symbol, keras.Model):
            if model and name != model:
                continue
            if not hasattr(symbol, "from_preset"):
                continue
            for _preset in symbol.presets:
                if preset and _preset != preset:
                    continue
                model = symbol.from_preset(preset)
                print(f"Using model {name} with preset {preset}")
                preprocessor = keras_nlp.models.__dict__[
                    name.replace("Backbone", "Preprocessor")
                ].from_preset(preset)

                return model, preprocessor

    raise ValueError(f"Model {model} or preset {preset} not found.")


def create_model(model, preset):
    # output_dim
    output_dim = 2
    # select backbone

    if "backbone" not in FLAGS.model.lower():
        backbones = [
            i
            for i in keras_nlp.models.__dict__
            if "Backbone" in i
            and i not in ["OPTBackbone", "GPT2Backbone", "WhisperBackbone"]
        ]
        raise ValueError(
            f"Please enter a Backbone, from this list - {backbones}, the script will take care of the downstream layers."
        )

    # Build the model
    backbone, preprocessor = load_model_and_preprocessor(
        model=model, preset=preset
    )
    # Freeze
    if FLAGS.freeze_backbone:
        backbone.trainable = False
    else:
        backbone.trainable = True
    # If the model has pooled_output
    if len(backbone.outputs) > 1:
        output = keras.layers.Dense(output_dim)(
            backbone.output["pooled_output"]
        )
    elif len(backbone.outputs) == 1:
        output = keras.layers.Dense(output_dim)(backbone.output)
    model = keras.models.Model(backbone.inputs, output)

    return model, preprocessor


def preprocess_data(dataset, preprocessor):
    """Run `proprocess_fn` on input dataset then batch & prefetch."""

    def preprocess_fn(feature, label):
        return preprocessor(feature), label

    return (
        dataset.map(preprocess_fn)
        .batch(FLAGS.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def main(_):
    print(tf.__version__)
    print("GPU available : ", tf.test.is_gpu_available())

    print("=" * 120)
    print(
        f"MODEL : {FLAGS.model}   PRESET : {FLAGS.preset}   DATASET : glue/mrpc"
    )
    print("=" * 120)

    # Load the model
    model, preprocessor = create_model(model=FLAGS.model, preset=FLAGS.preset)
    # Add loss and optimizer
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [keras.metrics.SparseCategoricalAccuracy()]

    # Load datasets
    train_ds, test_ds, validation_ds = load_data()
    train_ds = preprocess_data(dataset=train_ds, preprocessor=preprocessor)
    validation_ds = preprocess_data(
        dataset=validation_ds, preprocessor=preprocessor
    )

    lr = tf.keras.optimizers.schedules.PolynomialDecay(
        FLAGS.learning_rate,
        decay_steps=train_ds.cardinality() * FLAGS.epochs,
        end_learning_rate=0.0,
    )
    optimizer = tf.keras.optimizers.experimental.AdamW(
        lr, weight_decay=0.01, global_clipnorm=1.0
    )
    optimizer.exclude_from_weight_decay(
        var_names=["LayerNorm", "layer_norm", "bias"]
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Start training
    model.fit(train_ds, validation_data=validation_ds, epochs=FLAGS.epochs)


if __name__ == "__main__":
    flags.mark_flag_as_required("model")
    flags.mark_flag_as_required("preset")
    app.run(main)
