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
flags.DEFINE_string(
    "preset",
    None,
    "The model preset(eg. For bert it is 'bert_base_en', 'bert_tiny_en_uncased')",
)
flags.DEFINE_string("task", "stsb", "The task you want the model to train on.")
flags.DEFINE_float(
    "learning_rate", 0.005, "The learning_rate for the optimizer."
)
flags.DEFINE_integer("epochs", 2, "No of Epochs.")
flags.DEFINE_integer("batch_size", 8, "Batch Size.")


FLAGS = flags.FLAGS


def load_data(task_name):
    """
    Load GLUE dataset.
    Load GLUE dataset, and convert the dictionary format to (features, label),
    where features is a tuple of all input sentences.
    """
    if task_name in ("cola", "sst2"):
        feature_names = ("sentence",)
    elif task_name in ("mrpc", "stsb", "rte", "wnli"):
        feature_names = ("sentence1", "sentence2")
    elif task_name in ("mnli", "mnli_matched", "mnli_mismatched", "ax"):
        feature_names = ("premise", "hypothesis")
    elif task_name in "qnli":
        feature_names = ("question", "sentence")
    elif task_name in "qqp":
        feature_names = ("question1", "question2")
    else:
        raise ValueError(f"Unknown task_name {task_name}.")

    test_suffix = ""
    if task_name in ("mnli", "mnli_matched"):
        # For "mnli", just run default to "mnli_matched".
        task_name = "mnli"
        test_suffix = "_matched"
    elif task_name in ("mnli_mismatched"):
        task_name = "mnli"
        test_suffix = "_mismatched"

    def split_features(x):
        # GLUE comes with dictonary data, we convert it to a uniform format
        # (features, label), where features is a tuple consisting of all
        # features.
        features = tuple([x[name] for name in feature_names])
        label = x["label"]
        return (features, label)

    if task_name == "ax":
        # AX is trained and evaluated on MNLI, and has its own test split.
        train_ds, validation_ds = tfds.load(
            "glue/mnli",
            split=["train", "validation_matched"],
        )
        test_ds = tfds.load(
            "glue/ax",
            split="test",
        )
    else:
        train_ds, test_ds, validation_ds = tfds.load(
            f"glue/{task_name}",
            split=["train", "test" + test_suffix, "validation" + test_suffix],
        )

    # Extract out the index order of test dataset.
    idx_order = test_ds.map(lambda data: data["idx"])

    train_ds = train_ds.map(split_features, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(split_features, num_parallel_calls=tf.data.AUTOTUNE)
    validation_ds = validation_ds.map(
        split_features, num_parallel_calls=tf.data.AUTOTUNE
    )
    return train_ds, test_ds, validation_ds, idx_order


def create_model(model, preset, task):
    # output_dim
    if task in ("cola", "sst2", "mrpc", "qqp", "rte", "qnli", "wnli"):
        output_dim = 2
    elif task in ("mnli", "mnli_matched", "mnli_mismatched", "ax"):
        output_dim = 3
    elif task in ("stsb"):
        output_dim = 1
    else:
        raise ValueError(
            f"Task not supported! Please choose a task from {('cola', 'sst2', 'mrpc', 'qqp', 'rte', 'qnli', 'wnli', 'mnli', 'mnli_matched', 'mnli_mismatched', 'ax', 'stsb')}"
        )

    # select backbone
    backbone_dict = {
        "bert": keras_nlp.models.BertBackbone,
        "albert": keras_nlp.models.AlbertBackbone,
        "deberta": keras_nlp.models.DebertaV3Backbone,
        "distil-bert": keras_nlp.models.DistilBertBackbone,
        "roberta": keras_nlp.models.RobertaBackbone,
        "xlm-roberta": keras_nlp.models.XLMRobertaBackbone,
        "f_net": keras_nlp.models.FNetBackbone,
    }
    if model not in list(backbone_dict.keys()):
        raise ValueError(
            f"Model is either not an Encoder based model(eg. Bert, Albert) or "
            f"not supported at this moment! Please select a model from here - {tuple(backbone_dict.keys())}"
        )

    # Build the model
    backbone = backbone_dict[model].from_preset(preset)
    # If the model has pooled_output
    if len(backbone.outputs) > 1:
        output = keras.layers.Dense(output_dim)(
            backbone.output["pooled_output"]
        )
    elif len(backbone.outputs) == 1:
        output = keras.layers.Dense(output_dim)(backbone.output)
    model = keras.models.Model(backbone.inputs, output)

    return model


def preprocess_data(dataset, model, preset):
    """Run `proprocess_fn` on input dataset then batch & prefetch."""

    preprocessor_dict = {
        "bert": keras_nlp.models.BertPreprocessor,
        "albert": keras_nlp.models.AlbertPreprocessor,
        "deberta": keras_nlp.models.DebertaV3Preprocessor,
        "distil-bert": keras_nlp.models.DistilBertPreprocessor,
        "roberta": keras_nlp.models.RobertaPreprocessor,
        "xlm-roberta": keras_nlp.models.XLMRobertaPreprocessor,
        "f_net": keras_nlp.models.FNetPreprocessor,
    }
    if model not in list(preprocessor_dict.keys()):
        raise ValueError(
            f"Model does not have a preprocessor class. This class is required for preprocessing "
            f"of the data before feeding it to the model! Please select a model from here - {tuple(preprocessor_dict.keys())}"
        )

    preprocessor = preprocessor_dict[model].from_preset(preset)

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
        f"MODEL : {FLAGS.model}   PRESET : {FLAGS.preset}   DATASET : {FLAGS.task}"
    )
    print("=" * 120)

    # Load the model
    model = create_model(
        model=FLAGS.model, preset=FLAGS.preset, task=FLAGS.task
    )
    # Add loss and optimizer
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [keras.metrics.SparseCategoricalAccuracy()]
    if FLAGS.task == "stsb":
        loss = keras.losses.MeanSquaredError()
        metrics = [keras.metrics.MeanSquaredError()]

    # Load datasets
    train_ds, test_ds, validation_ds, idx_order = load_data(FLAGS.task)
    train_ds = preprocess_data(
        dataset=train_ds, model=FLAGS.model, preset=FLAGS.preset
    )
    validation_ds = preprocess_data(
        dataset=validation_ds, model=FLAGS.model, preset=FLAGS.preset
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
