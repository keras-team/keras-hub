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
import csv
import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app
from absl import flags
from tensorflow import keras

import keras_nlp

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "task_name",
    "mrpc",
    "The name of the GLUE task to finetune on.",
)

flags.DEFINE_integer(
    "batch_size",
    32,
    "Batch size of data.",
)

flags.DEFINE_integer(
    "epochs",
    2,
    "Number of epochs to run finetuning.",
)

flags.DEFINE_float(
    "learning_rate",
    5e-5,
    "Learning rate",
)

flags.DEFINE_string(
    "tpu_name",
    None,
    "The name of TPU to connect to. If None, no TPU will be used. If you only "
    "have one TPU, use `local`",
)

flags.DEFINE_string(
    "submission_directory",
    None,
    "The directory to save the glue submission file.",
)

flags.DEFINE_string(
    "load_finetuning_model",
    None,
    "The path to load the finetuning model. If None, the model is trained.",
)

flags.DEFINE_string(
    "save_finetuning_model",
    None,
    "The path to save the finetuning model. If None, the model is not saved.",
)


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


def preprocess_data(preprocess_fn, dataset):
    """Run `proprocess_fn` on input dataset then batch & prefetch."""
    return (
        dataset.map(preprocess_fn)
        .batch(FLAGS.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def generate_submission_files(finetuning_model, test_ds, idx_order):
    """Generate GLUE leaderboard submission files."""
    filenames = {
        "cola": "CoLA.tsv",
        "sst2": "SST-2.tsv",
        "mrpc": "MRPC.tsv",
        "qqp": "QQP.tsv",
        "stsb": "STS-B.tsv",
        "mnli_matched": "MNLI-m.tsv",
        "mnli_mismatched": "MNLI-mm.tsv",
        "qnli": "QNLI.tsv",
        "rte": "RTE.tsv",
        "wnli": "WNLI.tsv",
        "ax": "AX.tsv",
    }

    labelnames = {
        "mnli_matched": ["entailment", "neutral", "contradiction"],
        "mnli_mismatched": ["entailment", "neutral", "contradiction"],
        "ax": ["entailment", "neutral", "contradiction"],
        "qnli": ["entailment", "not_entailment"],
        "rte": ["entailment", "not_entailment"],
    }
    if not os.path.exists(FLAGS.submission_directory):
        os.makedirs(FLAGS.submission_directory)
    filename = FLAGS.submission_directory + "/" + filenames[FLAGS.task_name]
    labelname = labelnames.get(FLAGS.task_name)

    predictions = finetuning_model.predict(test_ds)
    if FLAGS.task_name == "stsb":
        predictions = np.squeeze(predictions)
    else:
        predictions = np.argmax(predictions, -1)

    # Map the predictions to the right index order.
    idx_order = list(idx_order.as_numpy_iterator())
    contents = ["" for _ in idx_order]
    for idx, pred in zip(idx_order, predictions):
        if labelname:
            pred_value = labelname[int(pred)]
        else:
            pred_value = pred
            if FLAGS.task_name == "stsb":
                pred_value = min(pred_value, 5)
                pred_value = max(pred_value, 0)
                pred_value = f"{pred_value:.3f}"
        contents[idx] = pred_value

    with tf.io.gfile.GFile(filename, "w") as f:
        # GLUE requires a format of index + tab + prediction.
        writer = csv.writer(f, delimiter="\t")
        # Write the required headline for GLUE.
        writer.writerow(["index", "prediction"])

        for idx, value in enumerate(contents):
            writer.writerow([idx, value])


def connect_to_tpu(tpu_name):
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(
        tpu=tpu_name
    )
    return tf.distribute.TPUStrategy(resolver)


def main(_):
    if FLAGS.tpu_name:
        strategy = connect_to_tpu(FLAGS.tpu_name)
        policy = keras.mixed_precision.Policy("mixed_bfloat16")
    else:
        # Use default strategy if not using TPU.
        strategy = tf.distribute.get_strategy()
        policy = keras.mixed_precision.Policy("mixed_float16")
    keras.mixed_precision.set_global_policy(policy)

    train_ds, test_ds, val_ds, idx_order = load_data(FLAGS.task_name)
    # ----- Custom code block starts -----
    bert_preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
        "bert_base_en_uncased"
    )

    # Users should change this function to implement the preprocessing required
    # by the model.
    def preprocess_fn(feature, label):
        return bert_preprocessor(feature), label

    # ----- Custom code block ends -----

    train_ds = preprocess_data(preprocess_fn, train_ds)
    val_ds = preprocess_data(preprocess_fn, val_ds)
    test_ds = preprocess_data(preprocess_fn, test_ds)

    if FLAGS.load_finetuning_model:
        with strategy.scope():
            finetuning_model = tf.keras.models.load_model(
                FLAGS.load_finetuning_model
            )
    else:
        with strategy.scope():
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            metrics = [keras.metrics.SparseCategoricalAccuracy()]
            if FLAGS.task_name == "stsb":
                num_classes = 1
                loss = keras.losses.MeanSquaredError()
                metrics = [keras.metrics.MeanSquaredError()]
            elif FLAGS.task_name in (
                "mnli",
                "mnli_mismatched",
                "mnli_matched",
                "ax",
            ):
                num_classes = 3
            else:
                num_classes = 2

            # ----- Custom code block starts -----
            # Users should change this `BertClassifier` to your own classifier.
            # Commonly the classifier is simply your model + several dense layers,
            # please refer to "Make the Finetuning Model" section in README for
            # detailed instructions.
            bert_model = keras_nlp.models.BertBackbone.from_preset(
                "bert_base_en_uncased"
            )
            finetuning_model = keras_nlp.models.BertClassifier(
                backbone=bert_model,
                num_classes=num_classes,
            )
            # ----- Custom code block ends -----
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
            finetuning_model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics,
            )

        finetuning_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=FLAGS.epochs,
        )
    with strategy.scope():
        if FLAGS.submission_directory:
            generate_submission_files(finetuning_model, test_ds, idx_order)
    if FLAGS.save_finetuning_model:
        # Don't need to save the optimizer.
        finetuning_model.optimizer = None
        finetuning_model.save(FLAGS.save_finetuning_model)


if __name__ == "__main__":
    app.run(main)
