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
"""Run finetuning on a GLUE task."""

import tempfile

import keras_tuner
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app
from absl import flags
from tensorflow import keras

import keras_nlp
from examples.bert.bert_config import FINETUNING_CONFIG
from examples.bert.bert_config import PREPROCESSING_CONFIG

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "vocab_file",
    None,
    "The vocabulary file for tokenization.",
)

flags.DEFINE_string(
    "saved_model_input",
    None,
    "The directory to load the pretrained model.",
)

flags.DEFINE_string(
    "saved_model_output",
    None,
    "The directory to save the finetuned model.",
)

flags.DEFINE_string(
    "task_name",
    "mrpc",
    "The name of the GLUE task to finetune on.",
)

flags.DEFINE_bool(
    "do_lower_case",
    True,
    "Whether to lower case the input text.",
)

flags.DEFINE_bool(
    "do_evaluation",
    True,
    "Whether to run evaluation on test data.",
)


def load_data(task_name):
    if task_name in ("cola", "sst2"):
        feature_names = ("sentence",)
    elif task_name in ("mrpc", "stsb", "rte", "wnli"):
        feature_names = ("sentence1", "sentence2")
    elif task_name in ("mnli", "mnli_matched", "mnli_mismatched"):
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
    elif task_name in ("mnli_mismatched",):
        task_name = "mnli"
        test_suffix = "_mismatched"

    def split_features(x):
        features = tuple([x[name] for name in feature_names])
        labels = x["label"]
        return features, labels

    train_ds, test_ds, validation_ds = tfds.load(
        f"glue/{task_name}",
        split=["train", "test" + test_suffix, "validation" + test_suffix],
    )
    train_ds = train_ds.map(split_features, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(split_features, num_parallel_calls=tf.data.AUTOTUNE)
    validation_ds = validation_ds.map(
        split_features, num_parallel_calls=tf.data.AUTOTUNE
    )
    return train_ds, test_ds, validation_ds


class BertHyperModel(keras_tuner.HyperModel):
    """Creates a hypermodel to help with the search space for finetuning."""

    def build(self, hp):
        model = keras.models.load_model(FLAGS.saved_model_input, compile=False)
        finetuning_model = keras_nlp.models.BertClassifier(
            backbone=model,
            num_classes=3 if FLAGS.task_name in ("mnli", "ax") else 2,
        )
        finetuning_model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=hp.Choice(
                    "lr", FINETUNING_CONFIG["learning_rates"]
                ),
            ),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )
        return finetuning_model


def main(_):
    print(f"Reading input model from {FLAGS.saved_model_input}")

    tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
        vocabulary=FLAGS.vocab_file,
        lowercase=FLAGS.do_lower_case,
    )
    packer = keras_nlp.layers.MultiSegmentPacker(
        sequence_length=PREPROCESSING_CONFIG["max_seq_length"],
        start_value=tokenizer.token_to_id("[CLS]"),
        end_value=tokenizer.token_to_id("[SEP]"),
    )

    def preprocess_data(inputs, labels):
        inputs = [tokenizer(x) for x in inputs]
        token_ids, segment_ids = packer(inputs)
        return {
            "token_ids": token_ids,
            "segment_ids": segment_ids,
            "padding_mask": tf.cast(token_ids != 0, "int32"),
        }, labels

    # Read and preprocess GLUE task data.
    train_ds, test_ds, validation_ds = load_data(FLAGS.task_name)

    batch_size = FINETUNING_CONFIG["batch_size"]
    train_ds = train_ds.batch(batch_size).map(
        preprocess_data, num_parallel_calls=tf.data.AUTOTUNE
    )
    validation_ds = validation_ds.batch(batch_size).map(
        preprocess_data, num_parallel_calls=tf.data.AUTOTUNE
    )
    test_ds = test_ds.batch(batch_size).map(
        preprocess_data, num_parallel_calls=tf.data.AUTOTUNE
    )

    # Create a hypermodel object for a RandomSearch.
    hypermodel = BertHyperModel()

    # Initialize the random search over the 4 learning rate parameters, for 4
    # trials and 3 epochs for each trial.
    tuner = keras_tuner.RandomSearch(
        hypermodel=hypermodel,
        objective=keras_tuner.Objective("val_loss", direction="min"),
        max_trials=4,
        overwrite=True,
        project_name="hyperparameter_tuner_results",
        directory=tempfile.mkdtemp(),
    )

    tuner.search(
        train_ds,
        epochs=FINETUNING_CONFIG["epochs"],
        validation_data=validation_ds,
    )

    # Extract the best hyperparameters after the search.
    best_hp = tuner.get_best_hyperparameters()[0]
    finetuning_model = tuner.get_best_models()[0]

    print(
        f"The best hyperparameters found are:\nLearning Rate: {best_hp['lr']}"
    )

    if FLAGS.do_evaluation:
        print("Evaluating on test set.")
        finetuning_model.evaluate(test_ds)

    if FLAGS.saved_model_output:
        print(f"Saving to {FLAGS.saved_model_output}")
        finetuning_model.save(FLAGS.saved_model_output)


if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("saved_model_input")
    app.run(main)
