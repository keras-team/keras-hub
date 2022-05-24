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

import datasets
import keras_tuner
import tensorflow as tf
import tensorflow_text as tf_text
from absl import app
from absl import flags
from tensorflow import keras

from examples.bert.bert_config import FINETUNING_CONFIG
from examples.bert.bert_config import MODEL_CONFIGS
from examples.bert.bert_config import PREPROCESSING_CONFIG

FLAGS = flags.FLAGS

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


def pack_inputs(
    inputs,
    seq_length,
    start_of_sequence_id,
    end_of_segment_id,
    padding_id,
):
    # In case inputs weren't truncated (as they should have been),
    # fall back to some ad-hoc truncation.
    trimmed_segments = tf_text.RoundRobinTrimmer(
        seq_length - len(inputs) - 1
    ).trim(inputs)
    # Combine segments.
    segments_combined, segment_ids = tf_text.combine_segments(
        trimmed_segments,
        start_of_sequence_id=start_of_sequence_id,
        end_of_segment_id=end_of_segment_id,
    )
    # Pad to dense Tensors.
    input_word_ids, _ = tf_text.pad_model_inputs(
        segments_combined, seq_length, pad_value=padding_id
    )
    input_type_ids, input_mask = tf_text.pad_model_inputs(
        segment_ids, seq_length, pad_value=0
    )
    # Assemble nest of input tensors as expected by BERT model.
    return {
        "input_ids": input_word_ids,
        "input_mask": input_mask,
        "segment_ids": input_type_ids,
    }


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
        raise ValueError(f"Unkown task_name {task_name}.")

    test_suffix = ""
    if task_name in ("mnli", "mnli_matched"):
        # For "mnli", just run default to "mnli_matched".
        task_name = "mnli"
        test_suffix = "_matched"
    elif task_name in ("mnli_mismatched",):
        task_name = "mnli"
        test_suffix = "_mismatched"

    def to_tf_dataset(split):
        # Format each sample as a tuple of string features and an int label.
        features = tuple([split[f] for f in feature_names])
        label = tf.cast(split["label"], tf.int32)
        return tf.data.Dataset.from_tensor_slices((features, label))

    data = datasets.load_dataset("glue", task_name)
    data.set_format(type="tensorflow")
    train_ds = to_tf_dataset(data["train"])
    test_ds = to_tf_dataset(data["test" + test_suffix])
    validation_ds = to_tf_dataset(data["validation" + test_suffix])
    return train_ds, test_ds, validation_ds


class BertClassificationFinetuner(keras.Model):
    """Adds a classification head to a pre-trained BERT model for finetuning"""

    def __init__(
        self, bert_model, hidden_size, num_classes, initializer, **kwargs
    ):
        super().__init__(**kwargs)
        self.bert_model = bert_model
        self._pooler_layer = keras.layers.Dense(
            hidden_size,
            activation="tanh",
            kernel_initializer=initializer,
            name="pooler",
        )
        self._logit_layer = tf.keras.layers.Dense(
            num_classes,
            kernel_initializer=initializer,
            name="logits",
        )

    def call(self, inputs):
        outputs = self.bert_model(inputs)
        # Get the first [CLS] token from each output.
        outputs = outputs[:, 0, :]
        outputs = self._pooler_layer(outputs)
        return self._logit_layer(outputs)


class BertHyperModel(keras_tuner.HyperModel):
    """Creates a hypermodel to help with the search space for finetuning."""

    def __init__(self, model_config):
        self.model_config = model_config

    def build(self, hp):
        model = keras.models.load_model(FLAGS.saved_model_input, compile=False)
        model_config = self.model_config
        finetuning_model = BertClassificationFinetuner(
            bert_model=model,
            hidden_size=model_config["hidden_size"],
            num_classes=3 if FLAGS.task_name in ("mnli", "ax") else 2,
            initializer=keras.initializers.TruncatedNormal(
                stddev=model_config["initializer_range"]
            ),
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

    vocab = []
    with open(FLAGS.vocab_file, "r") as vocab_file:
        for line in vocab_file:
            vocab.append(line.strip())
    tokenizer = tf_text.BertTokenizer(
        FLAGS.vocab_file,
        lower_case=FLAGS.do_lower_case,
        token_out_type=tf.int32,
    )
    start_id = vocab.index("[CLS]")
    end_id = vocab.index("[SEP]")
    pad_id = vocab.index("[PAD]")

    model_config = MODEL_CONFIGS[FLAGS.model_size]

    def preprocess_data(inputs, labels):
        inputs = [tokenizer.tokenize(x).merge_dims(1, -1) for x in inputs]
        inputs = pack_inputs(
            inputs,
            PREPROCESSING_CONFIG["max_seq_length"],
            start_of_sequence_id=start_id,
            end_of_segment_id=end_id,
            padding_id=pad_id,
        )
        return inputs, labels

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
    hypermodel = BertHyperModel(model_config)

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

    # TODO(mattdangerw): After incorporating keras_nlp tokenization, save an
    # end-to-end model includeing preprocessing that operates on raw strings.
    if FLAGS.saved_model_output:
        print(f"Saving to {FLAGS.saved_model_output}")
        finetuning_model.save(FLAGS.saved_model_output)


if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("saved_model_input")
    app.run(main)
