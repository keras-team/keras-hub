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

import json

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as tftext
from absl import app
from absl import flags
from tensorflow import keras

FLAGS = flags.FLAGS

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

flags.DEFINE_string(
    "saved_model_input",
    None,
    "The directory containing the input pretrained model to finetune.",
)

flags.DEFINE_string(
    "saved_model_output", None, "The directory to save the finetuned model in."
)


flags.DEFINE_string(
    "task_name", "mrpc", "The name of the GLUE task to finetune on."
)

flags.DEFINE_bool(
    "do_lower_case",
    True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.",
)

flags.DEFINE_integer("batch_size", 32, "The batch size.")

flags.DEFINE_integer("epochs", 3, "The number of training epochs.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")


def bert_pack_inputs(
    inputs,
    seq_length,
    start_of_sequence_id,
    end_of_segment_id,
    padding_id,
    truncator="round_robin",
):
    """Freestanding equivalent of the BertPackInputs layer."""
    # Sanitize inputs.
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    if not inputs:
        raise ValueError("At least one input is required for packing")
    input_ranks = [rt.shape.rank for rt in inputs]
    if None in input_ranks or len(set(input_ranks)) > 1:
        raise ValueError(
            "All inputs for packing must have the same known rank, "
            "found ranks " + ",".join(input_ranks)
        )
    # Flatten inputs to [batch_size, (tokens)].
    if input_ranks[0] > 2:
        inputs = [rt.merge_dims(1, -1) for rt in inputs]
    # In case inputs weren't truncated (as they should have been),
    # fall back to some ad-hoc truncation.
    num_special_tokens = len(inputs) + 1
    if truncator == "round_robin":
        trimmed_segments = tftext.RoundRobinTrimmer(
            seq_length - num_special_tokens
        ).trim(inputs)
    elif truncator == "waterfall":
        trimmed_segments = tftext.WaterfallTrimmer(
            seq_length - num_special_tokens
        ).trim(inputs)
    else:
        raise ValueError("Unsupported truncator: %s" % truncator)
    # Combine segments.
    segments_combined, segment_ids = tftext.combine_segments(
        trimmed_segments,
        start_of_sequence_id=start_of_sequence_id,
        end_of_segment_id=end_of_segment_id,
    )
    # Pad to dense Tensors.
    input_word_ids, _ = tftext.pad_model_inputs(
        segments_combined, seq_length, pad_value=padding_id
    )
    input_type_ids, input_mask = tftext.pad_model_inputs(
        segment_ids, seq_length, pad_value=0
    )
    # Work around broken shape inference.
    output_shape = tf.stack(
        [
            inputs[0].nrows(out_type=tf.int32),  # batch_size
            tf.cast(seq_length, dtype=tf.int32),
        ]
    )

    def _reshape(t):
        return tf.reshape(t, output_shape)

    # Assemble nest of input tensors as expected by BERT TransformerEncoder.
    return dict(
        input_ids=_reshape(input_word_ids),
        input_mask=_reshape(input_mask),
        segment_ids=_reshape(input_type_ids),
    )


def extract_features(inputs, task_name):
    """Transform tfds inputs to be an (features, labels) tuple."""
    if task_name == "cola":
        return (inputs["sentence"],), inputs["label"]
    elif task_name == "sst2":
        return (inputs["sentence"],), inputs["label"]
    elif task_name == "mrpc":
        return (inputs["sentence1"], inputs["sentence2"]), inputs["label"]
    elif task_name == "qqp":
        return (inputs["question1"], inputs["question2"]), inputs["label"]
    elif task_name == "stsb":
        return (inputs["sentence1"], inputs["sentence2"]), inputs["label"]
    elif task_name == "mnli":
        return (inputs["premise"], inputs["hypothesis"]), inputs["label"]
    elif task_name == "qnli":
        return (inputs["question"], inputs["sentence"]), inputs["label"]
    elif task_name == "rte":
        return (inputs["sentence1"], inputs["sentence2"]), inputs["label"]
    elif task_name == "wnli":
        return (inputs["sentence1"], inputs["sentence2"]), inputs["label"]
    elif task_name == "ax":
        return (inputs["premise"], inputs["hypothesis"]), inputs["label"]
    else:
        raise ValueError(f"Unkown task_name {task_name}.")


class BertFinetuner(keras.Model):
    def __init__(self, bert_model, hidden_size, num_classes, **kwargs):
        super().__init__(**kwargs)
        self._bert_model = bert_model
        self._pooler_layer = keras.layers.Dense(
            units=hidden_size,
            activation="tanh",
            name="pooler",
        )
        self._logit_layer = tf.keras.layers.Dense(
            num_classes,
            name="logits",
        )

    def call(self, inputs):
        outputs = self._bert_model(inputs)
        # Get the first [CLS] token from each output.
        outputs = outputs[:, 0, :]
        outputs = self._pooler_layer(outputs)
        return self._logit_layer(outputs)


def main(_):
    print(f"Reading input model from {FLAGS.saved_model_input}")
    model = keras.models.load_model(FLAGS.saved_model_input)

    vocab = []
    with open(FLAGS.vocab_file, "r") as vocab_file:
        for line in vocab_file:
            vocab.append(line.strip())
    tokenizer = tftext.BertTokenizer(
        FLAGS.vocab_file,
        lower_case=FLAGS.do_lower_case,
        token_out_type=tf.int32,
    )
    start_id = vocab.index("[CLS]")
    end_id = vocab.index("[SEP]")
    pad_id = vocab.index("[PAD]")

    with open(FLAGS.bert_config_file, "r") as bert_config_file:
        bert_config = json.loads(bert_config_file.read())

    def preprocess_data(inputs):
        inputs, labels = extract_features(inputs, task_name=FLAGS.task_name)
        inputs = [tokenizer.tokenize(x) for x in inputs]
        inputs = bert_pack_inputs(
            inputs,
            bert_config["max_sequence_length"],
            start_of_sequence_id=start_id,
            end_of_segment_id=end_id,
            padding_id=pad_id,
        )
        return inputs, labels

    # Read in GLUE task data
    [train_ds, test_ds, validation_ds] = tfds.load(
        f"glue/{FLAGS.task_name}",
        batch_size=FLAGS.batch_size,
        split=["train", "validation", "test"],
    )
    train_ds = train_ds.map(
        preprocess_data, num_parallel_calls=tf.data.AUTOTUNE
    )
    test_ds = test_ds.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    validation_ds = validation_ds.map(
        preprocess_data, num_parallel_calls=tf.data.AUTOTUNE
    )

    num_classes = 3 if FLAGS.task_name in ("mnli", "ax") else 2
    finetuning_model = BertFinetuner(
        model, bert_config["hidden_size"], num_classes
    )
    finetuning_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    finetuning_model.fit(
        train_ds, epochs=FLAGS.epochs, validation_data=validation_ds
    )
    finetuning_model.evaluate(test_ds)

    if FLAGS.saved_model_output:
        print(f"Saving to {FLAGS.saved_model_output}")
        finetuning_model.save(FLAGS.saved_model_output)


if __name__ == "__main__":
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("saved_model_input")
    app.run(main)
