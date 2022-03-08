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
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

# Import data module to include the customized serializable, required for
# loading tokenizer.
import examples.machine_translation.data  # noqa: F401.

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "sequence_length",
    20,
    "Input and output sequence length.",
)

flags.DEFINE_string(
    "saved_model_path",
    "saved_models/machine_translation_model",
    "The path to saved model",
)

flags.DEFINE_string("inputs", None, "The inputs to run machine translation on.")

EXAMPLES = [
    (
        "Tom doesn't listen to anyone.",
        "[start] Tomás no escucha a nadie. [end]",
    ),
    ("I got soaked to the skin.", "[start] Estoy chorreando. [end]"),
    ("I imagined that.", "[start] Me imaginé eso. [end]"),
    ("The baby is crying.", "[start] El bebé está llorando. [end]"),
    (
        "I've never felt so exhilarated.",
        "[start] Nunca me he sentido tan animado. [end]",
    ),
    (
        "Please forgive me for not having written sooner.",
        "[start] Perdóname por no haberte escrito antes, por favor. [end]",
    ),
    ("I expected more from you.", "[start] Esperaba más de vos. [end]"),
    ("I have a computer.", "[start] Tengo un computador. [end]"),
    ("Dinner's ready!", "[start] ¡La cena está lista! [end]"),
    ("Let me finish.", "[start] Déjame terminar. [end]"),
]


def decode_sequence(input_sentence, model, max_sequence_length, lookup_table):
    encoder_tokenizer = model.encoder_tokenizer
    decoder_tokenizer = model.decoder_tokenizer
    tokenized_input = encoder_tokenizer([input_sentence])

    start_token = decoder_tokenizer("[start]")[0].numpy()
    end_token = decoder_tokenizer("[end]")[0].numpy()

    decoded_sentence = [start_token]
    for i in range(max_sequence_length):
        decoder_inputs = tf.convert_to_tensor(
            [decoded_sentence],
            dtype=tf.int64,
        )
        decoder_inputs = tf.concat(
            [
                decoder_inputs,
                tf.zeros(
                    [1, max_sequence_length - i - 1],
                    dtype=tf.int64,
                ),
            ],
            axis=1,
        )
        input = {
            "encoder_inputs": tokenized_input,
            "decoder_inputs": decoder_inputs,
        }
        predictions = model(input)
        predicted_token = np.argmax(predictions[0, i, :])
        decoded_sentence.append(predicted_token)
        if predicted_token == end_token:
            break

    detokenized_output = []
    for token in decoded_sentence:
        detokenized_output.append(lookup_table[token])
    return " ".join(detokenized_output)


def main(_):
    loaded_model = tf.keras.models.load_model(FLAGS.saved_model_path)

    decoder_tokenizer = loaded_model.decoder_tokenizer
    vocab = decoder_tokenizer.get_vocabulary()
    index_lookup_table = dict(zip(range(len(vocab)), vocab))

    if FLAGS.inputs is not None:
        # Run inference on user-specified sentence.
        translated = decode_sequence(
            FLAGS.inputs,
            loaded_model,
            FLAGS.sequence_length,
            index_lookup_table,
        )
        logging.info(f"Translated results: {translated}")

    else:
        translated = []
        for example in EXAMPLES:
            translated.append(
                decode_sequence(
                    example[0],
                    loaded_model,
                    FLAGS.sequence_length,
                    index_lookup_table,
                )
            )

        for i in range(len(EXAMPLES)):
            print("ENGLISH SENTENCE: ", EXAMPLES[i][0])
            print("MACHINE TRANSLATED RESULT: ", translated[i])
            print("GOLDEN: ", EXAMPLES[i][1])


if __name__ == "__main__":
    app.run(main)
