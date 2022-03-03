import tensorflow as tf
import numpy as np
from absl import app

SAVED_MODEL_OUTPUT = "saved_models/machine_translation_model"
MAX_SEQUENCE_LENGTH = 20

EXAMPLES = [
    ("Tom doesn't listen to anyone.", "[start] Tomás no escucha a nadie. [end]"),
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
    tokenized_input_sentence = encoder_tokenizer([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_sequence_length):
        tokenized_target_sentence = decoder_tokenizer([decoded_sentence])[:, :-1]
        input = {
            "encoder_inputs": tokenized_input_sentence,
            "decoder_inputs": tokenized_target_sentence,
        }
        predictions = model(input)

        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = lookup_table[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break
    return decoded_sentence


def main(_):
    loaded_model = tf.keras.models.load_model(SAVED_MODEL_OUTPUT)

    decoder_tokenizer = loaded_model.decoder_tokenizer
    vocab = decoder_tokenizer.get_vocabulary()
    index_lookup_table = dict(zip(range(len(vocab)), vocab))

    translated = []
    for example in range(len(EXAMPLES)):
        translated.append(
            decode_sequence(
                example[0],
                loaded_model,
                MAX_SEQUENCE_LENGTH,
                index_lookup_table,
            )
        )

    for i in range(len(EXAMPLES)):
        print("ENGLISH SEnTENCE: ", EXAMPLES[i][0])
        print("MACHINE TRANSLATED RESULT: ", translated[i])
        print("GOLDEN: ", EXAMPLES[i][1])


if __name__ == "__main__":
    app.run(main)
