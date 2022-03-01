from examples.machine_translation.model import TranslationModel
from examples.machine_translation.data import get_dataset_and_tokenizer

import tensorflow as tf
import time
from absl import app

NUM_EPOCHS = 1

EMBEDDING_DIM = 64
INTERMEDIATE_DIM = 128
NUM_HEAD = 2

SEQUENCE_LENGTH = 20
VOCAB_SIZE = 150

SAVED_MODEL_OUTPUT = "saved_models/machine_translation_model"

embedding_params = {
    "sequence_length": SEQUENCE_LENGTH,
    "vocab_size": VOCAB_SIZE,
    "embed_dim": EMBEDDING_DIM,
}

encoder_params = {
    "intermediate_dim": INTERMEDIATE_DIM,
    "num_heads": NUM_HEAD,
}

decoder_params = {
    "intermediate_dim": INTERMEDIATE_DIM,
    "num_heads": NUM_HEAD,
}


def train_loop(model, train_ds, val_ds):
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001, decay_steps=20, decay_rate=0.98
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )
    metrics = tf.keras.metrics.SparseCategoricalAccuracy()
    val_metrics = tf.keras.metrics.SparseCategoricalAccuracy()

    num_steps = tf.data.experimental.cardinality(train_ds).numpy()

    for _ in range(NUM_EPOCHS):
        bar = tf.keras.utils.Progbar(num_steps)
        for i, batch in enumerate(train_ds):
            metrics.reset_state()
            data, label = batch
            mask = tf.cast((label != 0), tf.float32)
            with tf.GradientTape() as tape:
                pred = model(data)
                loss = loss_fn(label, pred) * mask
                loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            metrics.update_state(label, pred, sample_weight=mask)

            time.sleep(0.1)
            values = [("loss", loss), ("accuracy", metrics.result().numpy())]
            bar.update(i, values=values)

            break
        break

        val_metrics.reset_state()
        for batch in val_ds:
            data, label = batch
            mask = tf.cast((label != 0), tf.float32)
            pred = model(data)
            val_metrics.update_state(label, pred, sample_weight=tf.cast(mask, tf.int16))
        print("\nvalidation accuracy: ", val_metrics.result().numpy())


def main(_):
    (train_ds, val_ds, test_ds), (
        eng_tokenizer,
        spa_tokenizer,
    ) = get_dataset_and_tokenizer(SEQUENCE_LENGTH, VOCAB_SIZE)
    model = TranslationModel(
        num_encoders=2,
        num_decoders=2,
        encoder_tokenizer=eng_tokenizer,
        decoder_tokenizer=spa_tokenizer,
        embedding_params=embedding_params,
        encoder_params=encoder_params,
        decoder_params=decoder_params,
    )

    train_loop(model, train_ds, val_ds)

    print(f"Saving to {SAVED_MODEL_OUTPUT}")
    model.save(SAVED_MODEL_OUTPUT)

    loaded_model = tf.keras.models.load_model(SAVED_MODEL_OUTPUT)
    import pdb

    pdb.set_trace()
    data, label = next(iter(test_ds))
    loaded_model(data)


if __name__ == "__main__":
    app.run(main)
