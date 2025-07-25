"""GLUE benchmark script to test model performance.

To run the script, use this command:
```
python3 glue.py --model BertTextClassifier \
                --preset bert_base_en \
                --epochs 5 \
                --batch_size 16 \
                --learning_rate 0.001 \
                --mixed_precision_policy mixed_float16
```

Disclaimer: This script only supports GLUE/mrpc (for now).
"""

import inspect
import time

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app
from absl import flags
from absl import logging
from tensorflow import keras

import keras_hub

seed = 42
tf.random.set_seed(seed)


flags.DEFINE_string(
    "task",
    "mrpc",
    "The name of the GLUE task to finetune on.",
)
flags.DEFINE_string(
    "model", None, "The name of the classifier such as BertTextClassifier."
)
flags.DEFINE_string(
    "preset",
    None,
    "The model preset, e.g., 'bert_base_en_uncased' for `BertTextClassifier`",
)
flags.DEFINE_float(
    "learning_rate", 0.005, "The learning_rate for the optimizer."
)
flags.DEFINE_string(
    "mixed_precision_policy",
    "mixed_float16",
    "The global precision policy to use, e.g., 'mixed_float16' or 'float32'.",
)
flags.DEFINE_integer("epochs", 2, "The number of epochs.")
flags.DEFINE_integer("batch_size", 8, "Batch Size.")


FLAGS = flags.FLAGS


def load_data():
    """Load data.

    Load GLUE/MRPC dataset, and convert the dictionary format to
    (features, label), where `features` is a tuple of all input sentences.
    """
    feature_names = ("sentence1", "sentence2")

    def split_features(x):
        # GLUE comes with dictionary data, we convert it to a uniform format
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


def load_model(model, preset, num_classes):
    for name, symbol in keras_hub.models.__dict__.items():
        if inspect.isclass(symbol) and issubclass(symbol, keras.Model):
            if model and name != model:
                continue
            if not hasattr(symbol, "from_preset"):
                continue
            for current_preset in symbol.presets:
                if current_preset and current_preset != preset:
                    continue
                model = symbol.from_preset(current_preset, num_classes=num_classes)
                logging.info(f"\nUsing model {name} with preset {current_preset}\n")
                return model

    raise ValueError(f"Model {model} or preset {preset} not found.")


def main(_):
    keras.mixed_precision.set_global_policy(FLAGS.mixed_precision_policy)

    # Check task is supported.
    # TODO(chenmoneygithub): Add support for other glue tasks.
    if FLAGS.task != "mrpc":
        raise ValueError(
            f"For now only mrpc is supported, but received {FLAGS.task}."
        )

    logging.info(
        "Benchmarking configs...\n"
        "=========================\n"
        f"MODEL: {FLAGS.model}\n"
        f"PRESET: {FLAGS.preset}\n"
        f"TASK: glue/{FLAGS.task}\n"
        f"BATCH_SIZE: {FLAGS.batch_size}\n"
        f"EPOCHS: {FLAGS.epochs}\n"
        "=========================\n"
    )

    # Load datasets.
    train_ds, test_ds, validation_ds = load_data()
    train_ds = train_ds.batch(FLAGS.batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(FLAGS.batch_size).prefetch(tf.data.AUTOTUNE)
    validation_ds = validation_ds.batch(FLAGS.batch_size).prefetch(
        tf.data.AUTOTUNE
    )

    # Load the model.
    model = load_model(model=FLAGS.model, preset=FLAGS.preset, num_classes=2)
    # Set loss and metrics.
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [keras.metrics.SparseCategoricalAccuracy()]
    # Configure optimizer.
    lr = tf.keras.optimizers.schedules.PolynomialDecay(
        FLAGS.learning_rate,
        decay_steps=train_ds.cardinality() * FLAGS.epochs,
        end_learning_rate=0.0,
    )
    optimizer = tf.keras.optimizers.experimental.AdamW(lr, weight_decay=0.01)
    optimizer.exclude_from_weight_decay(
        var_names=["LayerNorm", "layer_norm", "bias"]
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Start training.
    logging.info("Starting Training...")

    st = time.time()
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=FLAGS.epochs,
    )

    wall_time = time.time() - st
    validation_accuracy = history.history["val_sparse_categorical_accuracy"][-1]
    examples_per_second = (
        FLAGS.epochs * FLAGS.batch_size * (len(train_ds) + len(validation_ds))
    ) / wall_time

    logging.info("Training Finished!")
    logging.info(f"Wall Time: {wall_time:.4f} seconds.")
    logging.info(f"Validation Accuracy: {validation_accuracy:.4f}")
    logging.info(f"examples_per_second: {examples_per_second:.4f}")


if __name__ == "__main__":
    flags.mark_flag_as_required("model")
    flags.mark_flag_as_required("preset")
    app.run(main)
