"""Benchmark for text generation."""

import time

import tensorflow as tf
from tensorflow import keras

import keras_hub

SEED = 42

DATASET_ARGS = {
    "vocab_size": 40000,
    "num_samples": 1000,
    "batch_size": 2,
}

MODEL_ARGS = {
    "max_length": 64,
    "embed_dim": 768,
    "num_layers": 8,
    "num_heads": 8,
    "ff_dim": 3072,
}

TEST_RUNS = [
    {
        "sampler": "greedy",
        "execution_methods": ["xla", "graph"],
    },
    {
        "sampler": "beam",
        "execution_methods": ["xla", "graph"],
    },
    {
        "sampler": "top_k",
        "execution_methods": ["xla", "graph"],
    },
    {
        "sampler": "top_p",
        "execution_methods": ["xla", "graph"],
    },
]


def generate_random_ds(vocab_size, num_samples, batch_size, length, seed):
    inputs = tf.random.uniform(
        shape=(num_samples, length),
        minval=0,
        maxval=vocab_size - 1,
        dtype=tf.dtypes.int32,
        seed=seed,
    )

    ds = tf.data.Dataset.from_tensor_slices(inputs)
    ds = ds.batch(batch_size)
    return ds


def build_model(
    vocab_size, max_length, embed_dim, num_layers, num_heads, ff_dim
):
    inputs = keras.layers.Input(shape=(None,), dtype="int32")
    # Embedding.
    x = keras_hub.layers.TokenAndPositionEmbedding(
        vocabulary_size=vocab_size,
        sequence_length=max_length,
        embedding_dim=embed_dim,
        mask_zero=True,
    )(inputs)
    # Transformer decoders.
    for _ in range(num_layers):
        x = keras_hub.layers.TransformerDecoder(
            num_heads=num_heads,
            intermediate_dim=ff_dim,
        )(x)
    # Output.
    outputs = keras.layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def generate_text(
    sampler,
    next,
    prompt,
    jit_compile,
):
    class TestModel(tf.keras.Model):
        def call(self, inputs):
            generated = keras_hub.samplers.get(sampler)(
                next=next,
                prompt=inputs,
            )
            return generated

    test_model = TestModel()
    test_model.compile(jit_compile=jit_compile)

    t0 = time.time()
    _ = test_model.predict(prompt)
    return time.time() - t0


def main():
    keras.utils.set_random_seed(SEED)
    csv_path = time.strftime("text_gen_%Y-%m-%d_%H-%M-%S.csv")

    ds = generate_random_ds(
        vocab_size=DATASET_ARGS["vocab_size"],
        num_samples=DATASET_ARGS["num_samples"],
        batch_size=DATASET_ARGS["batch_size"],
        length=MODEL_ARGS["max_length"],
        seed=SEED,
    )

    model = build_model(
        vocab_size=DATASET_ARGS["vocab_size"],
        max_length=MODEL_ARGS["max_length"],
        embed_dim=MODEL_ARGS["embed_dim"],
        num_layers=MODEL_ARGS["num_layers"],
        num_heads=MODEL_ARGS["num_heads"],
        ff_dim=MODEL_ARGS["ff_dim"],
    )

    def next(prompt, state, index):
        output = model(prompt)
        return output[:, index, :], state

    print("*************************************\n")

    with open(csv_path, "w") as res_handler:
        res_handler.write("decoding_strategy,execution_method,time\n")
        for test_run in TEST_RUNS:
            sampler = test_run["sampler"]

            for execution_method in test_run["execution_methods"]:
                print(f"Running {sampler} in {execution_method} mode")

                if execution_method == "graph":
                    jit_compile = False
                elif execution_method == "xla":
                    jit_compile = True

                time_taken = generate_text(
                    sampler=sampler,
                    next=next,
                    prompt=ds,
                    jit_compile=jit_compile,
                )
                print("Time taken: ", time_taken)
                res_handler.write(
                    f"{sampler},{execution_method},{time_taken}\n"
                )
                print()
            print("*************************************")

    print(f"Writing results to {csv_path}")


if __name__ == "__main__":
    main()
