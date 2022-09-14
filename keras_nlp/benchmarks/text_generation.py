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

"""Benchmark for text generation."""

import time

import tensorflow as tf
from tensorflow import keras

import keras_nlp
from keras_nlp.utils import beam_search
from keras_nlp.utils import greedy_search
from keras_nlp.utils import random_search
from keras_nlp.utils import top_k_search
from keras_nlp.utils import top_p_search

SEED = 42

DATASET_ARGS = {
    "vocab_size": 40000,
    "num_samples": 1000,
    "batch_size": 2,
}

TEXT_GEN_ARGS = {
    "max_length": 64,
    "end_token_id": 2,
    "pad_token_id": 0,
}

MODEL_ARGS = {
    "max_length": 300,
    "embed_dim": 768,
    "num_layers": 8,
    "num_heads": 8,
    "ff_dim": 3072,
}

TEST_RUNS = [
    {
        "decoding_fn": greedy_search,
        "execution_methods": ["xla", "graph"],
        "args": TEXT_GEN_ARGS,
    },
    {
        "decoding_fn": beam_search,
        "execution_methods": ["xla", "graph"],
        "args": {
            "num_beams": 2,
            "from_logits": True,
            **TEXT_GEN_ARGS,
        },
    },
    {
        "decoding_fn": random_search,
        "execution_methods": ["xla", "graph"],
        "args": {
            "seed": SEED,
            "from_logits": True,
            **TEXT_GEN_ARGS,
        },
    },
    {
        "decoding_fn": top_k_search,
        "execution_methods": ["xla", "graph"],
        "args": {
            "k": 5,
            "seed": SEED,
            "from_logits": True,
            **TEXT_GEN_ARGS,
        },
    },
    {
        "decoding_fn": top_p_search,
        "execution_methods": ["xla", "graph"],
        "args": {
            "p": 0.9,
            "seed": SEED,
            "from_logits": True,
            **TEXT_GEN_ARGS,
        },
    },
]


def generate_random_ds(vocab_size, num_samples, batch_size, seed):
    prompt_length = 2
    inputs = tf.random.uniform(
        shape=(num_samples, prompt_length),
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
    inputs = keras.layers.Input(shape=(None,), dtype=tf.int32)
    # Embedding.
    x = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=vocab_size,
        sequence_length=max_length,
        embedding_dim=embed_dim,
        mask_zero=True,
    )(inputs)
    # Transformer decoders.
    for _ in range(num_layers):
        x = keras_nlp.layers.TransformerDecoder(
            num_heads=num_heads,
            intermediate_dim=ff_dim,
        )(x)
    # Output.
    outputs = keras.layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def generate_text(
    decoding_fn,
    token_probability_fn,
    prompt,
    text_gen_args,
    jit_compile,
):
    class TestModel(tf.keras.Model):
        def call(self, inputs):
            generated = decoding_fn(
                token_probability_fn=token_probability_fn,
                prompt=inputs,
                **text_gen_args,
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

    def token_logits_fn(inputs):
        output = model(inputs)
        return output[:, -1, :]

    print("*************************************\n")

    with open(csv_path, "w") as res_handler:
        res_handler.write("decoding_strategy,execution_method,time\n")
        for test_run in TEST_RUNS:
            decoding_fn = test_run["decoding_fn"]
            decoding_strategy = decoding_fn.__name__

            for execution_method in test_run["execution_methods"]:
                print(f"Running {decoding_strategy} in {execution_method} mode")

                if execution_method == "graph":
                    jit_compile = False
                elif execution_method == "xla":
                    jit_compile = True

                time_taken = generate_text(
                    decoding_fn=decoding_fn,
                    token_probability_fn=token_logits_fn,
                    prompt=ds,
                    text_gen_args=test_run["args"],
                    jit_compile=jit_compile,
                )
                print("Time taken: ", time_taken)
                res_handler.write(
                    f"{decoding_strategy},{execution_method}," f"{time_taken}\n"
                )
                print()
            print("*************************************")

    print(f"Writing results to {csv_path}")


if __name__ == "__main__":
    main()
