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

# from keras_nlp.utils import beam_search
from keras_nlp.utils import greedy_search
from keras_nlp.utils import random_search
from keras_nlp.utils import top_k_search
from keras_nlp.utils import top_p_search

COMMON_ARGS = {
    "vocab_size": 40000,
    "num_samples": 1000,
    "batch_size": 2,
    "max_length": 64,
    "model_max_length": 300,
    "embed_dim": 768,
    "num_layers": 8,
    "num_heads": 8,
    "ff_dim": 3072,
    "seed": 42,
}

TEST_RUNS = [
    {
        "decoding_fn": greedy_search,
        "execution_methods": ["xla", "graph"],
        "args": {"end_token_id": 2, "pad_token_id": 0},
    },
    {
        "decoding_fn": random_search,
        "execution_methods": ["xla", "graph"],
        "args": {
            "seed": COMMON_ARGS["seed"],
            "from_logits": True,
            "end_token_id": 2,
            "pad_token_id": 0,
        },
    },
    {
        "decoding_fn": top_k_search,
        "execution_methods": ["xla", "graph"],
        "args": {
            "k": 5,
            "seed": COMMON_ARGS["seed"],
            "from_logits": True,
            "end_token_id": 2,
            "pad_token_id": 0,
        },
    },
    {
        "decoding_fn": top_p_search,
        "execution_methods": ["xla", "graph"],
        "args": {
            "p": 0.9,
            "seed": COMMON_ARGS["seed"],
            "from_logits": True,
            "end_token_id": 2,
            "pad_token_id": 0,
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
    embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
        vocabulary_size=vocab_size,
        sequence_length=max_length,
        embedding_dim=embed_dim,
        mask_zero=True,
    )
    x = embedding_layer(inputs)
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
    token_probability_fn,
    prompt,
    max_length,
    decoding_fn,
    text_gen_args,
    jit_compile,
):
    class TestModel(tf.keras.Model):
        def call(self, inputs):
            generated = decoding_fn(
                token_probability_fn=token_probability_fn,
                prompt=inputs,
                max_length=max_length,
                **text_gen_args,
            )
            return generated

    test_model = TestModel()
    test_model.compile(jit_compile=jit_compile)

    t0 = time.time()
    _ = test_model.predict(prompt)
    return time.time() - t0


def main():
    keras.utils.set_random_seed(COMMON_ARGS["seed"])

    ds = generate_random_ds(
        vocab_size=COMMON_ARGS["vocab_size"],
        num_samples=COMMON_ARGS["num_samples"],
        batch_size=COMMON_ARGS["batch_size"],
        seed=COMMON_ARGS["seed"],
    )

    model = build_model(
        vocab_size=COMMON_ARGS["vocab_size"],
        max_length=COMMON_ARGS["model_max_length"],
        embed_dim=COMMON_ARGS["embed_dim"],
        num_layers=COMMON_ARGS["num_layers"],
        num_heads=COMMON_ARGS["num_heads"],
        ff_dim=COMMON_ARGS["ff_dim"],
    )

    def token_logits_fn(inputs):
        output = model(inputs)
        return output[:, -1, :]  # return next token logits

    print("*************************************\n")

    with open("./results.csv", "w") as res_handler:
        res_handler.write("decoding_strategy,execution_method,time\n")
        for test_run in TEST_RUNS:
            decoding_fn = test_run["decoding_fn"]
            decoding_strategy = str(decoding_fn)

            for execution_method in test_run["execution_methods"]:
                print(f"Running {decoding_strategy} in {execution_method} mode")

                if execution_method == "graph":
                    jit_compile = False
                elif execution_method == "xla":
                    jit_compile = True

                time_taken = generate_text(
                    token_probability_fn=token_logits_fn,
                    prompt=ds,
                    max_length=COMMON_ARGS["max_length"],
                    decoding_fn=decoding_fn,
                    text_gen_args=test_run["args"],
                    jit_compile=jit_compile,
                )
                print("Time taken: ", time_taken)
                res_handler.write(
                    f"{decoding_strategy},{execution_method}," f"{time_taken}\n"
                )
                print()
            print("*************************************")


if __name__ == "__main__":
    main()
