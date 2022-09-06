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
from benchmarks.text_generation_config import COMMON_ARGS
from benchmarks.text_generation_config import TEST_RUNS
from tensorflow import keras

import keras_nlp
from keras_nlp.utils import beam_search
from keras_nlp.utils import greedy_search
from keras_nlp.utils import random_search
from keras_nlp.utils import top_k_search
from keras_nlp.utils import top_p_search

SUPPORTED_TEXT_GEN_METHODS = {
    "greedy_search": greedy_search,
    "random_search": random_search,
    "beam_search": beam_search,
    "top_k_search": top_k_search,
    "top_p_search": top_p_search,
}
EXECUTION_METHODS = ["graph", "xla"]


def generate_random_ds(vocab_size, num_samples, batch_size, seed):
    inputs = tf.random.uniform(
        shape=(num_samples, 2),
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
        decoder_layer = keras_nlp.layers.TransformerDecoder(
            num_heads=num_heads,
            intermediate_dim=ff_dim,
        )
        x = decoder_layer(x)  # Giving one argument only skips cross-attention.
    # Output.
    outputs = keras.layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def run_graph(
    token_probability_fn,
    prompt,
    max_length,
    text_gen_method,
    text_gen_args,
    jit_compile,
):
    text_gen_fn = SUPPORTED_TEXT_GEN_METHODS[text_gen_method]

    class TestModel(tf.keras.Model):
        def call(self, inputs):
            generated = text_gen_fn(
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
        res_handler.write("text_gen_method,execution_method,time\n")
        for test_run in TEST_RUNS:
            text_gen_method = test_run["name"]
            if text_gen_method not in SUPPORTED_TEXT_GEN_METHODS:
                raise Exception(
                    f"Unsupported text generation method: {text_gen_method}"
                    f"Should be one of {SUPPORTED_TEXT_GEN_METHODS.keys()}"
                )
            for execution_method in test_run["execution_methods"]:
                print(f"Running {text_gen_method} in {execution_method} mode")
                if execution_method not in EXECUTION_METHODS:
                    raise Exception(
                        f"Unsupported execution method: {execution_method}"
                        f"Should be one of {EXECUTION_METHODS}"
                    )
                if execution_method == "graph":
                    jit_compile = False
                elif execution_method == "xla":
                    jit_compile = True

                time_taken = run_graph(
                    token_probability_fn=token_logits_fn,
                    prompt=ds,
                    max_length=COMMON_ARGS["max_length"],
                    text_gen_method=text_gen_method,
                    text_gen_args=test_run["args"],
                    jit_compile=jit_compile,
                )
                print("Time taken: ", time_taken)
                res_handler.write(
                    f"{text_gen_method},{execution_method}," f"{time_taken}\n"
                )
                print()
            print("*************************************")


if __name__ == "__main__":
    main()
