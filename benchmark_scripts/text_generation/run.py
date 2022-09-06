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

import argparse
import json
import time

import tensorflow as tf
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


def main(config):
    args = config["common_args"]
    keras.utils.set_random_seed(args["seed"])

    ds = generate_random_ds(
        vocab_size=args["vocab_size"],
        num_samples=args["num_samples"],
        batch_size=args["batch_size"],
        seed=args["seed"],
    )

    model = build_model(
        vocab_size=args["vocab_size"],
        max_length=args["model_max_length"],
        embed_dim=args["embed_dim"],
        num_layers=args["num_layers"],
        num_heads=args["num_heads"],
        ff_dim=args["ff_dim"],
    )

    def token_logits_fn(inputs):
        output = model(inputs)
        return output[:, -1, :]  # return next token logits

    print("*************************************\n")

    with open("./results.csv", "w") as res_handler:
        res_handler.write("text_gen_method,execution_method,time\n")
        for test_run in config["test_runs"]:
            text_gen_method = test_run["name"]
            if text_gen_method not in SUPPORTED_TEXT_GEN_METHODS:
                raise Exception(
                    f"Unsupported text generation method: {text_gen_method}"
                    f"Should be one of {SUPPORTED_TEXT_GEN_METHODS.keys()}"
                )
            for execution_method in test_run["execution_methods"]:
                print(f"Running {text_gen_method} in {execution_method} mode")
                if execution_method == "graph":
                    jit_compile = False
                elif execution_method == "xla":
                    jit_compile = True
                else:
                    raise Exception(
                        "Unsupported execution method: "
                        f"{execution_method}. Should be one of "
                        f"{EXECUTION_METHODS}."
                    )
                time_taken = run_graph(
                    token_probability_fn=token_logits_fn,
                    prompt=ds,
                    max_length=args["max_length"],
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

    print(json.dumps(config, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="./benchmark_scripts/text_generation/config.json",
        help="Config file path.",
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = json.load(f)

    main(config)
