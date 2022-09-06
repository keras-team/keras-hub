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
        "name": "greedy_search",
        "execution_methods": ["xla", "graph"],
        "args": {"end_token_id": 2, "pad_token_id": 0},
    },
    {
        "name": "random_search",
        "execution_methods": ["xla", "graph"],
        "args": {
            "seed": COMMON_ARGS["seed"],
            "from_logits": True,
            "end_token_id": 2,
            "pad_token_id": 0,
        },
    },
    {
        "name": "top_k_search",
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
        "name": "top_p_search",
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
