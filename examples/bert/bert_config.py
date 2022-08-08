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

MODEL_CONFIGS = {
    "tiny": {
        "num_layers": 2,
        "hidden_size": 128,
        "dropout": 0.1,
        "num_attention_heads": 2,
        "inner_size": 512,
        "inner_activation": "gelu",
        "initializer_range": 0.02,
    },
    "mini": {
        "num_layers": 4,
        "hidden_size": 256,
        "dropout": 0.1,
        "num_attention_heads": 4,
        "inner_size": 1024,
        "inner_activation": "gelu",
        "initializer_range": 0.02,
    },
    "small": {
        "num_layers": 4,
        "hidden_size": 512,
        "dropout": 0.1,
        "num_attention_heads": 8,
        "inner_size": 2048,
        "inner_activation": "gelu",
        "initializer_range": 0.02,
    },
    "medium": {
        "num_layers": 8,
        "hidden_size": 512,
        "dropout": 0.1,
        "num_attention_heads": 8,
        "inner_size": 2048,
        "inner_activation": "gelu",
        "initializer_range": 0.02,
    },
    "base": {
        "num_layers": 12,
        "hidden_size": 768,
        "dropout": 0.1,
        "num_attention_heads": 12,
        "inner_size": 3072,
        "inner_activation": "gelu",
        "initializer_range": 0.02,
    },
    "large": {
        "num_layers": 24,
        "hidden_size": 1024,
        "dropout": 0.1,
        "num_attention_heads": 16,
        "inner_size": 4096,
        "inner_activation": "gelu",
        "initializer_range": 0.02,
    },
}

# Currently we have the same set of training parameters for all configurations.
# We should see if we need to split this for different architecture sizes.

PREPROCESSING_CONFIG = {
    "max_seq_length": 512,
    "max_predictions_per_seq": 76,
    "dupe_factor": 10,
    "masked_lm_prob": 0.15,
    "short_seq_prob": 0.1,
}

TRAINING_CONFIG = {
    "batch_size": 256,
    "epochs": 10,
    "learning_rate": 1e-4,
    "num_train_steps": 1_000_000,
    # Percentage of training steps used for learning rate warmup.
    "warmup_percentage": 0.1,
}

FINETUNING_CONFIG = {
    "batch_size": 32,
    "epochs": 3,
    "learning_rates": [5e-5, 4e-5, 3e-5, 2e-5],
}
