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

# TODO(jbischof): remove in favor of presets with load_weights=False
MODEL_CONFIGS = {
    "tiny": {
        "num_layers": 2,
        "hidden_dim": 128,
        "dropout": 0.1,
        "num_heads": 2,
        "intermediate_dim": 512,
    },
    "mini": {
        "num_layers": 4,
        "hidden_dim": 256,
        "dropout": 0.1,
        "num_heads": 4,
        "intermediate_dim": 1024,
    },
    "small": {
        "num_layers": 4,
        "hidden_dim": 512,
        "dropout": 0.1,
        "num_heads": 8,
        "intermediate_dim": 2048,
    },
    "medium": {
        "num_layers": 8,
        "hidden_dim": 512,
        "dropout": 0.1,
        "num_heads": 8,
        "intermediate_dim": 2048,
    },
    "base": {
        "num_layers": 12,
        "hidden_dim": 768,
        "dropout": 0.1,
        "num_heads": 12,
        "intermediate_dim": 3072,
    },
    "large": {
        "num_layers": 24,
        "hidden_dim": 1024,
        "dropout": 0.1,
        "num_heads": 16,
        "intermediate_dim": 4096,
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
