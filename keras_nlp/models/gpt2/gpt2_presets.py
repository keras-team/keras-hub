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
"""GPT-2 model preset configurations."""

# Metadata for loading pretrained model weights.
backbone_presets = {
    "gpt2_base": {
        "config": {
            "vocabulary_size": 50257,
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.1,
            "max_sequence_length": 1024,
        },
        "preprocessor_config": {},
        "description": (
            "Base size of GPT-2 with 124M parameters. Trained on WebText."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_base/v1/model.h5",
        "weights_hash": "f4ea6e1b214516dd7de452461ee6e16e",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_base/v1/vocab.json",
        "vocabulary_hash": "dffec25a898b1f5e569bec4dffd7e5c0",
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_base/v1/merges.txt",
        "merges_hash": "75a37753dd7a28a2c5df80c28bf06e4e",
    },
    "gpt2_medium": {
        "config": {
            "vocabulary_size": 50257,
            "num_layers": 24,
            "num_heads": 16,
            "hidden_dim": 1024,
            "intermediate_dim": 4096,
            "dropout": 0.1,
            "max_sequence_length": 1024,
        },
        "preprocessor_config": {},
        "description": (
            "Medium size of GPT-2 with 355M parameters. Trained on WebText."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_medium/v1/model.h5",
        "weights_hash": "580ff9b79c04fc90e6d6f47e975c5afe",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_medium/v1/vocab.json",
        "vocabulary_hash": "dffec25a898b1f5e569bec4dffd7e5c0",
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_medium/v1/merges.txt",
        "merges_hash": "75a37753dd7a28a2c5df80c28bf06e4e",
    },
    "gpt2_large": {
        "config": {
            "vocabulary_size": 50257,
            "num_layers": 36,
            "num_heads": 20,
            "hidden_dim": 1280,
            "intermediate_dim": 5120,
            "dropout": 0.1,
            "max_sequence_length": 1024,
        },
        "preprocessor_config": {},
        "description": (
            "Large size of GPT-2 with 774M parameters. Trained on WebText."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_large/v1/model.h5",
        "weights_hash": "67957cb3dfc9e965960dabe068811e1a",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_large/v1/vocab.json",
        "vocabulary_hash": "dffec25a898b1f5e569bec4dffd7e5c0",
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_large/v1/merges.txt",
        "merges_hash": "75a37753dd7a28a2c5df80c28bf06e4e",
    },
    "gpt2_extra_large": {
        "config": {
            "vocabulary_size": 50257,
            "num_layers": 48,
            "num_heads": 25,
            "hidden_dim": 1600,
            "intermediate_dim": 6400,
            "dropout": 0.1,
            "max_sequence_length": 1024,
        },
        "preprocessor_config": {},
        "description": (
            "Extra large size of GPT-2 with 1558M parameters. "
            "Trained on WebText."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_extra_large/v1/model.h5",
        "weights_hash": "d093c1ee0d9705d845c0190909aa2917",
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_extra_large/v1/vocab.json",
        "vocabulary_hash": "dffec25a898b1f5e569bec4dffd7e5c0",
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_extra_large/v1/merges.txt",
        "merges_hash": "75a37753dd7a28a2c5df80c28bf06e4e",
    },
}
