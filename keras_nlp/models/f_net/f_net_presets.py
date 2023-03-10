# Copyright 2023 The KerasNLP Authors
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
"""FNet model preset configurations."""

backbone_presets = {
    "f_net_base_en": {
        "metadata": {
            "description": (
                "12-layer FNet model where case is maintained. "
                "Trained on the C4 dataset."
            ),
            "params": 82861056,
            "official_name": "FNet",
            "path": "f_net",
            "model_card": "https://github.com/google-research/google-research/blob/master/f_net/README.md",
        },
        "config": {
            "vocabulary_size": 32000,
            "num_layers": 12,
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "dropout": 0.1,
            "max_sequence_length": 512,
            "num_segments": 4,
        },
        "preprocessor_config": {},
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/f_net_base_en/v1/model.h5",
        "weights_hash": "35db90842b85a985a0e54c86c00746fe",
        "spm_proto_url": "https://storage.googleapis.com/keras-nlp/models/f_net_base_en/v1/vocab.spm",
        "spm_proto_hash": "71c5f4610bef1daf116998a113a01f3d",
    },
    "f_net_large_en": {
        "metadata": {
            "description": (
                "24-layer FNet model where case is maintained. "
                "Trained on the C4 dataset."
            ),
            "params": 236945408,
            "official_name": "FNet",
            "path": "f_net",
            "model_card": "https://github.com/google-research/google-research/blob/master/f_net/README.md",
        },
        "config": {
            "vocabulary_size": 32000,
            "num_layers": 24,
            "hidden_dim": 1024,
            "intermediate_dim": 4096,
            "dropout": 0.1,
            "max_sequence_length": 512,
            "num_segments": 4,
        },
        "preprocessor_config": {},
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/f_net_large_en/v1/model.h5",
        "weights_hash": "7ae4a3faa67ff054f8cecffb5619f779",
        "spm_proto_url": "https://storage.googleapis.com/keras-nlp/models/f_net_large_en/v1/vocab.spm",
        "spm_proto_hash": "71c5f4610bef1daf116998a113a01f3d",
    },
}
