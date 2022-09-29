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


from collections import defaultdict

checkpoints = {
    "gpt2_base": {
        "model": "Gpt2Base",
        "vocabulary": "webtext",
        "description": (
            "Base size of GPT-2 with 124M parameters. Trained on WebText."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_base/model.h5",
        "weights_hash": "f4ea6e1b214516dd7de452461ee6e16e",
    },
    "gpt2_medium": {
        "model": "Gpt2Medium",
        "vocabulary": "webtext",
        "description": (
            "Medium size of GPT-2 with 355M parameters. Trained on WebText."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_medium/model.h5",
        "weights_hash": "580ff9b79c04fc90e6d6f47e975c5afe",
    },
    "gpt2_large": {
        "model": "Gpt2Large",
        "vocabulary": "webtext",
        "description": (
            "Large size of GPT-2 with 774M parameters. Trained on WebText."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_large/model.h5",
        "weights_hash": "67957cb3dfc9e965960dabe068811e1a",
    },
    "gpt2_extra_large": {
        "model": "Gpt2ExtraLarge",
        "vocabulary": "webtext",
        "description": (
            "Extra Large size of GPT-2 with 1558M parameters. "
            "Trained on WebText."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_extra_large/model.h5",
        "weights_hash": "d093c1ee0d9705d845c0190909aa2917",
    },
}


# TODO: Iron out this part after BPE tokenizer has been finalized. Also, check
# the to-do comment in `keras_nlp/models/bert.py`.
vocabularies = {
    "webtext": {
        "description": (
            "The BPE vocabulary for GPT-2 models trained on "
            "the WebText dataset."
        ),
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_base/vocab.json",
        "vocabulary_hash": "dffec25a898b1f5e569bec4dffd7e5c0",
        "vocabulary_size": 50257,
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_base/merges.txt",
        "merges_hash": "75a37753dd7a28a2c5df80c28bf06e4e",
    },
}

# Index checkpoints by arch compatibility.
checkpoints_per_arch = defaultdict(set)
for arch, metadata in checkpoints.items():
    checkpoints_per_arch[metadata["model"]].add(arch)


def compatible_checkpoints(arch):
    """Returns a list of compatible checkpoints per arch"""
    return checkpoints_per_arch[arch]
