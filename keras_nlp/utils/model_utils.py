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
import json
import os

import kagglehub
import keras

KAGGLE_PREFIX = "kaggle://"


def get_kaggle_handle(preset):
    kaggle_handle = preset.removeprefix(KAGGLE_PREFIX)
    # Auto add "/1" to the path if necessary.
    # TODO: remove the auto version when kaggle no longer requires it.
    num_segments = len(kaggle_handle.split("/"))
    if num_segments == 4:
        kaggle_handle += "/1"
    elif num_segments != 5:
        raise ValueError(
            "Unexpected kaggle preset handle. Kaggle model handles should have "
            "the form kaggle://{org}/{model}/keras/{variant}[/{version}]. For "
            "example, kaggle://keras-nlp/albert/keras/bert_base_en_uncased."
        )
    return kaggle_handle


def get_preset_config(cls, preset):
    if preset.startswith(KAGGLE_PREFIX):
        kaggle_handle = get_kaggle_handle(preset)
        filename = kagglehub.model_download(kaggle_handle, "config.json")
        with open(filename) as config_file:
            config = json.load(config_file)

        # TODO: Remove this and make our config.json format uniform.
        example = next(iter(cls.presets.values()))
        config["weights_url"] = os.path.basename(example["weights_url"])
        config["weights_hash"] = None
        config["vocabulary_url"] = os.path.basename(example["vocabulary_url"])
        config["vocabulary_hash"] = None
        if "merges_url" in example:
            config["merges_url"] = os.path.basename(example["merges_url"])
            config["merges_hash"] = None

        return config

    if not cls.presets:
        raise NotImplementedError(
            "No presets have been created for this class."
        )

    if preset not in cls.presets:
        raise ValueError(
            "`preset` must be one of "
            f"""{", ".join(cls.presets)}. Received: {preset}."""
        )

    return cls.presets[preset]


def get_file(preset, path, hash):
    if preset.startswith(KAGGLE_PREFIX):
        kaggle_handle = get_kaggle_handle(preset)
        return kagglehub.model_download(kaggle_handle, path)

    return keras.utils.get_file(
        os.path.basename(path),
        path,
        cache_subdir=os.path.join("models", preset),
        file_hash=hash,
    )
