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

import datetime
import json
import os

from absl import logging

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.backend import config as backend_config
from keras_nlp.backend import keras

try:
    import kagglehub
except ImportError:
    kagglehub = None

try:
    import huggingface_hub
    from huggingface_hub.utils import HFValidationError
except ImportError:
    huggingface_hub = None

KAGGLE_PREFIX = "kaggle://"
GS_PREFIX = "gs://"
HF_PREFIX = "hf://"

TOKENIZER_ASSET_DIR = "assets/tokenizer"
CONFIG_FILE = "config.json"
TOKENIZER_CONFIG_FILE = "tokenizer.json"


def get_file(preset, path):
    """Download a preset file in necessary and return the local path."""
    if not isinstance(preset, str):
        raise ValueError(
            f"A preset identifier must be a string. Received: preset={preset}"
        )
    if preset.startswith(KAGGLE_PREFIX):
        if kagglehub is None:
            raise ImportError(
                "`from_preset()` requires the `kagglehub` package. "
                "Please install with `pip install kagglehub`."
            )
        kaggle_handle = preset.removeprefix(KAGGLE_PREFIX)
        num_segments = len(kaggle_handle.split("/"))
        if num_segments not in (4, 5):
            raise ValueError(
                "Unexpected Kaggle preset. Kaggle model handles should have "
                "the form kaggle://{org}/{model}/keras/{variant}[/{version}]. "
                "For example, 'kaggle://username/bert/keras/bert_base_en' or "
                "'kaggle://username/bert/keras/bert_base_en/1' (to specify a "
                f"version). Received: preset={preset}"
            )
        return kagglehub.model_download(kaggle_handle, path)
    elif preset.startswith(GS_PREFIX):
        url = os.path.join(preset, path)
        url = url.replace(GS_PREFIX, "https://storage.googleapis.com/")
        subdir = preset.replace(GS_PREFIX, "gs_")
        subdir = subdir.replace("/", "_").replace("-", "_")
        filename = os.path.basename(path)
        subdir = os.path.join(subdir, os.path.dirname(path))
        return keras.utils.get_file(
            filename,
            url,
            cache_subdir=os.path.join("models", subdir),
        )
    elif preset.startswith(HF_PREFIX):
        if huggingface_hub is None:
            raise ImportError(
                f"`from_preset()` requires the `huggingface_hub` package to load from '{preset}'. "
                "Please install with `pip install huggingface_hub`."
            )
        hf_handle = preset.removeprefix(HF_PREFIX)
        try:
            return huggingface_hub.hf_hub_download(
                repo_id=hf_handle, filename=path
            )
        except HFValidationError as e:
            raise ValueError(
                "Unexpected Hugging Face preset. Hugging Face model handles "
                "should have the form 'hf://{org}/{model}'. For example, "
                f"'hf://username/bert_base_en'. Received: preset={preset}."
            ) from e
    elif os.path.exists(preset):
        # Assume a local filepath.
        return os.path.join(preset, path)
    else:
        raise ValueError(
            "Unknown preset identifier. A preset must be a one of:\n"
            "1) a built-in preset identifier like `'bert_base_en'`\n"
            "2) a Kaggle Models handle like `'kaggle://keras/bert/keras/bert_base_en'`\n"
            "3) a Hugging Face handle like `'hf://username/bert_base_en'`\n"
            "4) a path to a local preset directory like `'./bert_base_en`\n"
            "Use `print(cls.presets.keys())` to view all built-in presets for "
            "API symbol `cls`.\n"
            f"Received: preset='{preset}'"
        )


def get_tokenizer(layer):
    """Get the tokenizer from any KerasNLP model or layer."""
    # Avoid circular import.
    from keras_nlp.tokenizers.tokenizer import Tokenizer

    if isinstance(layer, Tokenizer):
        return layer
    if hasattr(layer, "tokenizer"):
        return layer.tokenizer
    if hasattr(layer, "preprocessor"):
        return getattr(layer.preprocessor, "tokenizer", None)
    return None


def recursive_pop(config, key):
    """Remove a key from a nested config object"""
    config.pop(key, None)
    for value in config.values():
        if isinstance(value, dict):
            recursive_pop(value, key)


def save_to_preset(
    layer,
    preset,
    save_weights=True,
    config_filename="config.json",
    weights_filename="model.weights.h5",
):
    """Save a KerasNLP layer to a preset directory."""
    os.makedirs(preset, exist_ok=True)

    # Save tokenizers assets.
    tokenizer = get_tokenizer(layer)
    assets = []
    if tokenizer:
        asset_dir = os.path.join(preset, TOKENIZER_ASSET_DIR)
        os.makedirs(asset_dir, exist_ok=True)
        tokenizer.save_assets(asset_dir)
        for asset_path in os.listdir(asset_dir):
            assets.append(os.path.join(TOKENIZER_ASSET_DIR, asset_path))

    # Optionally save weights.
    save_weights = save_weights and hasattr(layer, "save_weights")
    if save_weights:
        weights_path = os.path.join(preset, weights_filename)
        layer.save_weights(weights_path)

    # Save a serialized Keras object.
    config_path = os.path.join(preset, config_filename)
    config = keras.saving.serialize_keras_object(layer)
    # Include references to weights and assets.
    config["assets"] = assets
    config["weights"] = weights_filename if save_weights else None
    recursive_pop(config, "compile_config")
    recursive_pop(config, "build_config")
    with open(config_path, "w") as config_file:
        config_file.write(json.dumps(config, indent=4))

    from keras_nlp import __version__ as keras_nlp_version

    keras_version = keras.version() if hasattr(keras, "version") else None

    # Save any associated metadata.
    if config_filename == "config.json":
        metadata = {
            "keras_version": keras_version,
            "keras_nlp_version": keras_nlp_version,
            "parameter_count": layer.count_params(),
            "date_saved": datetime.datetime.now().strftime("%Y-%m-%d@%H:%M:%S"),
        }
        metadata_path = os.path.join(preset, "metadata.json")
        with open(metadata_path, "w") as metadata_file:
            metadata_file.write(json.dumps(metadata, indent=4))


def _validate_tokenizer(preset, allow_incomplete=False):
    config_path = get_file(preset, TOKENIZER_CONFIG_FILE)
    if not os.path.exists(config_path):
        if allow_incomplete:
            logging.warning(
                f"`{TOKENIZER_CONFIG_FILE}` is missing from the preset directory `{preset}`."
            )
            return
        else:
            raise FileNotFoundError(
                f"`{TOKENIZER_CONFIG_FILE}` is missing from the preset directory `{preset}`. "
                "To upload the model without a tokenizer, "
                "set `allow_incomplete=True`."
            )
    try:
        with open(config_path) as config_file:
            config = json.load(config_file)
    except Exception as e:
        raise ValueError(
            f"Tokenizer config file `{config_path}` is an invalid json file. "
            f"Error message: {e}"
        )
    layer = keras.saving.deserialize_keras_object(config)

    if not config["assets"]:
        raise ValueError(
            f"Tokenizer config file {config_path} is missing `asset`."
        )

    for asset in config["assets"]:
        asset_path = os.path.join(preset, asset)
        if not os.path.exists(asset_path):
            raise FileNotFoundError(
                f"Asset `{asset}` doesn't exist in the preset direcotry `{preset}`."
            )
    config_dir = os.path.dirname(config_path)
    asset_dir = os.path.join(config_dir, TOKENIZER_ASSET_DIR)

    tokenizer = get_tokenizer(layer)
    if not tokenizer:
        raise ValueError(f"Model or layer `{layer}` is missing tokenizer.")
    tokenizer.load_assets(asset_dir)


def _validate_backbone(preset):
    config_path = os.path.join(preset, CONFIG_FILE)
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"`{CONFIG_FILE}` is missing from the preset directory `{preset}`."
        )
    try:
        with open(config_path) as config_file:
            config = json.load(config_file)
    except Exception as e:
        raise ValueError(
            f"Config file `{config_path}` is an invalid json file. "
            f"Error message: {e}"
        )

    if config["weights"]:
        weights_path = os.path.join(preset, config["weights"])
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"The weights file is missing from the preset directory `{preset}`."
            )
    else:
        raise ValueError(
            f"No weights listed in `{CONFIG_FILE}`. Make sure to use "
            "`save_to_preset()` which adds additional data to a serialized "
            "Keras object."
        )


@keras_nlp_export("keras_nlp.upload_preset")
def upload_preset(
    uri,
    preset,
    allow_incomplete=False,
):
    """Upload a preset directory to a model hub.

    Args:
        uri: The URI identifying model to upload to.
             URIs with format
             `kaggle://<KAGGLE_USERNAME>/<MODEL>/<FRAMEWORK>/<VARIATION>`
             will be uploaded to Kaggle Hub while URIs with format
             `hf://[<HF_USERNAME>/]<MODEL>` will be uploaded to the Hugging
             Face Hub.
        preset: The path to the local model preset directory.
        allow_incomplete: If True, allows the upload of presets without
                          a tokenizer configuration. Otherwise, a tokenizer
                          is required.
    """

    # Check if preset directory exists.
    if not os.path.exists(preset):
        raise FileNotFoundError(f"The preset directory {preset} doesn't exist.")

    _validate_backbone(preset)
    _validate_tokenizer(preset, allow_incomplete)

    if uri.startswith(KAGGLE_PREFIX):
        kaggle_handle = uri.removeprefix(KAGGLE_PREFIX)
        kagglehub.model_upload(kaggle_handle, preset)
    elif uri.startswith(HF_PREFIX):
        if huggingface_hub is None:
            raise ImportError(
                f"`upload_preset()` requires the `huggingface_hub` package to upload to '{uri}'. "
                "Please install with `pip install huggingface_hub`."
            )
        hf_handle = uri.removeprefix(HF_PREFIX)
        try:
            repo_url = huggingface_hub.create_repo(
                repo_id=hf_handle, exist_ok=True
            )
        except HFValidationError as e:
            raise ValueError(
                "Unexpected Hugging Face URI. Hugging Face model handles "
                "should have the form 'hf://[{org}/]{model}'. For example, "
                "'hf://username/bert_base_en' or 'hf://bert_case_en' to implicitly"
                f"upload to your user account. Received: URI={uri}."
            ) from e
        huggingface_hub.upload_folder(
            repo_id=repo_url.repo_id, folder_path=preset
        )
    else:
        raise ValueError(
            "Unknown URI. An URI must be a one of:\n"
            "1) a Kaggle Model handle like `'kaggle://<KAGGLE_USERNAME>/<MODEL>/<FRAMEWORK>/<VARIATION>'`\n"
            "2) a Hugging Face handle like `'hf://[<HF_USERNAME>/]<MODEL>'`\n"
            f"Received: uri='{uri}'."
        )


def load_from_preset(
    preset,
    load_weights=True,
    config_file="config.json",
    config_overrides={},
):
    """Load a KerasNLP layer to a preset directory."""
    # Load a serialized Keras object.
    config_path = get_file(preset, config_file)
    with open(config_path) as config_file:
        config = json.load(config_file)
    config["config"] = {**config["config"], **config_overrides}
    layer = keras.saving.deserialize_keras_object(config)

    # Load any assets for our tokenizers.
    tokenizer = get_tokenizer(layer)
    if tokenizer and config["assets"]:
        for asset in config["assets"]:
            get_file(preset, asset)
        config_dir = os.path.dirname(config_path)
        asset_dir = os.path.join(config_dir, TOKENIZER_ASSET_DIR)
        tokenizer.load_assets(asset_dir)

    # Optionally load weights.
    load_weights = load_weights and config["weights"]
    if load_weights:
        # For jax, delete all previous allocated memory to avoid temporarily
        # duplicating variable allocations. torch and tensorflow have stateful
        # variable types and do not need this fix.
        if backend_config.backend() == "jax":
            for weight in layer.weights:
                if getattr(weight, "_value", None) is not None:
                    weight._value.delete()
        weights_path = get_file(preset, config["weights"])
        layer.load_weights(weights_path)

    return layer


def check_preset_class(
    preset,
    classes,
    config_file="config.json",
):
    """Validate a preset is being loaded on the correct class."""
    config_path = get_file(preset, config_file)
    with open(config_path) as config_file:
        config = json.load(config_file)
    cls = keras.saving.get_registered_object(config["registered_name"])
    if not isinstance(classes, (tuple, list)):
        classes = (classes,)
    # Allow subclasses for testing a base class, e.g.
    # `check_preset_class(preset, Backbone)`
    if not any(issubclass(cls, x) for x in classes):
        raise ValueError(
            f"Unexpected class in preset `'{preset}'`. "
            "When calling `from_preset()` on a class object, the preset class "
            f"much match allowed classes. Allowed classes are `{classes}`. "
            f"Received: `{cls}`."
        )
    return cls
