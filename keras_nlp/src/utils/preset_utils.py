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

import collections
import datetime
import inspect
import json
import os
import re

from absl import logging
from packaging.version import parse

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.backend import config as backend_config
from keras_nlp.src.backend import keras

try:
    import kagglehub
    from kagglehub.exceptions import KaggleApiHTTPError
except ImportError:
    kagglehub = None

try:
    import huggingface_hub
    from huggingface_hub.utils import EntryNotFoundError
    from huggingface_hub.utils import HFValidationError
except ImportError:
    huggingface_hub = None

KAGGLE_PREFIX = "kaggle://"
GS_PREFIX = "gs://"
HF_PREFIX = "hf://"

TOKENIZER_ASSET_DIR = "assets/tokenizer"

# Config file names.
CONFIG_FILE = "config.json"
TOKENIZER_CONFIG_FILE = "tokenizer.json"
TASK_CONFIG_FILE = "task.json"
PREPROCESSOR_CONFIG_FILE = "preprocessor.json"
METADATA_FILE = "metadata.json"

README_FILE = "README.md"

# Weight file names.
MODEL_WEIGHTS_FILE = "model.weights.h5"
TASK_WEIGHTS_FILE = "task.weights.h5"

# Global state for preset registry.
BUILTIN_PRESETS = {}
BUILTIN_PRESETS_FOR_CLASS = collections.defaultdict(dict)


def register_presets(presets, classes):
    """Register built-in presets for a set of classes.

    Note that this is intended only for models and presets shipped in the
    library itself.
    """
    for preset in presets:
        BUILTIN_PRESETS[preset] = presets[preset]
        for cls in classes:
            BUILTIN_PRESETS_FOR_CLASS[cls][preset] = presets[preset]


def list_presets(cls):
    """Find all registered built-in presets for a class."""
    return dict(BUILTIN_PRESETS_FOR_CLASS[cls])


def list_subclasses(cls):
    """Find all registered subclasses of a class."""
    custom_objects = keras.saving.get_custom_objects().values()
    subclasses = []
    for x in custom_objects:
        if inspect.isclass(x) and x != cls and issubclass(x, cls):
            subclasses.append(x)
    return subclasses


def get_file(preset, path):
    """Download a preset file in necessary and return the local path."""
    # TODO: Add tests for FileNotFound exceptions.
    if not isinstance(preset, str):
        raise ValueError(
            f"A preset identifier must be a string. Received: preset={preset}"
        )
    if preset in BUILTIN_PRESETS:
        preset = BUILTIN_PRESETS[preset]["kaggle_handle"]
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
        try:
            return kagglehub.model_download(kaggle_handle, path)
        except KaggleApiHTTPError as e:
            message = str(e)
            if message.find("403 Client Error"):
                raise FileNotFoundError(
                    f"`{path}` doesn't exist in preset directory `{preset}`."
                )
            else:
                raise ValueError(message)
        except ValueError as e:
            message = str(e)
            if message.find("is not present in the model files"):
                raise FileNotFoundError(
                    f"`{path}` doesn't exist in preset directory `{preset}`."
                )
            else:
                raise ValueError(message)

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
        except EntryNotFoundError as e:
            message = str(e)
            if message.find("403 Client Error"):
                raise FileNotFoundError(
                    f"`{path}` doesn't exist in preset directory `{preset}`."
                )
            else:
                raise ValueError(message)
    elif os.path.exists(preset):
        # Assume a local filepath.
        local_path = os.path.join(preset, path)
        if not os.path.exists(local_path):
            raise FileNotFoundError(
                f"`{path}` doesn't exist in preset directory `{preset}`."
            )
        return local_path
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


def check_file_exists(preset, path):
    try:
        get_file(preset, path)
    except FileNotFoundError:
        return False
    return True


def get_tokenizer(layer):
    """Get the tokenizer from any KerasNLP model or layer."""
    # Avoid circular import.
    from keras_nlp.src.tokenizers.tokenizer import Tokenizer

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


def check_keras_3():
    if not backend_config.keras_3():
        raise ValueError(
            "`save_to_preset` requires Keras 3. Run `pip install -U keras` "
            "upgrade your Keras version, or see https://keras.io/getting_started/ "
            "for more info on Keras versions and installation."
        )


def make_preset_dir(preset):
    os.makedirs(preset, exist_ok=True)


def save_tokenizer_assets(tokenizer, preset):
    if tokenizer:
        asset_dir = os.path.join(preset, TOKENIZER_ASSET_DIR)
        os.makedirs(asset_dir, exist_ok=True)
        tokenizer.save_assets(asset_dir)


def save_serialized_object(
    layer,
    preset,
    config_file=CONFIG_FILE,
    config_to_skip=[],
):
    check_keras_3()
    make_preset_dir(preset)
    config_path = os.path.join(preset, config_file)
    config = keras.saving.serialize_keras_object(layer)
    config_to_skip += ["compile_config", "build_config"]
    for c in config_to_skip:
        recursive_pop(config, c)
    with open(config_path, "w") as config_file:
        config_file.write(json.dumps(config, indent=4))


def save_metadata(layer, preset):
    from keras_nlp.src import __version__ as keras_nlp_version

    keras_version = keras.version() if hasattr(keras, "version") else None
    metadata = {
        "keras_version": keras_version,
        "keras_nlp_version": keras_nlp_version,
        "parameter_count": layer.count_params(),
        "date_saved": datetime.datetime.now().strftime("%Y-%m-%d@%H:%M:%S"),
    }
    metadata_path = os.path.join(preset, METADATA_FILE)
    with open(metadata_path, "w") as metadata_file:
        metadata_file.write(json.dumps(metadata, indent=4))


def _validate_tokenizer(preset, allow_incomplete=False):
    if not check_file_exists(preset, TOKENIZER_CONFIG_FILE):
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
    config_path = get_file(preset, TOKENIZER_CONFIG_FILE)
    try:
        with open(config_path) as config_file:
            config = json.load(config_file)
    except Exception as e:
        raise ValueError(
            f"Tokenizer config file `{config_path}` is an invalid json file. "
            f"Error message: {e}"
        )
    layer = keras.saving.deserialize_keras_object(config)

    for asset in layer.file_assets:
        asset_path = get_file(preset, os.path.join(TOKENIZER_ASSET_DIR, asset))
        if not os.path.exists(asset_path):
            tokenizer_asset_dir = os.path.dirname(asset_path)
            raise FileNotFoundError(
                f"Asset `{asset}` doesn't exist in the tokenizer asset direcotry"
                f" `{tokenizer_asset_dir}`."
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
            json.load(config_file)
    except Exception as e:
        raise ValueError(
            f"Config file `{config_path}` is an invalid json file. "
            f"Error message: {e}"
        )

    weights_path = os.path.join(preset, MODEL_WEIGHTS_FILE)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"The weights file is missing from the preset directory `{preset}`."
        )


def get_snake_case(name):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def create_model_card(preset):
    model_card_path = os.path.join(preset, README_FILE)
    markdown_content = ""

    config = load_config(preset, CONFIG_FILE)
    model_name = (
        config["class_name"].replace("Backbone", "")
        if config["class_name"].endswith("Backbone")
        else config["class_name"]
    )

    task_type = None
    if check_file_exists(preset, TASK_CONFIG_FILE):
        task_config = load_config(preset, TASK_CONFIG_FILE)
        task_type = (
            task_config["class_name"].replace(model_name, "")
            if task_config["class_name"].startswith(model_name)
            else task_config["class_name"]
        )

    # YAML
    markdown_content += "---\n"
    markdown_content += "library_name: keras-nlp\n"
    if task_type == "CausalLM":
        markdown_content += "pipeline_tag: text-generation\n"
    elif task_type == "Classifier":
        markdown_content += "pipeline_tag: text-classification\n"
    markdown_content += "---\n"

    model_link = (
        f"https://keras.io/api/keras_nlp/models/{get_snake_case(model_name)}"
    )
    markdown_content += (
        f"This is a [`{model_name}` model]({model_link}) "
        "uploaded using the KerasNLP library and can be used with JAX, "
        "TensorFlow, and PyTorch backends.\n"
    )
    if task_type:
        markdown_content += (
            f"This model is related to a `{task_type}` task.\n\n"
        )

    backbone_config = config["config"]
    markdown_content += "Model config:\n"
    for k, v in backbone_config.items():
        markdown_content += f"* **{k}:** {v}\n"
    markdown_content += "\n"
    markdown_content += (
        "This model card has been generated automatically and should be completed "
        "by the model author. See [Model Cards documentation]"
        "(https://huggingface.co/docs/hub/model-cards) for more information.\n"
    )

    with open(model_card_path, "w") as md_file:
        md_file.write(markdown_content)


def delete_model_card(preset):
    model_card_path = os.path.join(preset, README_FILE)
    try:
        os.remove(model_card_path)
    except FileNotFoundError:
        logging.warning(
            f"There was an attempt to delete file `{model_card_path}` but this"
            " file doesn't exist."
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
        if kagglehub is None:
            raise ImportError(
                "Uploading a model to Kaggle Hub requires the `kagglehub` package. "
                "Please install with `pip install kagglehub`."
            )
        if parse(kagglehub.__version__) < parse("0.2.4"):
            raise ImportError(
                "Uploading a model to Kaggle Hub requires the `kagglehub` package version `0.2.4` or higher. "
                "Please upgrade with `pip install --upgrade kagglehub`."
            )
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
        has_model_card = huggingface_hub.file_exists(
            repo_id=repo_url.repo_id, filename=README_FILE
        )
        if not has_model_card:
            # Remote repo doesn't have a model card so a basic model card is automatically generated.
            create_model_card(preset)
        try:
            huggingface_hub.upload_folder(
                repo_id=repo_url.repo_id, folder_path=preset
            )
        finally:
            if not has_model_card:
                # Clean up the preset directory in case user attempts to upload the
                # preset directory into Kaggle hub as well.
                delete_model_card(preset)
    else:
        raise ValueError(
            "Unknown URI. An URI must be a one of:\n"
            "1) a Kaggle Model handle like `'kaggle://<KAGGLE_USERNAME>/<MODEL>/<FRAMEWORK>/<VARIATION>'`\n"
            "2) a Hugging Face handle like `'hf://[<HF_USERNAME>/]<MODEL>'`\n"
            f"Received: uri='{uri}'."
        )


def load_config(preset, config_file=CONFIG_FILE):
    config_path = get_file(preset, config_file)
    with open(config_path) as config_file:
        config = json.load(config_file)
    return config


def validate_metadata(preset):
    if not check_file_exists(preset, METADATA_FILE):
        raise FileNotFoundError(
            f"The preset directory `{preset}` doesn't have a file named `{METADATA_FILE}`. "
            "This file is required to load a Keras model preset. Please verify "
            "that the model you are trying to load is a Keras model."
        )
    metadata = load_config(preset, METADATA_FILE)
    if "keras_version" not in metadata:
        raise ValueError(
            f"`{METADATA_FILE}` in the preset directory `{preset}` doesn't have `keras_version`. "
            "Please verify that the model you are trying to load is a Keras model."
        )


def load_serialized_object(
    preset,
    config_file=CONFIG_FILE,
    config_overrides={},
):
    config = load_config(preset, config_file)
    config["config"] = {**config["config"], **config_overrides}
    return keras.saving.deserialize_keras_object(config)


def check_config_class(
    preset,
    config_file=CONFIG_FILE,
):
    """Validate a preset is being loaded on the correct class."""
    config_path = get_file(preset, config_file)
    with open(config_path) as config_file:
        config = json.load(config_file)
    return keras.saving.get_registered_object(config["registered_name"])


def jax_memory_cleanup(layer):
    # For jax, delete all previous allocated memory to avoid temporarily
    # duplicating variable allocations. torch and tensorflow have stateful
    # variable types and do not need this fix.
    if backend_config.backend() == "jax":
        for weight in layer.weights:
            if getattr(weight, "_value", None) is not None:
                weight._value.delete()
