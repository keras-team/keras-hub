import collections
import datetime
import inspect
import json
import os
import re

import keras
from absl import logging

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.utils.keras_utils import print_msg

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
MODELSCOPE_PREFIX = "modelscope://"

KAGGLE_SCHEME = "kaggle"
GS_SCHEME = "gs"
HF_SCHEME = "hf"
MODELSCOPE_SCHEME = "modelscope"
ASSET_DIR = "assets"
TOKENIZER_ASSET_DIR = f"{ASSET_DIR}/tokenizer"

# Config file names.
CONFIG_FILE = "config.json"
TOKENIZER_CONFIG_FILE = "tokenizer.json"
AUDIO_CONVERTER_CONFIG_FILE = "audio_converter.json"
IMAGE_CONVERTER_CONFIG_FILE = "image_converter.json"
TASK_CONFIG_FILE = "task.json"
PREPROCESSOR_CONFIG_FILE = "preprocessor.json"
METADATA_FILE = "metadata.json"

# Weight file names.
MODEL_WEIGHTS_FILE = "model.weights.h5"
TASK_WEIGHTS_FILE = "task.weights.h5"

# HuggingFace filenames.
README_FILE = "README.md"
HF_CONFIG_FILE = "config.json"
HF_TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
SAFETENSOR_CONFIG_FILE = "model.safetensors.index.json"
SAFETENSOR_FILE = "model.safetensors"

# Global state for preset registry.
BUILTIN_PRESETS = {}
BUILTIN_PRESETS_FOR_BACKBONE = collections.defaultdict(dict)


def register_presets(presets, backbone_cls):
    """Register built-in presets for a set of classes.

    Note that this is intended only for models and presets shipped in the
    library itself.
    """
    for preset in presets:
        BUILTIN_PRESETS[preset] = presets[preset]
        BUILTIN_PRESETS_FOR_BACKBONE[backbone_cls][preset] = presets[preset]


def builtin_presets(cls):
    """Find all registered built-in presets for a class."""
    presets = {}
    if cls in BUILTIN_PRESETS_FOR_BACKBONE:
        presets.update(BUILTIN_PRESETS_FOR_BACKBONE[cls])
    backbone_cls = getattr(cls, "backbone_cls", None)
    if backbone_cls:
        presets.update(builtin_presets(backbone_cls))
    for subclass in list_subclasses(cls):
        presets.update(builtin_presets(subclass))
    return presets


def list_subclasses(cls):
    """Find all registered subclasses of a class."""
    # Deduplicate the lists, since we have to register object twice for compat.
    custom_objects = set(keras.saving.get_custom_objects().values())
    subclasses = []
    for x in custom_objects:
        if inspect.isclass(x) and x != cls and issubclass(x, cls):
            subclasses.append(x)
    return subclasses


def find_subclass(preset, cls, backbone_cls):
    """Find a subclass that is compatible with backbone_cls."""
    subclasses = list_subclasses(cls)
    subclasses = filter(lambda x: x.backbone_cls == backbone_cls, subclasses)
    subclasses = list(subclasses)
    if not subclasses:
        raise ValueError(
            f"Unable to find a subclass of {cls.__name__} that is compatible "
            f"with {backbone_cls.__name__} found in preset '{preset}'."
        )
    # If we find multiple subclasses, try to filter to direct subclasses of
    # the class we are trying to instantiate.
    if len(subclasses) > 1:
        directs = list(filter(lambda x: x in cls.__bases__, subclasses))
        if len(directs) > 1:
            subclasses = directs
    # Return the subclass that was registered first (prefer built-in classes).
    return subclasses[0]


def get_file(preset, path):
    """Download a preset file in necessary and return the local path."""
    # TODO: Add tests for FileNotFound exceptions.
    if not isinstance(preset, str):
        raise ValueError(
            f"A preset identifier must be a string. Received: preset={preset}"
        )
    if preset in BUILTIN_PRESETS:
        preset = BUILTIN_PRESETS[preset]["kaggle_handle"]

    scheme = None
    if "://" in preset:
        scheme = preset.split("://")[0].lower()

    if scheme == KAGGLE_SCHEME:
        if kagglehub is None:
            raise ImportError(
                "`from_preset()` requires the `kagglehub` package. "
                "Please install with `pip install kagglehub`."
            )
        kaggle_handle = preset.removeprefix(KAGGLE_SCHEME + "://")
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
    elif scheme in tf_registered_schemes():
        return tf_copy_gfile_to_cache(preset, path)
    elif scheme == MODELSCOPE_SCHEME:
        try:
            from modelscope.hub.snapshot_download import snapshot_download
        except ImportError:
            raise ImportError(
                "To load a preset from ModelScope {preset} using from_preset,"
                "install the modelscope package with: pip install modelscope."
            )
        modelscope_handle = preset.removeprefix(MODELSCOPE_SCHEME + "://")
        try:
            return_path = snapshot_download(modelscope_handle) + "/" + path
            if os.path.exists(return_path):
                return return_path
            raise FileNotFoundError(
                f"`{return_path}` doesn't exist in preset directory `{preset}`."
            )
        except ValueError as e:
            raise ValueError(
                "ModelScope handles should follow the format "
                f"'modelscope://{{org}}/{{model}}'  "
                "(e.g., 'modelscope://username/bert_base_en')."
                f"Received: preset='{preset}.'"
            ) from e
        except EntryNotFoundError as e:
            message = str(e)
            if message.find("403 Client Error"):
                raise FileNotFoundError(
                    f"`{path}` not exist in preset directory `{preset}`."
                )
            else:
                raise ValueError(message)
    elif scheme == HF_SCHEME:
        if huggingface_hub is None:
            raise ImportError(
                "`from_preset()` requires the `huggingface_hub` package to "
                "load from '{preset}'. "
                "Please install with `pip install huggingface_hub`."
            )
        hf_handle = preset.removeprefix(HF_SCHEME + "://")
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
        # Assume a local filepath.pyth
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
            "2) a Kaggle Models handle like "
            "`'kaggle://keras/bert/keras/bert_base_en'`\n"
            "3) a Hugging Face handle like `'hf://username/bert_base_en'`\n"
            "4) a path to a local preset directory like `'./bert_base_en`\n"
            "Use `print(cls.presets.keys())` to view all built-in presets for "
            "API symbol `cls`.\n"
            f"Received: preset='{preset}'"
        )


def tf_registered_schemes():
    try:
        import tensorflow as tf

        return tf.io.gfile.get_registered_schemes()
    except ImportError:
        return []


def tf_copy_gfile_to_cache(preset, path):
    """Much of this is adapted from get_file of keras core."""
    if "KERAS_HOME" in os.environ:
        base_dir = os.environ.get("KERAS_HOME")
    else:
        base_dir = os.path.expanduser(os.path.join("~", ".keras"))
    if not os.access(base_dir, os.W_OK):
        base_dir = os.path.join("/tmp", ".keras")

    url = os.path.join(preset, path)
    model_dir = preset.replace("://", "_").replace("-", "_").replace("/", "_")
    local_path = os.path.join(base_dir, "models", model_dir, path)

    if not os.path.exists(local_path):
        print_msg(f"Downloading data from {url}")
        try:
            import tensorflow as tf

            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            tf.io.gfile.copy(url, local_path)
        except Exception as e:
            # gfile.copy will leave an empty file after an error.
            # Work around this bug.
            os.remove(local_path)
            if isinstance(
                e, tf.errors.PermissionDeniedError, tf.errors.NotFoundError
            ):
                raise FileNotFoundError(
                    f"`{path}` doesn't exist in preset directory `{preset}`.",
                ) from e
            raise e

    return local_path


def check_file_exists(preset, path):
    try:
        get_file(preset, path)
    except FileNotFoundError:
        return False
    return True


def _validate_backbone(preset):
    config_path = os.path.join(preset, CONFIG_FILE)
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"`{CONFIG_FILE}` is missing from the preset directory `{preset}`."
        )
    try:
        with open(config_path, encoding="utf-8") as config_file:
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


def to_snake_case(name):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def create_model_card(preset):
    model_card_path = os.path.join(preset, README_FILE)
    markdown_content = ""

    config = load_json(preset, CONFIG_FILE)
    model_name = (
        config["class_name"].replace("Backbone", "")
        if config["class_name"].endswith("Backbone")
        else config["class_name"]
    )

    task_type = None
    if check_file_exists(preset, TASK_CONFIG_FILE):
        task_config = load_json(preset, TASK_CONFIG_FILE)
        task_type = (
            task_config["class_name"].replace(model_name, "")
            if task_config["class_name"].startswith(model_name)
            else task_config["class_name"]
        )

    # YAML
    markdown_content += "---\n"
    markdown_content += "library_name: keras-hub\n"
    if task_type == "CausalLM":
        markdown_content += "pipeline_tag: text-generation\n"
    elif task_type == "TextClassifier":
        markdown_content += "pipeline_tag: text-classification\n"
    markdown_content += "---\n"

    model_link = (
        f"https://keras.io/api/keras_hub/models/{to_snake_case(model_name)}"
    )
    markdown_content += (
        f"This is a [`{model_name}` model]({model_link}) "
        "uploaded using the KerasHub library and can be used with JAX, "
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
        "This model card has been generated automatically and should be "
        "completed by the model author. See [Model Cards documentation]"
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


@keras_hub_export("keras_hub.upload_preset")
def upload_preset(
    uri,
    preset,
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
    """

    # Check if preset directory exists.
    if not os.path.exists(preset):
        raise FileNotFoundError(f"The preset directory {preset} doesn't exist.")

    _validate_backbone(preset)

    if uri.startswith(KAGGLE_PREFIX):
        if kagglehub is None:
            raise ImportError(
                "Uploading a model to Kaggle Hub requires the `kagglehub` "
                "package. Please install with `pip install kagglehub`."
            )
        kaggle_handle = uri.removeprefix(KAGGLE_PREFIX)
        kagglehub.model_upload(kaggle_handle, preset)
    elif uri.startswith(HF_PREFIX):
        if huggingface_hub is None:
            raise ImportError(
                f"`upload_preset()` requires the `huggingface_hub` package "
                f"to upload to '{uri}'. "
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
                "'hf://username/bert_base_en' or 'hf://bert_case_en' to "
                f"implicitly upload to your user account. Received: URI={uri}."
            ) from e
        has_model_card = huggingface_hub.file_exists(
            repo_id=repo_url.repo_id, filename=README_FILE
        )
        if not has_model_card:
            # Remote repo doesn't have a model card so a basic model card is
            # automatically generated.
            create_model_card(preset)
        try:
            huggingface_hub.upload_folder(
                repo_id=repo_url.repo_id, folder_path=preset
            )
        finally:
            if not has_model_card:
                # Clean up the preset directory in case user attempts to upload
                # the preset directory into Kaggle hub as well.
                delete_model_card(preset)
    else:
        raise ValueError(
            "Unknown URI. An URI must be a one of:\n"
            "1) a Kaggle Model handle like "
            "`'kaggle://<KAGGLE_USERNAME>/<MODEL>/<FRAMEWORK>/<VARIATION>'`\n"
            "2) a Hugging Face handle like `'hf://[<HF_USERNAME>/]<MODEL>'`\n"
            f"Received: uri='{uri}'."
        )


def load_json(preset, config_file=CONFIG_FILE):
    config_path = get_file(preset, config_file)
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    return config


def check_config_class(config):
    """Validate a preset is being loaded on the correct class."""
    registered_name = config["registered_name"]
    if registered_name in ("Functional", "Sequential"):
        return keras.Model
    cls = keras.saving.get_registered_object(registered_name)
    if cls is None:
        raise ValueError(
            f"Attempting to load class {registered_name} with "
            "`from_preset()`, but there is no class registered with Keras "
            f"for {registered_name}. Make sure to register any custom "
            "classes with `register_keras_serializable()`."
        )
    return cls


def jax_memory_cleanup(layer):
    # For jax, delete all previous allocated memory to avoid temporarily
    # duplicating variable allocations. torch and tensorflow have stateful
    # variable types and do not need this fix.
    if keras.config.backend() == "jax":
        for weight in layer.weights:
            if getattr(weight, "_value", None) is not None:
                weight._value.delete()


def set_dtype_in_config(config, dtype=None):
    if dtype is None:
        return config

    config = config.copy()
    if "dtype" not in config["config"]:
        # Forward `dtype` to the config.
        config["config"]["dtype"] = dtype
    elif (
        "dtype" in config["config"]
        and isinstance(config["config"]["dtype"], dict)
        and "DTypePolicyMap" in config["config"]["dtype"]["class_name"]
    ):
        # If it is `DTypePolicyMap` in `config`, forward `dtype` as its default
        # policy.
        policy_map_config = config["config"]["dtype"]["config"]
        policy_map_config["default_policy"] = dtype
        for k in policy_map_config["policy_map"].keys():
            policy_map_config["policy_map"][k]["config"]["source_name"] = dtype
    return config


def get_preset_loader(preset):
    if not check_file_exists(preset, CONFIG_FILE):
        raise ValueError(
            f"Preset {preset} has no {CONFIG_FILE}. Make sure the URI or "
            "directory you are trying to load is a valid KerasHub preset and "
            "and that you have permissions to read/download from this location."
        )
    # We currently assume all formats we support have a `config.json`, this is
    # true, for Keras, Transformers, and timm. We infer the on disk format by
    # inspecting the `config.json` file.
    config = load_json(preset, CONFIG_FILE)
    if "registered_name" in config:
        # If we see registered_name, we assume a serialized Keras object.
        return KerasPresetLoader(preset, config)
    elif "model_type" in config:
        # Avoid circular import.
        from keras_hub.src.utils.transformers.preset_loader import (
            TransformersPresetLoader,
        )

        # If we see model_type, we assume a Transformers style config.
        return TransformersPresetLoader(preset, config)
    elif "architecture" in config:
        # Avoid circular import.
        from keras_hub.src.utils.timm.preset_loader import TimmPresetLoader

        # If we see "architecture", we assume a timm config. We could make this
        # more robust later on if we need to.
        return TimmPresetLoader(preset, config)

    else:
        contents = json.dumps(config, indent=4)
        raise ValueError(
            f"Unrecognized format for {CONFIG_FILE} in {preset}. "
            "Create a preset with the `save_to_preset` utility on KerasHub "
            f"models. Contents of {CONFIG_FILE}:\n{contents}"
        )


def get_preset_saver(preset):
    # Unlike loading, we only support one form of saving; Keras serialized
    # configs and saved weights. We keep the rough API structure as loading
    # just for simplicity.
    return KerasPresetSaver(preset)


class PresetLoader:
    def __init__(self, preset, config):
        self.config = config
        self.preset = preset

    def get_backbone_kwargs(self, **kwargs):
        backbone_kwargs = {}

        # Forward `dtype` to backbone.
        backbone_kwargs["dtype"] = kwargs.pop("dtype", None)

        # Forward `height` and `width` to backbone when using `TextToImage`.
        if "image_shape" in kwargs:
            backbone_kwargs["image_shape"] = kwargs.pop("image_shape", None)

        return backbone_kwargs, kwargs

    def check_backbone_class(self):
        """Infer the backbone architecture."""
        raise NotImplementedError

    def load_backbone(self, cls, load_weights, **kwargs):
        """Load the backbone model from the preset."""
        raise NotImplementedError

    def load_tokenizer(self, cls, config_file=TOKENIZER_CONFIG_FILE, **kwargs):
        """Load a tokenizer layer from the preset."""
        raise NotImplementedError

    def load_audio_converter(self, cls, **kwargs):
        """Load an audio converter layer from the preset."""
        raise NotImplementedError

    def load_image_converter(self, cls, **kwargs):
        """Load an image converter layer from the preset."""
        raise NotImplementedError

    def load_task(self, cls, load_weights, load_task_weights, **kwargs):
        """Load a task model from the preset.

        By default, we create a task from a backbone and preprocessor with
        default arguments. This means
        """
        if "backbone" not in kwargs:
            backbone_class = cls.backbone_cls
            backbone_kwargs, kwargs = self.get_backbone_kwargs(**kwargs)
            kwargs["backbone"] = self.load_backbone(
                backbone_class, load_weights, **backbone_kwargs
            )
        if "preprocessor" not in kwargs and cls.preprocessor_cls:
            kwargs["preprocessor"] = self.load_preprocessor(
                cls.preprocessor_cls,
            )

        return cls(**kwargs)

    def load_preprocessor(
        self, cls, config_file=PREPROCESSOR_CONFIG_FILE, **kwargs
    ):
        """Load a prepocessor layer from the preset.

        By default, we create a preprocessor from a tokenizer with default
        arguments. This allow us to support transformers checkpoints by
        only converting the backbone and tokenizer.
        """
        kwargs = cls._add_missing_kwargs(self, kwargs)
        return cls(**kwargs)


class KerasPresetLoader(PresetLoader):
    def check_backbone_class(self):
        return check_config_class(self.config)

    def load_backbone(self, cls, load_weights, **kwargs):
        backbone = self._load_serialized_object(self.config, **kwargs)
        if load_weights:
            jax_memory_cleanup(backbone)
            backbone.load_weights(get_file(self.preset, MODEL_WEIGHTS_FILE))
        return backbone

    def load_tokenizer(self, cls, config_file=TOKENIZER_CONFIG_FILE, **kwargs):
        tokenizer_config = load_json(self.preset, config_file)
        tokenizer = self._load_serialized_object(tokenizer_config, **kwargs)
        if hasattr(tokenizer, "load_preset_assets"):
            tokenizer.load_preset_assets(self.preset)
        return tokenizer

    def load_audio_converter(self, cls, **kwargs):
        converter_config = load_json(self.preset, AUDIO_CONVERTER_CONFIG_FILE)
        return self._load_serialized_object(converter_config, **kwargs)

    def load_image_converter(self, cls, **kwargs):
        converter_config = load_json(self.preset, IMAGE_CONVERTER_CONFIG_FILE)
        return self._load_serialized_object(converter_config, **kwargs)

    def load_task(self, cls, load_weights, load_task_weights, **kwargs):
        # If there is no `task.json` or it's for the wrong class delegate to the
        # super class loader.
        if not check_file_exists(self.preset, TASK_CONFIG_FILE):
            return super().load_task(
                cls, load_weights, load_task_weights, **kwargs
            )
        task_config = load_json(self.preset, TASK_CONFIG_FILE)
        if not issubclass(check_config_class(task_config), cls):
            return super().load_task(
                cls, load_weights, load_task_weights, **kwargs
            )
        # We found a `task.json` with a complete config for our class.
        # Forward backbone args.
        backbone_kwargs, kwargs = self.get_backbone_kwargs(**kwargs)
        if "backbone" in task_config["config"]:
            backbone_config = task_config["config"]["backbone"]["config"]
            backbone_config = {**backbone_config, **backbone_kwargs}
            task_config["config"]["backbone"]["config"] = backbone_config
        task = self._load_serialized_object(task_config, **kwargs)
        if task.preprocessor and hasattr(
            task.preprocessor, "load_preset_assets"
        ):
            task.preprocessor.load_preset_assets(self.preset)
        if load_weights:
            has_task_weights = check_file_exists(self.preset, TASK_WEIGHTS_FILE)
            if has_task_weights and load_task_weights:
                jax_memory_cleanup(task)
                task_weights = get_file(self.preset, TASK_WEIGHTS_FILE)
                task.load_task_weights(task_weights)
            else:
                jax_memory_cleanup(task.backbone)
            backbone_weights = get_file(self.preset, MODEL_WEIGHTS_FILE)
            task.backbone.load_weights(backbone_weights)
        return task

    def load_preprocessor(
        self, cls, config_file=PREPROCESSOR_CONFIG_FILE, **kwargs
    ):
        # If there is no `preprocessing.json` or it's for the wrong class,
        # delegate to the super class loader.
        if not check_file_exists(self.preset, config_file):
            return super().load_preprocessor(cls, **kwargs)
        preprocessor_json = load_json(self.preset, config_file)
        if not issubclass(check_config_class(preprocessor_json), cls):
            return super().load_preprocessor(cls, **kwargs)
        # We found a `preprocessing.json` with a complete config for our class.
        preprocessor = self._load_serialized_object(preprocessor_json, **kwargs)
        if hasattr(preprocessor, "load_preset_assets"):
            preprocessor.load_preset_assets(self.preset)
        return preprocessor

    def _load_serialized_object(self, config, **kwargs):
        # `dtype` in config might be a serialized `DTypePolicy` or
        # `DTypePolicyMap`. Ensure that `dtype` is properly configured.
        dtype = kwargs.pop("dtype", None)
        config = set_dtype_in_config(config, dtype)

        config["config"] = {**config["config"], **kwargs}
        return keras.saving.deserialize_keras_object(config)


class KerasPresetSaver:
    def __init__(self, preset_dir):
        os.makedirs(preset_dir, exist_ok=True)
        self.preset_dir = preset_dir

    def save_backbone(self, backbone):
        self._save_serialized_object(backbone, config_file=CONFIG_FILE)
        backbone_weight_path = os.path.join(self.preset_dir, MODEL_WEIGHTS_FILE)
        backbone.save_weights(backbone_weight_path)
        self._save_metadata(backbone)

    def save_tokenizer(self, tokenizer):
        config_file = TOKENIZER_CONFIG_FILE
        if hasattr(tokenizer, "config_file"):
            config_file = tokenizer.config_file
        self._save_serialized_object(tokenizer, config_file)
        # Save assets.
        subdir = config_file.split(".")[0]
        asset_dir = os.path.join(self.preset_dir, ASSET_DIR, subdir)
        os.makedirs(asset_dir, exist_ok=True)
        tokenizer.save_assets(asset_dir)

    def save_audio_converter(self, converter):
        self._save_serialized_object(converter, AUDIO_CONVERTER_CONFIG_FILE)

    def save_image_converter(self, converter):
        self._save_serialized_object(converter, IMAGE_CONVERTER_CONFIG_FILE)

    def save_task(self, task):
        # Save task specific config and weights.
        self._save_serialized_object(task, TASK_CONFIG_FILE)
        if task.has_task_weights():
            task_weight_path = os.path.join(self.preset_dir, TASK_WEIGHTS_FILE)
            task.save_task_weights(task_weight_path)
        # Save backbone.
        if hasattr(task.backbone, "save_to_preset"):
            task.backbone.save_to_preset(self.preset_dir)
        else:
            # Allow saving a `keras.Model` that is not a backbone subclass.
            self.save_backbone(task.backbone)
        # Save preprocessor.
        if task.preprocessor and hasattr(task.preprocessor, "save_to_preset"):
            task.preprocessor.save_to_preset(self.preset_dir)
        else:
            # Allow saving a `keras.Layer` that is not a preprocessor subclass.
            self.save_preprocessor(task.preprocessor)

    def save_preprocessor(self, preprocessor):
        config_file = PREPROCESSOR_CONFIG_FILE
        if hasattr(preprocessor, "config_file"):
            config_file = preprocessor.config_file
        self._save_serialized_object(preprocessor, config_file)
        for layer in preprocessor._flatten_layers(include_self=False):
            if hasattr(layer, "save_to_preset"):
                layer.save_to_preset(self.preset_dir)

    def _recursive_pop(self, config, key):
        """Remove a key from a nested config object"""
        config.pop(key, None)
        for value in config.values():
            if isinstance(value, dict):
                self._recursive_pop(value, key)

    def _save_serialized_object(self, layer, config_file):
        config_path = os.path.join(self.preset_dir, config_file)
        config = keras.saving.serialize_keras_object(layer)
        config_to_skip = ["compile_config", "build_config"]
        for key in config_to_skip:
            self._recursive_pop(config, key)
        with open(config_path, "w") as config_file:
            config_file.write(json.dumps(config, indent=4))

    def _save_metadata(self, layer):
        from keras_hub.src.models.task import Task
        from keras_hub.src.version_utils import __version__ as keras_hub_version

        # Find all tasks that are compatible with the backbone.
        # E.g. for `BertBackbone` we would have `TextClassifier` and `MaskedLM`.
        # For `ResNetBackbone` we would have `ImageClassifier`.
        tasks = list_subclasses(Task)
        tasks = filter(lambda x: x.backbone_cls is type(layer), tasks)
        tasks = [task.__base__.__name__ for task in tasks]
        # Keep task list alphabetical.
        tasks = sorted(tasks)

        keras_version = keras.version() if hasattr(keras, "version") else None
        metadata = {
            "keras_version": keras_version,
            "keras_hub_version": keras_hub_version,
            "parameter_count": layer.count_params(),
            "date_saved": datetime.datetime.now().strftime("%Y-%m-%d@%H:%M:%S"),
            "tasks": tasks,
        }
        metadata_path = os.path.join(self.preset_dir, METADATA_FILE)
        with open(metadata_path, "w") as metadata_file:
            metadata_file.write(json.dumps(metadata, indent=4))
