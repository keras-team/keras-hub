# Copyright 2024 The KerasHub Authors
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

import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.preprocessing_layer import (
    PreprocessingLayer,
)
from keras_hub.src.utils.preset_utils import PREPROCESSOR_CONFIG_FILE
from keras_hub.src.utils.preset_utils import builtin_presets
from keras_hub.src.utils.preset_utils import find_subclass
from keras_hub.src.utils.preset_utils import get_preset_loader
from keras_hub.src.utils.preset_utils import save_serialized_object
from keras_hub.src.utils.python_utils import classproperty


@keras_hub_export("keras_hub.models.Preprocessor")
class Preprocessor(PreprocessingLayer):
    """Base class for preprocessing layers.

    A `Preprocessor` layer wraps a `keras_hub.tokenizer.Tokenizer` to provide a
    complete preprocessing setup for a given task. For example a masked language
    modeling preprocessor will take in raw input strings, and output
    `(x, y, sample_weight)` tuples. Where `x` contains token id sequences with
    some

    This class can be subclassed similar to any `keras.layers.Layer`, by
    defining `build()`, `call()` and `get_config()` methods. All subclasses
    should set the `tokenizer` property on construction.
    """

    backbone_cls = None
    tokenizer_cls = None
    audio_converter_cls = None
    image_converter_cls = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer = None
        self._image_converter = None
        self._audio_converter = None

    def __setattr__(self, name, value):
        # Work around torch setattr for properties.
        if name in ["tokenizer", "audio_converter", "image_converter"]:
            return object.__setattr__(self, name, value)
        return super().__setattr__(name, value)

    @property
    def tokenizer(self):
        """The tokenizer used to tokenize strings."""
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        self._tokenizer = value

    @property
    def audio_converter(self):
        """The audio converter used to preprocess audio data."""
        return self._audio_converter

    @audio_converter.setter
    def audio_converter(self, value):
        self._audio_converter = value

    @property
    def image_converter(self):
        """The image converter used to preprocess image data."""
        return self._image_converter

    @image_converter.setter
    def image_converter(self, value):
        self._image_converter = value

    def get_config(self):
        config = super().get_config()
        if self.tokenizer:
            config["tokenizer"] = keras.layers.serialize(self.tokenizer)
        if self.audio_converter:
            config["audio_converter"] = keras.layers.serialize(
                self.audio_converter
            )
        if self.image_converter:
            config["image_converter"] = keras.layers.serialize(
                self.image_converter
            )
        return config

    @classmethod
    def from_config(cls, config):
        if "tokenizer" in config and isinstance(config["tokenizer"], dict):
            config["tokenizer"] = keras.layers.deserialize(config["tokenizer"])
        if "audio_converter" in config and isinstance(
            config["audio_converter"], dict
        ):
            config["audio_converter"] = keras.layers.deserialize(
                config["audio_converter"]
            )
        if "image_converter" in config and isinstance(
            config["image_converter"], dict
        ):
            config["image_converter"] = keras.layers.deserialize(
                config["image_converter"]
            )
        return cls(**config)

    @classproperty
    def presets(cls):
        """List built-in presets for a `Preprocessor` subclass."""
        return builtin_presets(cls)

    @classmethod
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        """Instantiate a `keras_hub.models.Preprocessor` from a model preset.

        A preset is a directory of configs, weights and other file assets used
        to save and load a pre-trained model. The `preset` can be passed as
        one of:

        1. a built-in preset identifier like `'bert_base_en'`
        2. a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
        3. a Hugging Face handle like `'hf://user/bert_base_en'`
        4. a path to a local preset directory like `'./bert_base_en'`

        For any `Preprocessor` subclass, you can run `cls.presets.keys()` to
        list all built-in presets available on the class.

        As there are usually multiple preprocessing classes for a given model,
        this method should be called on a specific subclass like
        `keras_hub.models.BertTextClassifierPreprocessor.from_preset()`.

        Args:
            preset: string. A built-in preset identifier, a Kaggle Models
                handle, a Hugging Face handle, or a path to a local directory.

        Examples:
        ```python
        # Load a preprocessor for Gemma generation.
        preprocessor = keras_hub.models.GemmaCausalLMPreprocessor.from_preset(
            "gemma_2b_en",
        )

        # Load a preprocessor for Bert classification.
        preprocessor = keras_hub.models.BertTextClassifierPreprocessor.from_preset(
            "bert_base_en",
        )
        ```
        """
        if cls == Preprocessor:
            raise ValueError(
                "Do not call `Preprocessor.from_preset()` directly. Instead "
                "choose a particular task preprocessing class, e.g. "
                "`keras_hub.models.TextClassifierPreprocessor.from_preset()`."
            )

        loader = get_preset_loader(preset)
        backbone_cls = loader.check_backbone_class()
        # Detect the correct subclass if we need to.
        if cls.backbone_cls != backbone_cls:
            cls = find_subclass(preset, cls, backbone_cls)
        return loader.load_preprocessor(cls, **kwargs)

    def save_to_preset(self, preset_dir):
        """Save preprocessor to a preset directory.

        Args:
            preset_dir: The path to the local model preset directory.
        """
        save_serialized_object(
            self,
            preset_dir,
            config_file=PREPROCESSOR_CONFIG_FILE,
        )
        if self.tokenizer:
            self.tokenizer.save_to_preset(preset_dir)
        if self.audio_converter:
            self.audio_converter.save_to_preset(preset_dir)
        if self.image_converter:
            self.image_converter.save_to_preset(preset_dir)
