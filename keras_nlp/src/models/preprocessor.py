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

import keras

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.layers.preprocessing.preprocessing_layer import (
    PreprocessingLayer,
)
from keras_nlp.src.utils.preset_utils import PREPROCESSOR_CONFIG_FILE
from keras_nlp.src.utils.preset_utils import TOKENIZER_CONFIG_FILE
from keras_nlp.src.utils.preset_utils import check_config_class
from keras_nlp.src.utils.preset_utils import check_file_exists
from keras_nlp.src.utils.preset_utils import check_format
from keras_nlp.src.utils.preset_utils import list_presets
from keras_nlp.src.utils.preset_utils import list_subclasses
from keras_nlp.src.utils.preset_utils import load_serialized_object
from keras_nlp.src.utils.preset_utils import save_serialized_object
from keras_nlp.src.utils.python_utils import classproperty


@keras_nlp_export("keras_nlp.models.Preprocessor")
class Preprocessor(PreprocessingLayer):
    """Base class for preprocessing layers.

    A `Preprocessor` layer wraps a `keras_nlp.tokenizer.Tokenizer` to provide a
    complete preprocessing setup for a given task. For example a masked language
    modeling preprocessor will take in raw input strings, and output
    `(x, y, sample_weight)` tuples. Where `x` contains token id sequences with
    some

    This class can be subclassed similar to any `keras.layers.Layer`, by
    defining `build()`, `call()` and `get_config()` methods. All subclasses
    should set the `tokenizer` property on construction.
    """

    tokenizer_cls = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokenizer = None

    def __setattr__(self, name, value):
        # Work around torch setattr for properties.
        if name in ["tokenizer"]:
            return object.__setattr__(self, name, value)
        return super().__setattr__(name, value)

    @property
    def tokenizer(self):
        """The tokenizer used to tokenize strings."""
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        self._tokenizer = value

    def get_config(self):
        config = super().get_config()
        config["tokenizer"] = keras.layers.serialize(self.tokenizer)
        return config

    @classmethod
    def from_config(cls, config):
        if "tokenizer" in config and isinstance(config["tokenizer"], dict):
            config["tokenizer"] = keras.layers.deserialize(config["tokenizer"])
        return cls(**config)

    @classproperty
    def presets(cls):
        presets = list_presets(cls)
        # We can also load backbone presets.
        if cls.tokenizer_cls is not None:
            presets.update(cls.tokenizer_cls.presets)
        for subclass in list_subclasses(cls):
            presets.update(subclass.presets)
        return presets

    @classmethod
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        """Instantiate a `keras_nlp.models.Preprocessor` from a model preset.

        A preset is a directory of configs, weights and other file assets used
        to save and load a pre-trained model. The `preset` can be passed as a
        one of:

        1. a built in preset identifier like `'bert_base_en'`
        2. a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
        3. a Hugging Face handle like `'hf://user/bert_base_en'`
        4. a path to a local preset directory like `'./bert_base_en'`

        For any `Preprocessor` subclass, you can run `cls.presets.keys()` to
        list all built-in presets available on the class.

        As there are usually multiple preprocessing classes for a given model,
        this method should be called on a specific subclass like
        `keras_nlp.models.BertPreprocessor.from_preset()`.

        Args:
            preset: string. A built in preset identifier, a Kaggle Models
                handle, a Hugging Face handle, or a path to a local directory.

        Examples:
        ```python
        # Load a preprocessor for Gemma generation.
        preprocessor = keras_nlp.models.GemmaCausalLMPreprocessor.from_preset(
            "gemma_2b_en",
        )

        # Load a preprocessor for Bert classification.
        preprocessor = keras_nlp.models.BertPreprocessor.from_preset(
            "bert_base_en",
        )
        ```
        """
        format = check_format(preset)

        if format == "transformers":
            if cls.tokenizer_cls is None:
                raise ValueError("Tokenizer class is None")
            tokenizer = cls.tokenizer_cls.from_preset(preset)
            return cls(tokenizer=tokenizer, **kwargs)

        if cls == Preprocessor:
            raise ValueError(
                "Do not call `Preprocessor.from_preset()` directly. Instead call a "
                "choose a particular task class, e.g. "
                "`keras_nlp.models.BertPreprocessor.from_preset()`."
            )
        # Check if we should load a `preprocessor.json` directly.
        load_preprocessor_config = False
        if check_file_exists(preset, PREPROCESSOR_CONFIG_FILE):
            preprocessor_preset_cls = check_config_class(
                preset, PREPROCESSOR_CONFIG_FILE
            )
            if issubclass(preprocessor_preset_cls, cls):
                load_preprocessor_config = True
        if load_preprocessor_config:
            # Preprocessor case.
            preprocessor = load_serialized_object(
                preset,
                PREPROCESSOR_CONFIG_FILE,
            )
            preprocessor.tokenizer.load_preset_assets(preset)
            return preprocessor

        # Tokenizer case.
        # If `preprocessor.json` doesn't exist or preprocessor preset class is
        # different from the calling class, create the preprocessor based on
        # `tokenizer.json`.
        tokenizer_preset_cls = check_config_class(
            preset, config_file=TOKENIZER_CONFIG_FILE
        )
        if tokenizer_preset_cls is not cls.tokenizer_cls:
            subclasses = list_subclasses(cls)
            subclasses = tuple(
                filter(
                    lambda x: x.tokenizer_cls == tokenizer_preset_cls,
                    subclasses,
                )
            )
            if len(subclasses) == 0:
                raise ValueError(
                    f"No registered subclass of `{cls.__name__}` can load "
                    f"a `{tokenizer_preset_cls.__name__}`."
                )
            if len(subclasses) > 1:
                names = ", ".join(f"`{x.__name__}`" for x in subclasses)
                raise ValueError(
                    f"Ambiguous call to `{cls.__name__}.from_preset()`. "
                    f"Found multiple possible subclasses {names}. "
                    "Please call `from_preset` on a subclass directly."
                )

        tokenizer = load_serialized_object(preset, TOKENIZER_CONFIG_FILE)
        tokenizer.load_preset_assets(preset)
        preprocessor = cls(tokenizer=tokenizer, **kwargs)

        return preprocessor

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
        self.tokenizer.save_to_preset(preset_dir)
