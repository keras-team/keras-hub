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

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.backend import keras
from keras_nlp.layers.preprocessing.preprocessing_layer import (
    PreprocessingLayer,
)
from keras_nlp.utils.preset_utils import PREPROCESSOR_CONFIG_FILE
from keras_nlp.utils.preset_utils import TOKENIZER_CONFIG_FILE
from keras_nlp.utils.preset_utils import check_config_class
from keras_nlp.utils.preset_utils import list_presets
from keras_nlp.utils.preset_utils import list_subclasses
from keras_nlp.utils.preset_utils import load_from_preset
from keras_nlp.utils.preset_utils import save_to_preset
from keras_nlp.utils.python_utils import classproperty


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
        if cls == Preprocessor:
            raise ValueError(
                "Do not call `Preprocessor.from_preset()` directly. Instead call a "
                "choose a particular task class, e.g. "
                "`keras_nlp.models.BertPreprocessor.from_preset()`."
            )
        config_file = "tokenizer.json"
        preset_cls = check_config_class(preset, config_file=config_file)
        if preset_cls is not cls.tokenizer_cls:
            subclasses = list_subclasses(cls)
            subclasses = tuple(
                filter(lambda x: x.tokenizer_cls == preset_cls, subclasses)
            )
            if len(subclasses) == 0:
                raise ValueError(
                    f"No registered subclass of `{cls.__name__}` can load "
                    f"a `{preset_cls.__name__}`."
                )
            if len(subclasses) > 1:
                names = ", ".join(f"`{x.__name__}`" for x in subclasses)
                raise ValueError(
                    f"Ambiguous call to `{cls.__name__}.from_preset()`. "
                    f"Found multiple possible subclasses {names}. "
                    "Please call `from_preset` on a subclass directly."
                )
            cls = subclasses[0]
        tokenizer = load_from_preset(
            preset,
            config_file=config_file,
        )
        return cls(tokenizer=tokenizer, **kwargs)

    def save_to_preset(self, preset):
        """TODO: add docstring."""
        save_to_preset(
            self.tokenizer,
            preset,
            config_filename=TOKENIZER_CONFIG_FILE,
        )

        preprocessor_config_path = os.path.join(
            preset, PREPROCESSOR_CONFIG_FILE
        )
        preprocessor_config = keras.saving.serialize_keras_object(self)
        with open(preprocessor_config_path, "w") as config_file:
            config_file.write(json.dumps(preprocessor_config, indent=4))
