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
import os

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.layers.preprocessing.preprocessing_layer import (
    PreprocessingLayer,
)
from keras_nlp.src.utils.preset_utils import TOKENIZER_ASSET_DIR
from keras_nlp.src.utils.preset_utils import TOKENIZER_CONFIG_FILE
from keras_nlp.src.utils.preset_utils import check_config_class
from keras_nlp.src.utils.preset_utils import check_format
from keras_nlp.src.utils.preset_utils import get_file
from keras_nlp.src.utils.preset_utils import list_presets
from keras_nlp.src.utils.preset_utils import list_subclasses
from keras_nlp.src.utils.preset_utils import load_serialized_object
from keras_nlp.src.utils.preset_utils import save_serialized_object
from keras_nlp.src.utils.preset_utils import save_tokenizer_assets
from keras_nlp.src.utils.python_utils import classproperty
from keras_nlp.src.utils.transformers.convert import load_transformers_tokenizer


@keras_nlp_export(
    [
        "keras_nlp.models.Tokenizer",
        "keras_nlp.tokenizers.Tokenizer",
    ]
)
class Tokenizer(PreprocessingLayer):
    """A base class for tokenizer layers.

    Tokenizers in the KerasNLP library should all subclass this layer.
    The class provides two core methods `tokenize()` and `detokenize()` for
    going from plain text to sequences and back. A tokenizer is a subclass of
    `keras.layers.Layer` and can be combined into a `keras.Model`.

    Subclassers should always implement the `tokenize()` method, which will also
    be the default when calling the layer directly on inputs.

    Subclassers can optionally implement the `detokenize()` method if the
    tokenization is reversible. Otherwise, this can be skipped.

    Subclassers should implement `get_vocabulary()`, `vocabulary_size()`,
    `token_to_id()` and `id_to_token()` if applicable. For some simple
    "vocab free" tokenizers, such as a whitespace splitter show below, these
    methods do not apply and can be skipped.

    Example:

    ```python
    class WhitespaceSplitterTokenizer(keras_nlp.tokenizers.Tokenizer):
        def tokenize(self, inputs):
            return tf.strings.split(inputs)

        def detokenize(self, inputs):
            return tf.strings.reduce_join(inputs, separator=" ", axis=-1)

    tokenizer = WhitespaceSplitterTokenizer()

    # Tokenize some inputs.
    tokenizer.tokenize("This is a test")

    # Shorthard for `tokenize()`.
    tokenizer("This is a test")

    # Detokenize some outputs.
    tokenizer.detokenize(["This", "is", "a", "test"])
    ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_assets = None

    def tokenize(self, inputs, *args, **kwargs):
        """Transform input tensors of strings into output tokens.

        Args:
            inputs: Input tensor, or dict/list/tuple of input tensors.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError(
            "No implementation of `tokenize()` was found for "
            f"{self.__class__.__name__}. All tokenizers should implement "
            "`tokenize()`."
        )

    def detokenize(self, inputs, *args, **kwargs):
        """Transform tokens back into strings.

        Args:
            inputs: Input tensor, or dict/list/tuple of input tensors.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError(
            "No implementation of `detokenize()` was found for "
            f"{self.__class__.__name__}."
        )

    def get_vocabulary(self):
        """Get the tokenizer vocabulary as a list of strings terms."""
        raise NotImplementedError(
            "No implementation of `get_vocabulary()` was found for "
            f"{self.__class__.__name__}."
        )

    def vocabulary_size(self):
        """Returns the total size of the token id space."""
        raise NotImplementedError(
            "No implementation of `vocabulary_size()` was found for "
            f"{self.__class__.__name__}."
        )

    def id_to_token(self, id):
        """Convert an integer id to a string token."""
        raise NotImplementedError(
            "No implementation of `id_to_token()` was found for "
            f"{self.__class__.__name__}."
        )

    def token_to_id(self, token):
        """Convert a string token to an integer id."""
        raise NotImplementedError(
            "No implementation of `token_to_id()` was found for "
            f"{self.__class__.__name__}."
        )

    def save_to_preset(self, preset_dir):
        """Save tokenizer to a preset directory.

        Args:
            preset_dir: The path to the local model preset directory.
        """
        save_serialized_object(
            self,
            preset_dir,
            config_file=TOKENIZER_CONFIG_FILE,
        )
        save_tokenizer_assets(self, preset_dir)

    def call(self, inputs, *args, training=None, **kwargs):
        return self.tokenize(inputs, *args, **kwargs)

    def load_preset_assets(self, preset):
        asset_path = None
        for asset in self.file_assets:
            asset_path = get_file(
                preset, os.path.join(TOKENIZER_ASSET_DIR, asset)
            )
        tokenizer_asset_dir = os.path.dirname(asset_path)
        self.load_assets(tokenizer_asset_dir)

    @classproperty
    def presets(cls):
        """List built-in presets for a `Task` subclass."""
        presets = list_presets(cls)
        for subclass in list_subclasses(cls):
            presets.update(subclass.presets)
        return presets

    @classmethod
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        """Instantiate a `keras_nlp.models.Tokenizer` from a model preset.

        A preset is a directory of configs, weights and other file assets used
        to save and load a pre-trained model. The `preset` can be passed as a
        one of:

        1. a built in preset identifier like `'bert_base_en'`
        2. a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
        3. a Hugging Face handle like `'hf://user/bert_base_en'`
        4. a path to a local preset directory like `'./bert_base_en'`

        For any `Tokenizer` subclass, you can run `cls.presets.keys()` to list
        all built-in presets available on the class.

        This constructor can be called in one of two ways. Either from the base
        class like `keras_nlp.models.Tokenizer.from_preset()`, or from
        a model class like `keras_nlp.models.GemmaTokenizer.from_preset()`.
        If calling from the base class, the subclass of the returning object
        will be inferred from the config in the preset directory.

        Args:
            preset: string. A built in preset identifier, a Kaggle Models
                handle, a Hugging Face handle, or a path to a local directory.
            load_weights: bool. If `True`, the weights will be loaded into the
                model architecture. If `False`, the weights will be randomly
                initialized.

        Examples:
        ```python
        # Load a preset tokenizer.
        tokenizer = keras_nlp.tokenizerTokenizer.from_preset("bert_base_en")

        # Tokenize some input.
        tokenizer("The quick brown fox tripped.")

        # Detokenize some input.
        tokenizer.detokenize([5, 6, 7, 8, 9])
        ```
        """
        format = check_format(preset)
        if format == "transformers":
            return load_transformers_tokenizer(cls, preset)

        preset_cls = check_config_class(
            preset, config_file=TOKENIZER_CONFIG_FILE
        )
        if not issubclass(preset_cls, cls):
            raise ValueError(
                f"Preset has type `{preset_cls.__name__}` which is not a "
                f"a subclass of calling class `{cls.__name__}`. Call "
                f"`from_preset` directly on `{preset_cls.__name__}` instead."
            )

        tokenizer = load_serialized_object(preset, TOKENIZER_CONFIG_FILE)
        tokenizer.load_preset_assets(preset)
        return tokenizer
