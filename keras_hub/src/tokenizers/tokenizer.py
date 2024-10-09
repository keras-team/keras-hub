import os

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.preprocessing_layer import (
    PreprocessingLayer,
)
from keras_hub.src.utils.preset_utils import ASSET_DIR
from keras_hub.src.utils.preset_utils import TOKENIZER_CONFIG_FILE
from keras_hub.src.utils.preset_utils import builtin_presets
from keras_hub.src.utils.preset_utils import find_subclass
from keras_hub.src.utils.preset_utils import get_file
from keras_hub.src.utils.preset_utils import get_preset_loader
from keras_hub.src.utils.preset_utils import get_preset_saver
from keras_hub.src.utils.python_utils import classproperty
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export(
    [
        "keras_hub.models.Tokenizer",
        "keras_hub.tokenizers.Tokenizer",
    ]
)
class Tokenizer(PreprocessingLayer):
    """A base class for tokenizer layers.

    Tokenizers in the KerasHub library should all subclass this layer.
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
    class WhitespaceSplitterTokenizer(keras_hub.tokenizers.Tokenizer):
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

    backbone_cls = None

    def __init__(self, *args, **kwargs):
        self.config_file = kwargs.pop("config_file", TOKENIZER_CONFIG_FILE)
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

    @property
    def special_tokens(self):
        """List all built-in special tokens for the tokenizer."""
        if not hasattr(self, "_special_token_attrs"):
            return []
        tokens = set(getattr(self, a) for a in self._special_token_attrs)
        return list(tokens)

    @property
    def special_token_ids(self):
        """List all built-in special token ids for the tokenizer."""
        if not hasattr(self, "_special_token_attrs"):
            return []
        ids = set(getattr(self, f"{a}_id") for a in self._special_token_attrs)
        if None in ids:
            raise ValueError(
                "Cannot access `special_token_ids` before a vocabulary has "
                "been set on the tokenizer."
            )
        return list(ids)

    def _add_special_token(self, token, name):
        if not hasattr(self, "_special_token_attrs"):
            self._special_token_attrs = []
        self._special_token_attrs.append(name)
        setattr(self, name, token)
        try:
            id = self.token_to_id(token)
        except (ValueError, AttributeError):
            id = None
        setattr(self, f"{name}_id", id)

    def _update_special_token_ids(self):
        if not hasattr(self, "_special_token_attrs"):
            return
        vocabulary = self.get_vocabulary()
        for attr in set(self._special_token_attrs):
            token = getattr(self, attr)
            if token not in vocabulary:
                classname = self.__class__.__name__
                raise ValueError(
                    f"Cannot find special token `'{token}'` in the provided "
                    f"vocabulary for `{classname}`. Please ensure `'{token}'` "
                    "is in the provided vocabulary when creating the Tokenizer."
                )
        for attr in self._special_token_attrs:
            token = getattr(self, attr)
            setattr(self, f"{attr}_id", self.token_to_id(token))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "config_file": self.config_file,
            }
        )
        return config

    def save_to_preset(self, preset_dir):
        """Save tokenizer to a preset directory.

        Args:
            preset_dir: The path to the local model preset directory.
        """
        saver = get_preset_saver(preset_dir)
        saver.save_tokenizer(self)

    @preprocessing_function
    def call(self, inputs, *args, training=None, **kwargs):
        return self.tokenize(inputs, *args, **kwargs)

    def load_preset_assets(self, preset):
        asset_path = None
        for asset in self.file_assets:
            subdir = self.config_file.split(".")[0]
            preset_path = os.path.join(ASSET_DIR, subdir, asset)
            asset_path = get_file(preset, preset_path)
        tokenizer_config_file = os.path.dirname(asset_path)
        self.load_assets(tokenizer_config_file)

    @classproperty
    def presets(cls):
        """List built-in presets for a `Tokenizer` subclass."""
        return builtin_presets(cls)

    @classmethod
    def from_preset(
        cls,
        preset,
        config_file=TOKENIZER_CONFIG_FILE,
        **kwargs,
    ):
        """Instantiate a `keras_hub.models.Tokenizer` from a model preset.

        A preset is a directory of configs, weights and other file assets used
        to save and load a pre-trained model. The `preset` can be passed as
        one of:

        1. a built-in preset identifier like `'bert_base_en'`
        2. a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
        3. a Hugging Face handle like `'hf://user/bert_base_en'`
        4. a path to a local preset directory like `'./bert_base_en'`

        For any `Tokenizer` subclass, you can run `cls.presets.keys()` to list
        all built-in presets available on the class.

        This constructor can be called in one of two ways. Either from the base
        class like `keras_hub.models.Tokenizer.from_preset()`, or from
        a model class like `keras_hub.models.GemmaTokenizer.from_preset()`.
        If calling from the base class, the subclass of the returning object
        will be inferred from the config in the preset directory.

        Args:
            preset: string. A built-in preset identifier, a Kaggle Models
                handle, a Hugging Face handle, or a path to a local directory.
            load_weights: bool. If `True`, the weights will be loaded into the
                model architecture. If `False`, the weights will be randomly
                initialized.

        Examples:
        ```python
        # Load a preset tokenizer.
        tokenizer = keras_hub.tokenizer.Tokenizer.from_preset("bert_base_en")

        # Tokenize some input.
        tokenizer("The quick brown fox tripped.")

        # Detokenize some input.
        tokenizer.detokenize([5, 6, 7, 8, 9])
        ```
        """
        loader = get_preset_loader(preset)
        backbone_cls = loader.check_backbone_class()
        if cls.backbone_cls != backbone_cls:
            cls = find_subclass(preset, cls, backbone_cls)
        return loader.load_tokenizer(cls, config_file, **kwargs)
