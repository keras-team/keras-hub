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

from keras_nlp.backend import keras
from keras_nlp.layers.modeling.lora_dense import LoraDense
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring


@keras.saving.register_keras_serializable(package="keras_nlp")
class Backbone(keras.Model):
    def __init__(self, *args, add_lora_layers=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._token_embedding = None
        self.has_lora_layers = False
        if add_lora_layers:
            self.add_lora_layers()

    def __setattr__(self, name, value):
        # Work around torch setattr for properties.
        if name in ["token_embedding"]:
            return object.__setattr__(self, name, value)
        return super().__setattr__(name, value)

    @property
    def token_embedding(self):
        """A `keras.layers.Embedding` instance for embedding token ids.

        This layer integer token ids to the hidden dim of the model.
        """
        return self._token_embedding

    @token_embedding.setter
    def token_embedding(self, value):
        # Workaround tf.keras h5 checkpoint loading, which is sensitive to layer
        # count mismatches and does not deduplicate layers. This could go away
        # if we update our checkpoints to the newer `.weights.h5` format.
        self._setattr_tracking = False
        self._token_embedding = value
        self._setattr_tracking = True

    @property
    def lora_layer_paths(self):
        """Path to model layers to replace with lora adapaters.

        Should be overridden by subclasses who store key/value paths under
        different names.
        """
        return ["_query_dense", "_value_dense"]

    def add_lora_layers(self, rank=8, alpha=32):
        """Add lora layers, and freeze all other weights in a model.

        Adding lora layers will dramatically reduce the number of trainable
        parameters in a model, reducing the memory requirements for `fit()`.
        """
        if self.has_lora_layers:
            return
        self.has_lora_layers = True

        # Add `LoraDense` adapter layers.
        for layer in self._flatten_layers(include_self=False):
            if not layer._flatten_layers(include_self=False):
                layer.trainable = False
            for path in self.lora_layer_paths:
                dense = getattr(layer, path, None)
                if dense:
                    lora_dense = LoraDense(dense, rank=rank, alpha=alpha)
                    setattr(layer, path, lora_dense)

        # Clear compile cache.
        if self.compiled:
            self.compile_from_config(self.get_compile_config())

    def merge_lora_layers(self):
        """Merge lora updates back into the original dense kernels.

        Removing lora layers will "squash" all kernel updates back into the
        original dense layers. Removing any slowdown during inference caused by
        the additional parameters.
        """
        if not self.has_lora_layers:
            return
        self.has_lora_layers = False

        # Merge lora weights back into dense layers.
        for layer in self._flatten_layers(include_self=False):
            layer.trainable = True
            for path in self.lora_layer_paths:
                lora_dense = getattr(layer, path, None)
                if lora_dense and isinstance(lora_dense, LoraDense):
                    dense = lora_dense.merge_weights()
                    setattr(layer, path, dense)

        # Clear compile cache.
        if self.compiled:
            self.compile_from_config(self.get_compile_config())

    def get_config(self):
        # Don't chain to super here. The default `get_config()` for functional
        # models is nested and cannot be passed to our Backbone constructors.
        return {
            "name": self.name,
            "trainable": self.trainable,
            "add_lora_layers": self.has_lora_layers,
        }

    @classmethod
    def from_config(cls, config):
        # The default `from_config()` for functional models will return a
        # vanilla `keras.Model`. We override it to get a subclass instance back.
        return cls(**config)

    @classproperty
    def presets(cls):
        return {}

    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=True,
        **kwargs,
    ):
        """Instantiate {{model_name}} model from preset architecture and weights.

        Args:
            preset: string. Must be one of "{{preset_names}}".
            load_weights: Whether to load pre-trained weights into model.
                Defaults to `True`.

        Examples:
        ```python
        # Load architecture and weights from preset
        model = keras_nlp.models.{{model_name}}.from_preset(
            "{{example_preset_name}}"
        )

        # Load randomly initialized model from preset architecture
        model = keras_nlp.models.{{model_name}}.from_preset(
            "{{example_preset_name}}",
            load_weights=False
        )
        ```
        """

        if not cls.presets:
            raise NotImplementedError(
                "No presets have been created for this class."
            )

        if preset not in cls.presets:
            raise ValueError(
                "`preset` must be one of "
                f"""{", ".join(cls.presets)}. Received: {preset}."""
            )
        metadata = cls.presets[preset]
        config = metadata["config"]
        model = cls.from_config({**config, **kwargs})

        if not load_weights:
            return model

        weights = keras.utils.get_file(
            "model.h5",
            metadata["weights_url"],
            cache_subdir=os.path.join("models", preset),
            file_hash=metadata["weights_hash"],
        )
        model.load_weights(weights)
        return model

    def __init_subclass__(cls, **kwargs):
        # Use __init_subclass__ to setup a correct docstring for from_preset.
        super().__init_subclass__(**kwargs)

        # If the subclass does not define from_preset, assign a wrapper so that
        # each class can have a distinct docstring.
        if "from_preset" not in cls.__dict__:

            def from_preset(calling_cls, *args, **kwargs):
                return super(cls, calling_cls).from_preset(*args, **kwargs)

            cls.from_preset = classmethod(from_preset)

        # Format and assign the docstring unless the subclass has overridden it.
        if cls.from_preset.__doc__ is None:
            cls.from_preset.__func__.__doc__ = Backbone.from_preset.__doc__
            format_docstring(
                model_name=cls.__name__,
                example_preset_name=next(iter(cls.presets), ""),
                preset_names='", "'.join(cls.presets),
            )(cls.from_preset.__func__)
