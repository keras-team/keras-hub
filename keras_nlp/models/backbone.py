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
from keras_nlp.models.lora import add_lora_layers
from keras_nlp.models.lora import merge_lora_layers
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring


@keras.saving.register_keras_serializable(package="keras_nlp")
class Backbone(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._token_embedding = None
        self.has_lora_layers = False

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
        """The default layers to target with lora for this model."""
        return ["query", "value"]

    def add_lora_layers(
        self,
        lora_layer_paths=[],
        trainable_weight_paths=[],
        rank=8,
        alpha=32,
    ):
        """Add LoRA adapter layers for this backbone.

        This method provides high-level API for using [LoRA](https://arxiv.org/pdf/2106.09685.pdf)
        to fine-tune a model in a parameter efficient manner.

        The method will proceed in two steps:
        1) First, freeze all weights in the model besides a configurable set of
           extra trainable weigths.
        2) Second, replace a configurable set of of dense layers with
           `keras_nlp.layers.LoraDense` layers (the parameters are always
           trainable).

        The overall effect will be to dramatically reduce the number of
        trainable parameters in the model, moderately speeding training and
        dramatically reducing memory usage.

        Args:
            lora_layer_paths: A list of string regex paths that will be used to
                target dense layers to replace with `keras_nlp.layers.LoraDense`
                adapter layers. This matching will done by looking at the `path`
                of a dense layers variables. For example, if the `path` of a
                dense kernel is `decoder_layer_4/self_attention/query/kernel`,
                the layer could be matched with `"query"` (to target all query
                layers), or `"decoder_layer_4.*query"` (to target the
                specific layer). If this parameter is unset, the backbone
                defined defaults will be used (usually query and value
                projections).
            trainable_weight_paths: A list of string regex paths that will be
                used to target model parameters to keep trainable. Matching
                will be done in the same manner as with `lora_layer_paths`.
                Pass `["bias"]` to leave all bias paramaters accross the model
                trainable.
            rank: int The inner rank of the decomposed dense transformation. The
                lower this number, the less trainable parameters the layer will
                have.
            alpha: float. A constant value used for scaling the lora update. The
                lora update to the original dense transformation will be scaled by
                `alpha / rank`.
        """
        if not lora_layer_paths:
            lora_layer_paths = self.lora_layer_paths
        add_lora_layers(
            self,
            lora_layer_paths=lora_layer_paths,
            trainable_weight_paths=trainable_weight_paths,
            rank=rank,
            alpha=alpha,
        )
        self.has_lora_layers = True

    def merge_lora_layers(self):
        """Merge all LoRA updates and remove LoRA adapter layers.

        This method will merge all LoRA update back into the original dense
        kernels, and restore the model architecture to it's orgiinal state.

        The method will proceed in two steps:
        1) First, merge all lora updates and remove all `LoraDense` layers.
        1) Second, unfreeze all model weights.

        After `merge_lora_layers`, the model will have the same checkpoint
        structure and inference latency as before calling `add_lora_layers()`.
        """
        merge_lora_layers(self)
        self.has_lora_layers = False

    def get_config(self):
        if self.has_lora_layers:
            raise ValueError(
                "Attempting to serialize a model with lora layers. Call "
                "`model.merge_lora_layers()` before saving, cloning, or "
                f"serializing a model. Received: model={self}"
            )
        # Don't chain to super here. The default `get_config()` for functional
        # models is nested and cannot be passed to our Backbone constructors.
        return {
            "name": self.name,
            "trainable": self.trainable,
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
