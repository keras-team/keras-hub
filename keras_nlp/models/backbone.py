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

from keras_nlp.backend import config
from keras_nlp.backend import keras
from keras_nlp.utils.preset_utils import check_preset_class
from keras_nlp.utils.preset_utils import load_from_preset
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring


@keras.saving.register_keras_serializable(package="keras_nlp")
class Backbone(keras.Model):
    def __init__(self, *args, dtype=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._functional_layer_ids = set(
            id(layer) for layer in self._flatten_layers()
        )
        self._initialized = True

    def __dir__(self):
        if config.keras_3():
            return super().__dir__()

        # Temporary fixes for Keras 2 saving. This mimics the following PR for
        # older version of Keras: https://github.com/keras-team/keras/pull/18982
        def filter_fn(attr):
            if attr in [
                "_layer_checkpoint_dependencies",
                "transformer_layers",
                "encoder_transformer_layers",
                "decoder_transformer_layers",
            ]:
                return False
            return id(getattr(self, attr)) not in self._functional_layer_ids

        return filter(filter_fn, super().__dir__())

    def __setattr__(self, name, value):
        # Work around setattr issues for Keras 2 and Keras 3 torch backend.
        # Since all our state is covered by functional model we can route
        # around custom setattr calls.
        is_property = isinstance(getattr(type(self), name, None), property)
        is_unitialized = not hasattr(self, "_initialized")
        is_torch = config.backend() == "torch"
        is_keras_2 = not config.keras_3()
        if is_torch and (is_property or is_unitialized):
            return object.__setattr__(self, name, value)
        if is_keras_2 and is_unitialized:
            return object.__setattr__(self, name, value)
        return super().__setattr__(name, value)

    @property
    def token_embedding(self):
        """A `keras.layers.Embedding` instance for embedding token ids.

        This layer embeds integer token ids to the hidden dim of the model.
        """
        return self._token_embedding

    @token_embedding.setter
    def token_embedding(self, value):
        self._token_embedding = value

    def get_config(self):
        # Don't chain to super here. `get_config()` for functional models is
        # a nested layer config and cannot be passed to Backbone constructors.
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
        # We support short IDs for official presets, e.g. `"bert_base_en"`.
        # Map these to a Kaggle Models handle.
        if preset in cls.presets:
            preset = cls.presets[preset]["kaggle_handle"]

        check_preset_class(preset, cls)
        return load_from_preset(
            preset,
            load_weights=load_weights,
            config_overrides=kwargs,
        )

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

    def enable_lora(self, rank):
        """Enable Lora on the backbone.

        Calling this method will freeze all weights on the backbone,
        while enabling Lora on the query & value `EinsumDense` layers
        of the attention layers.
        """
        target_names = ["query_dense", "value_dense", "query", "value"]
        self.trainable = True
        self._lora_enabled_layers = []
        self._lora_rank = rank
        for layer in self._flatten_layers(include_self=False):
            layer.trainable = False
        all_layers = self._flatten_layers(include_self=False)
        all_layers = [lyr for lyr in all_layers if lyr.weights]
        for i, layer in enumerate(all_layers):
            for name in target_names:
                if layer.name == name:
                    if hasattr(layer, "enable_lora"):
                        layer.trainable = True
                        layer.enable_lora(rank)
                        self._lora_enabled_layers.append(i)

    def save_lora_weights(self, filepath):
        if not getattr(self, "_lora_enabled_layers", []):
            raise ValueError(
                "There are no lora-enabled layers in this model. "
                "Make sure to call `.enable_lora(rank)` first."
            )
        if not str(filepath).endswith(".lora.h5"):
            raise ValueError(
                "The filename must end in `.lora.h5`. "
                f"Received: filepath={filepath}"
            )

        store = keras.src.saving.saving_lib.H5IOStore(filepath, mode="w")
        lora_store = store.make("lora")
        lora_store["rank"] = self._lora_rank
        # We cannot identify layers by name since names are non-unique,
        # so we identify them by index in the topologically sorted list
        # of layers that have weights.
        all_layers = self._flatten_layers(include_self=False)
        all_layers = [lyr for lyr in all_layers if lyr.weights]
        for layer_index in self._lora_enabled_layers:
            # We only lora the einsumdense layers,
            # so the factored weights are always named `kernel`
            layer = all_layers[layer_index]
            inner_store = store.make(f"lora/{layer_index}")
            inner_store["lora_kernel_a"] = layer.lora_kernel_a
            inner_store["lora_kernel_b"] = layer.lora_kernel_b
        store.close()

    def load_lora_weights(self, filepath):
        store = keras.src.saving.saving_lib.H5IOStore(filepath, mode="r")
        lora_store = store.get("lora")
        rank = int(lora_store["rank"][()])

        if not getattr(self, "_lora_enabled_layers", []):
            self.enable_lora(rank)
        else:
            if self._lora_rank != rank:
                raise ValueError(
                    f"The Lora rank expected by file '{filepath}' "
                    f"is rank={rank}, but the model was called with "
                    f"`.enable_lora(rank={self._lora_rank})`. "
                    "Both ranks must match."
                )
        all_layers = self._flatten_layers(include_self=False)
        all_layers = [lyr for lyr in all_layers if lyr.weights]
        for layer_index in self._lora_enabled_layers:
            layer = all_layers[layer_index]
            lora_kernel_a = store.get(f"lora/{layer_index}")["lora_kernel_a"]
            lora_kernel_b = store.get(f"lora/{layer_index}")["lora_kernel_b"]
            layer.lora_kernel_a.assign(lora_kernel_a)
            layer.lora_kernel_b.assign(lora_kernel_b)
        store.close()
