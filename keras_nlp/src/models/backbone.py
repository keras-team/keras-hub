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

import keras

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.utils.preset_utils import CONFIG_FILE
from keras_nlp.src.utils.preset_utils import MODEL_WEIGHTS_FILE
from keras_nlp.src.utils.preset_utils import check_config_class
from keras_nlp.src.utils.preset_utils import get_file
from keras_nlp.src.utils.preset_utils import jax_memory_cleanup
from keras_nlp.src.utils.preset_utils import list_presets
from keras_nlp.src.utils.preset_utils import list_subclasses
from keras_nlp.src.utils.preset_utils import load_serialized_object
from keras_nlp.src.utils.preset_utils import save_metadata
from keras_nlp.src.utils.preset_utils import save_serialized_object
from keras_nlp.src.utils.preset_utils import validate_metadata
from keras_nlp.src.utils.python_utils import classproperty


@keras_nlp_export("keras_nlp.models.Backbone")
class Backbone(keras.Model):
    """Base class for all `Backbone` models.

    A `Backbone` is the basic architecture for a given NLP model. Unlike a
    `keras_nlp.models.Task`, a `Backbone` is not tailored to any specific loss
    function and training setup. A `Backbone` generally outputs the last hidden
    states of an architecture before any output predictions.

    A `Backbone` can be used in one of two ways:

    1. Through a `Task` class, which will wrap and extend a `Backbone` so it
       can be used with high level Keras functions like `fit()`, `predict()` or
       `evaluate()`. `Task` classes are built with a particular training
       objective in mind (e.g. classification or language modeling).
    2. Directly, by extending underlying functional model with additional
       outputs and training setup. This is the most flexible approach, and can
       allow for any outputs, loss, or custom training loop.

    All backbones include a `from_preset()` constructor which can be used to
    load a pre-trained config and weights.

    Example:
    ```python
    # Load a BERT backbone with pre-trained weights.
    backbone = keras_nlp.models.Backbone.from_preset(
        "bert_base_en",
    )
    # Load a GPT2 backbone with pre-trained weights at bfloat16 precision.
    backbone = keras_nlp.models.Backbone.from_preset(
        "gpt2_base_en",
        dtype="bfloat16",
        trainable=False,
    )
    ```
    """

    def __init__(self, *args, dtype=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._functional_layer_ids = set(
            id(layer) for layer in self._flatten_layers()
        )
        self._initialized = True
        if dtype is not None:
            if isinstance(dtype, keras.DTypePolicy):
                self.dtype_policy = dtype
            else:
                self.dtype_policy = keras.DTypePolicy(dtype)

    def __setattr__(self, name, value):
        # Work around setattr issues for Keras 2 and Keras 3 torch backend.
        # Since all our state is covered by functional model we can route
        # around custom setattr calls.
        is_property = isinstance(getattr(type(self), name, None), property)
        is_unitialized = not hasattr(self, "_initialized")
        simple_setattr = keras.config.backend() == "torch"
        if simple_setattr and (is_property or is_unitialized):
            return object.__setattr__(self, name, value)
        return super().__setattr__(name, value)

    @property
    def token_embedding(self):
        """A `keras.layers.Embedding` instance for embedding token ids.

        This layer embeds integer token ids to the hidden dim of the model.
        """
        return getattr(self, "_token_embedding", None)

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
        """List built-in presets for a `Task` subclass."""
        presets = list_presets(cls)
        for subclass in list_subclasses(cls):
            presets.update(subclass.presets)
        return presets

    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=True,
        **kwargs,
    ):
        """Instantiate a `keras_nlp.models.Backbone` from a model preset.

        A preset is a directory of configs, weights and other file assets used
        to save and load a pre-trained model. The `preset` can be passed as a
        one of:

        1. a built in preset identifier like `'bert_base_en'`
        2. a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
        3. a Hugging Face handle like `'hf://user/bert_base_en'`
        4. a path to a local preset directory like `'./bert_base_en'`

        This constructor can be called in one of two ways. Either from the base
        class like `keras_nlp.models.Backbone.from_preset()`, or from
        a model class like `keras_nlp.models.GemmaBackbone.from_preset()`.
        If calling from the base class, the subclass of the returning object
        will be inferred from the config in the preset directory.

        For any `Backbone` subclass, you can run `cls.presets.keys()` to list
        all built-in presets available on the class.

        Args:
            preset: string. A built in preset identifier, a Kaggle Models
                handle, a Hugging Face handle, or a path to a local directory.
            load_weights: bool. If `True`, the weights will be loaded into the
                model architecture. If `False`, the weights will be randomly
                initialized.

        Examples:
        ```python
        # Load a Gemma backbone with pre-trained weights.
        model = keras_nlp.models.Backbone.from_preset(
            "gemma_2b_en",
        )

        # Load a Bert backbone with a pre-trained config and random weights.
        model = keras_nlp.models.Backbone.from_preset(
            "bert_base_en",
            load_weights=False,
        )
        ```
        """
        validate_metadata(preset)
        preset_cls = check_config_class(preset)
        if not issubclass(preset_cls, cls):
            raise ValueError(
                f"Preset has type `{preset_cls.__name__}` which is not a "
                f"a subclass of calling class `{cls.__name__}`. Call "
                f"`from_preset` directly on `{preset_cls.__name__}` instead."
            )

        backbone = load_serialized_object(preset, CONFIG_FILE)
        if load_weights:
            jax_memory_cleanup(backbone)
            backbone.load_weights(get_file(preset, MODEL_WEIGHTS_FILE))

        return backbone

    def save_to_preset(self, preset_dir):
        """Save backbone to a preset directory.

        Args:
            preset_dir: The path to the local model preset directory.
        """
        save_serialized_object(self, preset_dir, config_file=CONFIG_FILE)
        self.save_weights(os.path.join(preset_dir, MODEL_WEIGHTS_FILE))
        save_metadata(self, preset_dir)

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
