import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.utils.preset_utils import builtin_presets
from keras_hub.src.utils.preset_utils import get_preset_loader
from keras_hub.src.utils.preset_utils import get_preset_saver
from keras_hub.src.utils.python_utils import classproperty


@keras_hub_export("keras_hub.models.Backbone")
class Backbone(keras.Model):
    """Base class for all `Backbone` models.

    A `Backbone` is the basic architecture for a given NLP model. Unlike a
    `keras_hub.models.Task`, a `Backbone` is not tailored to any specific loss
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
    backbone = keras_hub.models.Backbone.from_preset(
        "bert_base_en",
    )
    # Load a GPT2 backbone with pre-trained weights at bfloat16 precision.
    backbone = keras_hub.models.Backbone.from_preset(
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
            try:
                self.dtype_policy = keras.dtype_policies.get(dtype)
            # Before Keras 3.2, there is no `keras.dtype_policies.get`.
            except AttributeError:
                if isinstance(dtype, keras.DTypePolicy):
                    dtype = dtype.name
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
        config = {
            "name": self.name,
            "trainable": self.trainable,
        }

        # Add quantization support by utilizing `DTypePolicyMap`
        try:
            if isinstance(
                self.dtype_policy, keras.dtype_policies.DTypePolicyMap
            ):
                config.update({"dtype": self.dtype_policy})
            else:
                policy_map = keras.dtype_policies.DTypePolicyMap()
                for layer in self._flatten_layers():
                    if layer.quantization_mode is not None:
                        policy_map[layer.path] = layer.dtype_policy
                if len(policy_map) > 0:
                    config.update({"dtype": policy_map})
        # Before Keras 3.2, there is no `keras.dtype_policies.get`.
        except AttributeError:
            pass
        return config

    @classmethod
    def from_config(cls, config):
        # The default `from_config()` for functional models will return a
        # vanilla `keras.Model`. We override it to get a subclass instance back.
        return cls(**config)

    @classproperty
    def presets(cls):
        """List built-in presets for a `Backbone` subclass."""
        return builtin_presets(cls)

    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=True,
        **kwargs,
    ):
        """Instantiate a `keras_hub.models.Backbone` from a model preset.

        A preset is a directory of configs, weights and other file assets used
        to save and load a pre-trained model. The `preset` can be passed as a
        one of:

        1. a built-in preset identifier like `'bert_base_en'`
        2. a Kaggle Models handle like `'kaggle://user/bert/keras/bert_base_en'`
        3. a Hugging Face handle like `'hf://user/bert_base_en'`
        4. a path to a local preset directory like `'./bert_base_en'`

        This constructor can be called in one of two ways. Either from the base
        class like `keras_hub.models.Backbone.from_preset()`, or from
        a model class like `keras_hub.models.GemmaBackbone.from_preset()`.
        If calling from the base class, the subclass of the returning object
        will be inferred from the config in the preset directory.

        For any `Backbone` subclass, you can run `cls.presets.keys()` to list
        all built-in presets available on the class.

        Args:
            preset: string. A built-in preset identifier, a Kaggle Models
                handle, a Hugging Face handle, or a path to a local directory.
            load_weights: bool. If `True`, the weights will be loaded into the
                model architecture. If `False`, the weights will be randomly
                initialized.

        Examples:
        ```python
        # Load a Gemma backbone with pre-trained weights.
        model = keras_hub.models.Backbone.from_preset(
            "gemma_2b_en",
        )

        # Load a Bert backbone with a pre-trained config and random weights.
        model = keras_hub.models.Backbone.from_preset(
            "bert_base_en",
            load_weights=False,
        )
        ```
        """
        loader = get_preset_loader(preset)
        backbone_cls = loader.check_backbone_class()
        if not issubclass(backbone_cls, cls):
            raise ValueError(
                f"Saved preset has type `{backbone_cls.__name__}` which is not "
                f"a subclass of calling class `{cls.__name__}`. Call "
                f"`from_preset` directly on `{backbone_cls.__name__}` instead."
            )
        return loader.load_backbone(backbone_cls, load_weights, **kwargs)

    def save_to_preset(self, preset_dir):
        """Save backbone to a preset directory.

        Args:
            preset_dir: The path to the local model preset directory.
        """
        saver = get_preset_saver(preset_dir)
        saver.save_backbone(self)

    def get_lora_target_names(self):
        """Returns list of layer names which are to be LoRA-fied.

        Subclasses can override this method if the names of layers to be
        LoRa-fied are different.
        """
        return ["query_dense", "value_dense", "query", "value"]

    def enable_lora(self, rank):
        """Enable Lora on the backbone.

        Calling this method will freeze all weights on the backbone,
        while enabling Lora on the query & value `EinsumDense` layers
        of the attention layers.
        """
        target_names = self.get_lora_target_names()

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
