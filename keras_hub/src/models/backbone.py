import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.utils.preset_utils import builtin_presets
from keras_hub.src.utils.preset_utils import get_preset_loader
from keras_hub.src.utils.preset_utils import get_preset_saver
from keras_hub.src.utils.python_utils import classproperty


def _dorafy_einsum_dense_layer(layer, rank, lora_alpha):
    """Enable DoRA on a single `EinsumDense` layer.

    DoRA decomposes the pretrained kernel into a magnitude vector `m` and a
    directional matrix. The direction is updated via LoRA's low-rank factors,
    and the magnitude is a free parameter initialized to the column-wise L2
    norms of the original kernel.

    This function:
      1. Enables LoRA on the layer (adding `lora_kernel_a` / `lora_kernel_b`).
      2. Adds a trainable `dora_magnitude` weight of shape
         `(kernel.shape[-1],)` initialized to the column norms of the frozen
         base kernel.
      3. Replaces the layer's `call` with a DoRA forward that rescales the
         LoRA-merged kernel to column-unit norm and multiplies by magnitude.
    """
    layer.enable_lora(rank, lora_alpha=lora_alpha)
    base_kernel = layer._kernel
    # Column norms of the pretrained kernel. The last axis of the kernel is
    # the output dim; all other axes collapse into the "row" direction.
    reduce_axes = tuple(range(len(base_kernel.shape) - 1))
    initial_magnitude = ops.sqrt(
        ops.sum(ops.square(base_kernel), axis=reduce_axes)
    )
    layer._tracker.unlock()
    layer.dora_magnitude = layer.add_weight(
        name="dora_magnitude",
        shape=(base_kernel.shape[-1],),
        initializer="zeros",
        trainable=True,
    )
    layer._tracker.lock()
    layer.dora_magnitude.assign(initial_magnitude)
    layer.dora_enabled = True
    layer.dora_reduce_axes = reduce_axes
    _install_dora_call(layer)


def _install_dora_call(layer):
    """Replace `layer.call` with a DoRA-aware forward."""
    equation = layer.equation
    reduce_axes = layer.dora_reduce_axes
    eps = 1e-6

    def dora_call(inputs, training=None):
        # `layer.kernel` already includes the LoRA update.
        merged = layer.kernel
        norms = ops.sqrt(
            ops.sum(
                ops.square(merged),
                axis=reduce_axes,
                keepdims=True,
            )
        )
        effective = (layer.dora_magnitude / (ops.squeeze(norms) + eps)) * merged
        x = ops.einsum(equation, inputs, effective)
        if layer.bias is not None:
            x = ops.add(x, layer.bias)
        if layer.activation is not None:
            x = layer.activation(x)
        return x

    layer.call = dora_call


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
        dtype = self.dtype_policy
        if not isinstance(dtype, keras.dtype_policies.DTypePolicyMap):
            policy_map = keras.dtype_policies.DTypePolicyMap()
            for layer in self._flatten_layers():
                if layer.quantization_mode is not None:
                    policy_map[layer.path] = layer.dtype_policy
            if len(policy_map) > 0:
                dtype = policy_map

        config.update({"dtype": keras.dtype_policies.serialize(dtype)})
        return config

    @classmethod
    def from_config(cls, config):
        # The default `from_config()` for functional models will return a
        # vanilla `keras.Model`. We override it to get a subclass instance back.
        config = config.copy()
        if "dtype" in config and isinstance(config["dtype"], dict):
            config["dtype"] = keras.dtype_policies.get(config["dtype"])
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
        4. a ModelScope handle like `'modelscope://user/bert_base_en'`
        5. a path to a local preset directory like `'./bert_base_en'`

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

    def save_to_preset(self, preset_dir, max_shard_size=10):
        """Save backbone to a preset directory.

        Args:
            preset_dir: The path to the local model preset directory.
            max_shard_size: `int` or `float`. Maximum size in GB for each
                sharded file. If `None`, no sharding will be done. Defaults to
                `10`.
        """
        saver = get_preset_saver(preset_dir)
        saver.save_backbone(self, max_shard_size=max_shard_size)

    def default_lora_layer_names(self):
        """Returns list of layer names which are to be LoRA-fied."""
        return ["query_dense", "value_dense", "query", "value"]

    def default_dora_layer_names(self):
        """Returns list of layer names which are to be DoRA-fied."""
        return self.default_lora_layer_names()

    def enable_lora(self, rank, target_layer_names=None):
        """Enable Lora on the backbone.

        Calling this method will freeze all weights on the backbone,
        while enabling Lora on the query & value `EinsumDense` layers
        of the attention layers.

        Args:
            rank: The rank of the LoRA factorization.
            target_layer_names: A list of strings, the names of the layers to
                apply LoRA to. If `None`, this will be populated with the
                default LoRA layer names as returned by
                `backbone.default_lora_layer_names()`.
        """
        if target_layer_names is None:
            target_layer_names = self.default_lora_layer_names()
        self.trainable = True
        self._lora_enabled_layers = []
        self._lora_rank = rank
        for layer in self._flatten_layers(include_self=False):
            layer.trainable = False
        all_layers = self._flatten_layers(include_self=False)
        all_layers = [lyr for lyr in all_layers if lyr.weights]
        for i, layer in enumerate(all_layers):
            for name in target_layer_names:
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

    def enable_dora(self, rank, target_layer_names=None, lora_alpha=None):
        """Enable DoRA on the backbone.

        DoRA (Weight-Decomposed Low-Rank Adaptation, Liu et al., 2024)
        decomposes each target weight matrix into a magnitude vector and a
        directional matrix, and only trains the magnitude vector together with
        a LoRA update on the direction. Calling this method will freeze all
        weights on the backbone, enable LoRA on the target `EinsumDense`
        layers, and add a trainable `dora_magnitude` weight on each so that
        the effective kernel at forward time is
        `m * (W_0 + s * B @ A) / ||W_0 + s * B @ A||_c`, where `||.||_c` is
        the column-wise L2 norm of the combined kernel.

        Args:
            rank: int. The rank of the LoRA factorization used for the
                directional update.
            target_layer_names: list of strings, optional. The names of the
                layers to DoRA-fy. Defaults to
                `backbone.default_dora_layer_names()`.
            lora_alpha: int, optional. The alpha scaling factor for the LoRA
                update (scale = `lora_alpha / rank`). Defaults to `rank`,
                giving a scale of 1.

        Example:
        ```python
        backbone = keras_hub.models.GemmaBackbone.from_preset("gemma_2b_en")
        backbone.enable_dora(rank=4)
        ```
        """
        if target_layer_names is None:
            target_layer_names = self.default_dora_layer_names()
        self.trainable = True
        self._dora_enabled_layers = []
        self._dora_rank = rank
        self._dora_lora_alpha = lora_alpha if lora_alpha is not None else rank
        for layer in self._flatten_layers(include_self=False):
            layer.trainable = False
        all_layers = self._flatten_layers(include_self=False)
        all_layers = [lyr for lyr in all_layers if lyr.weights]
        for i, layer in enumerate(all_layers):
            for name in target_layer_names:
                if layer.name == name and hasattr(layer, "enable_lora"):
                    layer.trainable = True
                    _dorafy_einsum_dense_layer(
                        layer, rank, self._dora_lora_alpha
                    )
                    self._dora_enabled_layers.append(i)

    def save_dora_weights(self, filepath):
        """Save DoRA factors and magnitudes to `filepath`.

        The file must end in `.dora.h5`. Only the trained DoRA state
        (`lora_kernel_a`, `lora_kernel_b`, and `dora_magnitude` per enabled
        layer) is saved — the frozen base weights are not written.
        """
        if not getattr(self, "_dora_enabled_layers", []):
            raise ValueError(
                "There are no dora-enabled layers in this model. "
                "Make sure to call `.enable_dora(rank)` first."
            )
        if not str(filepath).endswith(".dora.h5"):
            raise ValueError(
                "The filename must end in `.dora.h5`. "
                f"Received: filepath={filepath}"
            )

        store = keras.src.saving.saving_lib.H5IOStore(filepath, mode="w")
        dora_store = store.make("dora")
        dora_store["rank"] = self._dora_rank
        dora_store["lora_alpha"] = self._dora_lora_alpha
        all_layers = self._flatten_layers(include_self=False)
        all_layers = [lyr for lyr in all_layers if lyr.weights]
        for layer_index in self._dora_enabled_layers:
            layer = all_layers[layer_index]
            inner_store = store.make(f"dora/{layer_index}")
            inner_store["lora_kernel_a"] = layer.lora_kernel_a
            inner_store["lora_kernel_b"] = layer.lora_kernel_b
            inner_store["dora_magnitude"] = layer.dora_magnitude
        store.close()

    def load_dora_weights(self, filepath):
        """Load DoRA factors and magnitudes from `filepath`.

        If DoRA has not been enabled yet, this will enable it with the rank
        recorded in the file.
        """
        store = keras.src.saving.saving_lib.H5IOStore(filepath, mode="r")
        dora_store = store.get("dora")
        rank = int(dora_store["rank"][()])
        lora_alpha = int(dora_store["lora_alpha"][()])

        if not getattr(self, "_dora_enabled_layers", []):
            self.enable_dora(rank, lora_alpha=lora_alpha)
        else:
            if self._dora_rank != rank:
                raise ValueError(
                    f"The DoRA rank expected by file '{filepath}' "
                    f"is rank={rank}, but the model was called with "
                    f"`.enable_dora(rank={self._dora_rank})`. "
                    "Both ranks must match."
                )
        all_layers = self._flatten_layers(include_self=False)
        all_layers = [lyr for lyr in all_layers if lyr.weights]
        for layer_index in self._dora_enabled_layers:
            layer = all_layers[layer_index]
            inner = store.get(f"dora/{layer_index}")
            layer.lora_kernel_a.assign(inner["lora_kernel_a"])
            layer.lora_kernel_b.assign(inner["lora_kernel_b"])
            layer.dora_magnitude.assign(inner["dora_magnitude"])
        store.close()

    def export_to_transformers(self, path):
        """Export the backbone model to HuggingFace Transformers format.

        This saves the backbone's configuration and weights in a format
        compatible with HuggingFace Transformers. For unsupported model
        architectures, a ValueError is raised.

        Args:
            path: str. Path to save the exported model.
        """
        from keras_hub.src.utils.transformers.export.hf_exporter import (
            export_backbone,
        )

        export_backbone(self, path)
