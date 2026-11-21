"""Convert diffusers pipelines to KerasHub."""

import contextlib

from keras_hub.src.utils.diffusers import convert_flux
from keras_hub.src.utils.diffusers import convert_flux2
from keras_hub.src.utils.preset_utils import PresetLoader
from keras_hub.src.utils.preset_utils import check_file_exists
from keras_hub.src.utils.preset_utils import jax_memory_cleanup
from keras_hub.src.utils.preset_utils import load_json
from keras_hub.src.utils.transformers.safetensor_utils import SafetensorLoader

try:
    import safetensors
except ImportError:
    safetensors = None


class DiffusersSafetensorLoader(SafetensorLoader):
    """A `SafetensorLoader` over one component folder of a diffusers repo.

    Diffusers models (`transformer/`, `vae/`) are saved as
    `diffusion_pytorch_model.safetensors`, or sharded with a
    `diffusion_pytorch_model.safetensors.index.json`; transformers models
    (`text_encoder/`) as `model.safetensors` or with a
    `model.safetensors.index.json`.

    Only `subfolder` is probed. A diffusers repo keeps no weights at its
    root, so the root is never read: that saves a request per component on
    `hf://` repos, and a root `model.safetensors.index.json` cannot take
    precedence over the weights of the folder.

    Args:
        preset: The diffusers repo, e.g. `"hf://org/model"` or a local path.
        subfolder: The component folder, e.g. `"transformer"`.
    """

    def __init__(self, preset, subfolder):
        # `SafetensorLoader.__init__` is not called on purpose: it looks for a
        # `model.safetensors.index.json` at the root of `preset`.
        contextlib.ExitStack.__init__(self)
        if safetensors is None:
            raise ImportError(
                "Converting from the diffusers model format requires the "
                "safetensors package. Please install with "
                "`pip install safetensors`."
            )
        self.preset = preset
        self.prefix = None
        self.safetensor_config = None
        self.fname = None
        self.safetensor_files = {}
        self.safetensor_keys = {}
        self.safetensor_aliases = {}
        for name in ("diffusion_pytorch_model", "model"):
            index_file = f"{subfolder}/{name}.safetensors.index.json"
            if check_file_exists(preset, index_file):
                index = load_json(preset, index_file)
                weight_map = {
                    key: f"{subfolder}/{fname}"
                    for key, fname in index["weight_map"].items()
                }
                self.safetensor_config = {**index, "weight_map": weight_map}
                return
            fname = f"{subfolder}/{name}.safetensors"
            if check_file_exists(preset, fname):
                self.fname = fname
                return
        raise FileNotFoundError(
            f"`{subfolder}/` in preset `{preset}` has no safetensors weights."
        )


class DiffusersPresetLoader(PresetLoader):
    """Loads KerasHub models from diffusers pipeline repos.

    A diffusers repo has no root `config.json`. Its `model_index.json` lists
    the components of the pipeline as `"name": [library, class name]`, and
    each component has its own folder (`transformer/`, `vae/`,
    `text_encoder/`, `tokenizer/`, ...) with its config and weights.

    The backbone is the `transformer` component, and the `_class_name` of
    its config selects a converter module. As in `TransformersPresetLoader`,
    everything model-specific is delegated to that converter:

    - `backbone_cls`, `convert_backbone_config(transformer_config)` and
      `convert_weights(backbone, loader)` load the backbone.
    - `load_task_config(loader, load_weights, **kwargs)`, if defined, returns
      the task kwargs other than the backbone and the preprocessor, such as
      the other pipeline components. It receives this loader to build them
      with `load_component_config()` and `load_component_weights()`, and the
      `from_preset()` kwargs, which take precedence over what it returns.
    - `convert_tokenizer(cls, preset, **kwargs)`, if defined, loads the
      tokenizer.

    Without the last two, only the backbone can be loaded.
    """

    def __init__(self, preset, config):
        super().__init__(preset, config)
        self.pipeline_class = config.get("_class_name")
        # `model_index.json` lists the components, so pipelines without a
        # transformer (e.g. the UNet-based Stable Diffusion ones) are
        # rejected without probing their folders.
        if self._component_class("transformer") is None:
            components = sorted(
                name for name in config if self._component_class(name)
            )
            raise ValueError(
                "KerasHub has no converter for the diffusers pipeline "
                f"`{self.pipeline_class}` in preset `{preset}`. Only "
                "pipelines with a `transformer` component, such as FLUX, can "
                f"be loaded. Received components: {components}"
            )
        self.transformer_config = load_json(preset, "transformer/config.json")
        class_name = self.transformer_config.get("_class_name")
        if class_name == "FluxTransformer2DModel":
            self.converter = convert_flux
        elif class_name == "Flux2Transformer2DModel":
            self.converter = convert_flux2
        else:
            raise ValueError(
                "KerasHub has no converter for diffusers transformers of "
                f"class `'{class_name}'`."
            )

    def _component_class(self, name):
        """The class `model_index.json` lists for component `name`, or `None`.

        A pipeline without the component omits it or, for an optional
        component, lists it as `null` or `[null, null]`.
        """
        entry = self.config.get(name)
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            return entry[1]
        return None

    def _check_component(self, name, expected_class):
        class_name = self._component_class(name)
        if class_name is None:
            raise ValueError(
                f"KerasHub needs a `{expected_class}` as the `{name}` of the "
                f"diffusers pipeline `{self.pipeline_class}`, but its "
                f"`model_index.json` has no `{name}` component. Received: "
                f"{name}={self.config.get(name)!r}"
            )
        if class_name != expected_class:
            raise ValueError(
                f"KerasHub can only load a `{expected_class}` as the `{name}` "
                f"of a diffusers pipeline. Received: `{class_name}`"
            )

    def _check_converter_has(self, hook, cls):
        if not hasattr(self.converter, hook):
            raise ValueError(
                "KerasHub can only load the backbone "
                f"(`{self.converter.backbone_cls.__name__}`) of the diffusers "
                f"pipeline `{self.pipeline_class}`; loading a `{cls.__name__}` "
                "from it is not supported yet."
            )

    def load_component_config(self, name, expected_class):
        """Load the config of a pipeline component.

        Args:
            name: The key of the component in `model_index.json`, which is
                also the name of its folder, e.g. `"vae"`.
            expected_class: The class `model_index.json` must list for the
                component, e.g. `"AutoencoderKLFlux2"`.
        """
        self._check_component(name, expected_class)
        return load_json(self.preset, f"{name}/config.json")

    def load_component_weights(self, name, model, convert_weights):
        """Load the weights of a pipeline component into a Keras model.

        Args:
            name: The folder of the component, e.g. `"vae"`.
            model: The Keras model built for the component.
            convert_weights: Called as `convert_weights(model, loader)`, with
                a `DiffusersSafetensorLoader` over the folder.
        """
        jax_memory_cleanup(model)
        with DiffusersSafetensorLoader(self.preset, name) as loader:
            convert_weights(model, loader)

    def check_backbone_class(self):
        return self.converter.backbone_cls

    def load_backbone(self, cls, load_weights, **kwargs):
        keras_config = self.converter.convert_backbone_config(
            self.transformer_config
        )
        backbone = cls(**{**keras_config, **kwargs})
        if load_weights:
            self.load_component_weights(
                "transformer", backbone, self.converter.convert_weights
            )
        return backbone

    def load_task(self, cls, load_weights, load_task_weights, **kwargs):
        self._check_converter_has("load_task_config", cls)
        task_config = self.converter.load_task_config(
            self, load_weights, **kwargs
        )
        kwargs = {**task_config, **kwargs}
        return super().load_task(cls, load_weights, load_task_weights, **kwargs)

    def load_tokenizer(self, cls, config_file=None, **kwargs):
        self._check_converter_has("convert_tokenizer", cls)
        return self.converter.convert_tokenizer(cls, self.preset, **kwargs)
