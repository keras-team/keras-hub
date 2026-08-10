"""vLLM plugin entry point that registers KerasHub's serving pieces.

vLLM calls every ``vllm.general_plugins`` entry point once at startup, in
the driver and in each worker process. Registering from here rather than
from the backend keeps the integration self-contained: keras-hub declares
what it provides, and the backends stay unaware of it.
"""

# vLLM is optional, and this module is imported wherever the entry point is
# discovered. Same guarded-import pattern as `registry.py`.
try:
    from vllm.model_executor.model_loader import BaseModelLoader
except ImportError:
    BaseModelLoader = object

from keras_hub.src.vllm.hf_config import KERAS_HUB_ARCHITECTURE
from keras_hub.src.vllm.hf_config import register_hf_config


class KerasHubPresetLoader(BaseModelLoader):
    """The `keras_hub` load format: nothing left to load.

    `KerasHubTorchModel` builds its backbone with `CausalLM.from_preset`,
    which brings the weights with it, so both methods here are empty. The
    format exists because vLLM always runs a loader and none of the built
    ones fit: they stream weight files, which a KerasHub model directory
    does not have, and `dummy` would overwrite the preset with random
    values.

    Defined here rather than beside the model so that registering it does
    not import torch and the KerasHub model stack into every vLLM process.
    """

    def download_model(self, model_config):
        pass

    def load_weights(self, model, model_config):
        pass


def register_keras_hub():
    """Registers everything needed to serve KerasHub presets on vLLM.

    The serving model differs by engine — `KerasHubVllmModel` on
    tpu-inference's flax/nnx path, `KerasHubTorchModel` on vLLM's own torch
    engine — so the model registration branches on the platform this
    process serves on. Everything else is common:

    - `register_hf_config()` teaches transformers the `keras_hub`
      model_type, so `AutoConfig` can read the config KerasHub writes.
    - `TokenizerRegistry` maps `tokenizer_mode="keras_hub"` to
      `KerasHubTokenizer`, so vLLM tokenizes with the preset's own tokenizer.
    - `RENDERER_REGISTRY` maps the same mode to vLLM's standard HF prompt
      renderer, which drives any registered tokenizer.

    The GPU path additionally registers the `keras_hub` load format, which
    keeps the stock loaders away from a model directory that holds only a
    config (see `KerasHubPresetLoader`).

    Without vLLM there is nothing to register and this returns; vLLM
    continues unaffected.
    """
    # One block, because vLLM calls this during startup: an ImportError
    # that escapes here would take the engine down rather than leave the
    # integration unregistered.
    try:
        from vllm.renderers.registry import RENDERER_REGISTRY
        from vllm.tokenizers.registry import TokenizerRegistry
    except ImportError:
        return

    register_hf_config()
    TokenizerRegistry.register(
        "keras_hub", "keras_hub.src.vllm.tokenizer", "KerasHubTokenizer"
    )
    RENDERER_REGISTRY.register("keras_hub", "vllm.renderers.hf", "HfRenderer")

    # Both engines claim the same architecture string, so exactly one
    # registers it. The hardware decides, not which packages happen to be
    # installed: tpu-inference can sit in a GPU image without serving
    # anything there.
    if serves_on_tpu():
        _register_tpu_model()
    else:
        _register_gpu_model()


def serves_on_tpu():
    """Whether this process serves on TPU.

    Only breaks the tie when tpu-inference is installed somewhere that is
    not a TPU. If vLLM cannot answer -- an older version, or the platform
    not resolved at plugin time -- this falls back to whether the TPU
    backend is installed at all, which is what decided this before the GPU
    path existed. A tie-breaker must not leave either path unregistered.

    Returns:
        Whether this process serves on TPU.
    """
    try:
        from vllm.platforms import current_platform

        return current_platform.is_tpu()
    except Exception:
        pass
    try:
        import tpu_inference  # noqa: F401

        return True
    except ImportError:
        return False


def _register_tpu_model():
    """Registers the serving model for tpu-inference's flax/nnx path."""
    try:
        from tpu_inference.models.common.model_loader import register_model

        from keras_hub.src.vllm.keras_hub_vllm_wrapper import KerasHubVllmModel
    except ImportError:
        return

    register_model(KERAS_HUB_ARCHITECTURE, KerasHubVllmModel)


def _register_gpu_model():
    """Registers the serving model and load format for vLLM's torch engine.

    The model goes in by module path, so torch and the KerasHub model stack
    import only when the engine resolves the architecture — which is why
    `KerasHubPresetLoader` lives here rather than beside the model class.
    """
    if BaseModelLoader is object:
        # The guarded import at the top of this module failed, so the
        # loader is not a real `BaseModelLoader` and vLLM would reject it
        # with a ValueError -- out of a plugin, that stops the engine
        # starting at all. Leave the path unregistered instead.
        return
    try:
        from vllm import ModelRegistry
        from vllm.model_executor.model_loader import register_model_loader
    except ImportError:
        return

    ModelRegistry.register_model(
        KERAS_HUB_ARCHITECTURE,
        "keras_hub.src.vllm.torch_wrapper:KerasHubTorchModel",
    )
    register_model_loader("keras_hub")(KerasHubPresetLoader)
