"""vLLM plugin entry point that registers KerasHub's serving pieces.

vLLM calls every ``vllm.general_plugins`` entry point once at startup, in
the driver and in each worker process. Registering from here rather than
from the backend keeps the integration self-contained: keras-hub declares
what it provides, and the backends stay unaware of it.
"""

from keras_hub.src.vllm.hf_config import KERAS_HUB_ARCHITECTURE
from keras_hub.src.vllm.hf_config import register_hf_config


def register_keras_hub():
    """Registers everything needed to serve KerasHub presets on vLLM.

    The serving model differs by engine — `KerasHubVllmModel` on
    tpu-inference's flax/nnx path, `KerasHubTorchModel` on vLLM's own torch
    engine — so the model registration branches on which backend is
    installed, with TPU taking precedence. Everything else is common:

    - `register_hf_config()` teaches transformers the `keras_hub`
      model_type, so `AutoConfig` can read the config KerasHub writes.
    - `TokenizerRegistry` maps `tokenizer_mode="keras_hub"` to
      `KerasHubTokenizer`, so vLLM tokenizes with the preset's own tokenizer.
    - `RENDERER_REGISTRY` maps the same mode to vLLM's standard HF prompt
      renderer, which drives any registered tokenizer.

    The GPU path additionally registers the `keras_hub` load format, whose
    loader fills the model from the preset (the model directory carries no
    weight files for the stock loaders to stream).

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

    try:
        from tpu_inference.models.common.model_loader import register_model

        from keras_hub.src.vllm.keras_hub_vllm_wrapper import KerasHubVllmModel
    except ImportError:
        _register_torch_model()
        return
    register_model(KERAS_HUB_ARCHITECTURE, KerasHubVllmModel)


def _register_torch_model():
    """Registers the GPU serving pieces with stock vLLM.

    The model class is registered by module path, so torch and the wrapper
    only import when the engine actually resolves the architecture; the
    load-format registration imports the loader class directly, which is
    fine inside an engine process.
    """
    try:
        from vllm import ModelRegistry
        from vllm.model_executor.model_loader import register_model_loader

        from keras_hub.src.vllm.torch_wrapper import KerasHubPresetLoader
    except ImportError:
        return

    ModelRegistry.register_model(
        KERAS_HUB_ARCHITECTURE,
        "keras_hub.src.vllm.torch_wrapper:KerasHubTorchModel",
    )
    register_model_loader("keras_hub")(KerasHubPresetLoader)
