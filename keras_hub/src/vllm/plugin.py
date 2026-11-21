"""vLLM plugin entry point that registers KerasHub's serving pieces.

vLLM calls every ``vllm.general_plugins`` entry point once at startup, in
the driver and in each worker process. Registering from here rather than
from the backend keeps the integration self-contained: keras-hub declares
what it provides, and tpu-inference stays unaware of it.
"""

from keras_hub.src.vllm.hf_config import KERAS_HUB_ARCHITECTURE
from keras_hub.src.vllm.hf_config import register_hf_config


def register_keras_hub():
    """Registers everything needed to serve KerasHub presets on vLLM.

    Four registrations, all through public interfaces:

    - `register_hf_config()` teaches transformers the `keras_hub`
      model_type, so `AutoConfig` can read the config KerasHub writes.
    - `register_model` puts `KerasHubVllmModel` in the backend's model
      registry under the `KerasHubForCausalLM` architecture string, so the
      loader resolves it like any other model.
    - `TokenizerRegistry` maps `tokenizer_mode="keras_hub"` to
      `KerasHubTokenizer`, so vLLM tokenizes with the preset's own tokenizer.
    - `RENDERER_REGISTRY` maps the same mode to vLLM's standard HF prompt
      renderer, which drives any registered tokenizer.

    Serving runs on tpu-inference's native JAX path, so without it there is
    nothing to register and this returns; vLLM continues unaffected.
    """
    # One block, because vLLM calls this during startup: an ImportError
    # that escapes here would take the engine down rather than leave the
    # integration unregistered.
    try:
        from tpu_inference.models.common.model_loader import register_model
        from vllm.renderers.registry import RENDERER_REGISTRY
        from vllm.tokenizers.registry import TokenizerRegistry

        from keras_hub.src.vllm.keras_hub_vllm_wrapper import KerasHubVllmModel
    except ImportError:
        return

    register_hf_config()
    register_model(KERAS_HUB_ARCHITECTURE, KerasHubVllmModel)
    TokenizerRegistry.register(
        "keras_hub", "keras_hub.src.vllm.tokenizer", "KerasHubTokenizer"
    )
    RENDERER_REGISTRY.register("keras_hub", "vllm.renderers.hf", "HfRenderer")
