"""A neutral HF config so vLLM can serve any KerasHub preset by name.

vLLM validates ``config.json`` through transformers' ``AutoConfig`` (keyed by
``model_type``) and through its own model registry (keyed by ``architectures``).
Neither ships a generic decoder entry, so rather than borrow a real model's name
we define our own: ``model_type = "keras_hub"`` / architecture
``"KerasHubForCausalLM"``.

This config is never used to build a model. tpu-inference's native loader routes
any config carrying ``keras_hub_preset`` to its ``KerasHubVllmModel`` model
class (the real KerasHub backbone on the flax/nnx path); this object only
carries the dims vLLM reads for KV-cache profiling, which
``setup_vllm_model`` fills from each preset's own config.
"""

import logging

KERAS_HUB_ARCHITECTURE = "KerasHubForCausalLM"
KERAS_HUB_MODEL_TYPE = "keras_hub"


_config_class = None


def _build_config_class():
    """Defines ``KerasHubConfig`` once, lazily, and caches it.

    Kept out of import time because it pulls in transformers; callers reach it
    through ``register_hf_config``. Cached so every call (the driver, then
    tpu-inference's plugin hook in each worker) returns the same class object
    and re-registration is a true no-op.
    """
    global _config_class
    if _config_class is not None:
        return _config_class

    from transformers import PretrainedConfig

    class KerasHubConfig(PretrainedConfig):
        """Carrier config for a KerasHub preset served on vLLM-TPU.

        Subclasses ``PretrainedConfig`` (no model family) and only holds the
        fields vLLM reads for KV-cache profiling. Defaults are placeholders;
        ``setup_vllm_model`` overwrites them with the preset's real dims.
        """

        model_type = KERAS_HUB_MODEL_TYPE

        def __init__(
            self,
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=None,
            head_dim=None,
            intermediate_size=11008,
            max_position_embeddings=8192,
            sliding_window=None,
            **kwargs,
        ):
            self.vocab_size = vocab_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.num_key_value_heads = (
                num_key_value_heads
                if num_key_value_heads is not None
                else num_attention_heads
            )
            self.head_dim = (
                head_dim
                if head_dim is not None
                else hidden_size // num_attention_heads
            )
            self.intermediate_size = intermediate_size
            self.max_position_embeddings = max_position_embeddings
            self.sliding_window = sliding_window
            super().__init__(**kwargs)

    _config_class = KerasHubConfig
    return KerasHubConfig


def register_hf_config():
    """Registers ``KerasHubConfig`` with transformers' ``AutoConfig``.

    Idempotent and best-effort: safe to call from ``setup_vllm_model`` (the
    process that writes the config) and again from tpu-inference's import in
    each serving worker, so ``AutoConfig.from_pretrained`` resolves
    ``model_type = "keras_hub"`` wherever the config is loaded.

    Returns:
        The registered ``KerasHubConfig`` class, or ``None`` when
        transformers is unavailable.
    """
    try:
        from transformers import AutoConfig
    except ImportError as e:  # pragma: no cover - transformers is a soft dep
        logging.warning(
            "transformers unavailable; cannot register %s (%s).",
            KERAS_HUB_MODEL_TYPE,
            e,
        )
        return None

    config_cls = _build_config_class()
    try:
        AutoConfig.register(KERAS_HUB_MODEL_TYPE, config_cls)
    except ValueError:
        # Registered under a different class object (e.g. two copies of
        # keras-hub in one process) — the cached class makes this rare.
        pass
    return config_cls
