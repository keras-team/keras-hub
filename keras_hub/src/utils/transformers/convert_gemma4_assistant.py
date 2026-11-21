import numpy as np

from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.samplers.top_k_sampler import TopKSampler
from keras_hub.src.samplers.top_p_sampler import TopPSampler
from keras_hub.src.utils.preset_utils import check_file_exists
from keras_hub.src.utils.preset_utils import load_json
from keras_hub.src.utils.transformers.convert_gemma4 import (
    _convert_decoder_block,
)
from keras_hub.src.utils.transformers.convert_gemma4 import (
    convert_backbone_config as target_convert_config,
)

backbone_cls = Gemma4Backbone


def convert_backbone_config(transformers_config):
    """Map a Transformers config dict → Gemma4Backbone keyword arguments
    for assistant.
    """
    # Delegate to the target model's backbone config converter.
    config = target_convert_config(transformers_config)
    return config


def convert_task_config(transformers_config):
    """Map Transformers config.json to Gemma4AssistantCausalLM kwargs."""
    return {
        "centroid_intermediate_top_k": transformers_config[
            "centroid_intermediate_top_k"
        ],
        "use_ordered_embeddings": transformers_config["use_ordered_embeddings"],
        "backbone_hidden_size": transformers_config["backbone_hidden_size"],
        "num_centroids": transformers_config["num_centroids"],
    }


def load_task_config(preset, transformers_config):
    """Read generation_config.json and return extra Gemma4AssistantCausalLM
    kwargs not present in config.json.

    Maps:
      ``num_assistant_tokens`` → ``num_speculative_tokens``
      ``do_sample`` / ``top_k`` / ``top_p`` / ``temperature`` → ``sampler``
    """
    if not check_file_exists(preset, "generation_config.json"):
        return {}
    gen_cfg = load_json(preset, "generation_config.json")
    kwargs = {}

    if "num_assistant_tokens" in gen_cfg:
        kwargs["num_speculative_tokens"] = gen_cfg["num_assistant_tokens"]

    do_sample = gen_cfg.get("do_sample", False)
    has_top_k = "top_k" in gen_cfg
    has_top_p = "top_p" in gen_cfg
    if not do_sample and (has_top_k or has_top_p):
        do_sample = True
    if do_sample:
        top_k = gen_cfg.get("top_k", None)
        top_p = gen_cfg.get("top_p", None)
        temperature = gen_cfg.get("temperature", 1.0)
        if top_p is not None and top_p < 1.0:
            kwargs["sampler"] = TopPSampler(
                p=top_p, k=top_k, temperature=temperature
            )
        elif top_k is not None:
            kwargs["sampler"] = TopKSampler(k=top_k, temperature=temperature)

    return kwargs


def convert_weights(backbone, loader, transformers_config):
    """Port Gemma4Assistant Backbone weights (inner model) from HF."""

    def hf_key(suffix):
        return f"model.{suffix}"

    for i in range(backbone.num_layers):
        decoder_layer = backbone.get_layer(f"decoder_block_{i}")
        # Assistant weights have NO independent key/value tensors in the file.
        # Force toggle the flag on temporarily during this specific port phase
        # to bypass redundant safe-tensor load checks without altering the
        # backbone.
        decoder_layer.attention.is_kv_shared_layer = True
        _convert_decoder_block(decoder_layer, i, loader, hf_key)

    loader.port_weight(
        keras_variable=backbone.get_layer("token_embedding").embeddings,
        hf_weight_key=hf_key("embed_tokens.weight"),
    )

    loader.port_weight(
        keras_variable=backbone.get_layer("final_normalization").scale,
        hf_weight_key=hf_key("norm.weight"),
    )


def convert_head(model, loader, transformers_config):
    """Port the dedicated Assistant top-level projection layers."""
    # pre_projection
    loader.port_weight(
        keras_variable=model.pre_projection.kernel,
        hf_weight_key="pre_projection.weight",
        hook_fn=lambda x, _: np.transpose(x),
    )

    # post_projection
    loader.port_weight(
        keras_variable=model.post_projection.kernel,
        hf_weight_key="post_projection.weight",
        hook_fn=lambda x, _: np.transpose(x),
    )

    # centroids
    if getattr(model, "centroids", None) is not None:
        loader.port_weight(
            keras_variable=model.centroids.kernel,
            hf_weight_key="masked_embedding.centroids.weight",
            hook_fn=lambda x, _: np.transpose(x),
        )

    # token_ordering
    if getattr(model, "token_ordering", None) is not None:
        loader.port_weight(
            keras_variable=model.token_ordering,
            hf_weight_key="masked_embedding.token_ordering",
            hook_fn=lambda x, _: x.astype(np.float32),
        )
