import numpy as np

from keras_hub.src.models.mistral3.mistral3_backbone import Mistral3Backbone
from keras_hub.src.models.mistral3.mistral3_vision_encoder import (
    Mistral3MultiModalProjector,
)
from keras_hub.src.models.mistral3.mistral3_vision_encoder import (
    Mistral3VisionEncoder,
)
from keras_hub.src.utils.preset_utils import check_file_exists
from keras_hub.src.utils.preset_utils import get_file
from keras_hub.src.utils.preset_utils import load_json
from keras_hub.src.utils.transformers.convert_mistral import (
    _convert_tekken_tokenizer as _convert_mistral_tekken_tokenizer,
)
from keras_hub.src.utils.transformers.convert_mistral import (
    convert_backbone_config as convert_text_backbone_config,
)

backbone_cls = Mistral3Backbone


_PIXTRAL_DEFAULT_RESCALE_FACTOR = 1 / 255


def _get_rope_theta(config, default=10000.0):
    rope_theta = config.get("rope_parameters", {}).get("rope_theta")
    if rope_theta is None:
        rope_theta = config.get("rope_theta", default)
    return rope_theta


def _convert_tekken_tokenizer(path):
    """Like `convert_mistral._convert_tekken_tokenizer`, but also returns
    `control_tokens`: the reserved-block special-token pieces, which
    `Mistral3Tokenizer` registers as unsplittable alongside its vision
    tokens.
    """
    try:
        from mistral_common.tokens.tokenizers.tekken import Tekkenizer
    except ImportError:
        raise ImportError(
            "Converting a Tekken (`tekken.json`) tokenizer requires the "
            "`mistral_common` package. Please install it with "
            "`pip install mistral-common`."
        )

    vocabulary, merges, split_pattern = _convert_mistral_tekken_tokenizer(path)

    tokenizer = Tekkenizer.from_file(path)
    control_tokens = [
        tokenizer.id_to_piece(rank)
        for rank in range(tokenizer.num_special_tokens)
    ]

    return vocabulary, merges, split_pattern, control_tokens


def _load_pixtral_defaults_from_mistral_common():
    # Some checkpoints (e.g. Mistral Small 3.2) ship no
    # `preprocessor_config.json`; fall back to `mistral_common`'s fixed
    # constants instead of duplicating the numbers here.
    try:
        from mistral_common.tokens.tokenizers.image import DATASET_MEAN
        from mistral_common.tokens.tokenizers.image import DATASET_STD
    except ImportError:
        raise ImportError(
            "Converting a Mistral3 checkpoint with no "
            "`preprocessor_config.json` requires the `mistral_common` "
            "package. Please install it with `pip install mistral-common`."
        )
    return list(DATASET_MEAN), list(DATASET_STD)


def load_image_converter_config(preset, transformers_config):
    vision_config = transformers_config["vision_config"]
    if check_file_exists(preset, "preprocessor_config.json"):
        preprocessor_config = load_json(preset, "preprocessor_config.json")
        mean = preprocessor_config["image_mean"]
        std = preprocessor_config["image_std"]
        rescale_factor = preprocessor_config["rescale_factor"]
        patch_size = preprocessor_config["patch_size"]
        if isinstance(patch_size, dict):
            patch_size = patch_size.get("height") or patch_size.get("width")
        size = preprocessor_config["size"]
        longest_edge = (
            size.get("longest_edge") if isinstance(size, dict) else None
        )
    else:
        mean, std = _load_pixtral_defaults_from_mistral_common()
        rescale_factor = _PIXTRAL_DEFAULT_RESCALE_FACTOR
        patch_size = vision_config["patch_size"]
        longest_edge = vision_config["image_size"]

    config = {}
    if mean is not None and std is not None:
        config["offset"] = [-m / s for m, s in zip(mean, std)]
        config["scale"] = [rescale_factor / s for s in std]
    if patch_size is not None:
        config["patch_size"] = patch_size
    if longest_edge is not None:
        config["longest_edge"] = longest_edge
    config["spatial_merge_size"] = transformers_config["spatial_merge_size"]
    return config


def convert_backbone_config(transformers_config):
    text_config = transformers_config["text_config"]
    vision_config = transformers_config["vision_config"]
    backbone_config = convert_text_backbone_config(text_config)

    vision_hidden_dim = vision_config["hidden_size"]
    vision_num_heads = vision_config["num_attention_heads"]
    vision_head_dim = vision_config.get("head_dim") or (
        vision_hidden_dim // vision_num_heads
    )
    vision_image_size = vision_config["image_size"]
    vision_patch_size = vision_config["patch_size"]
    vision_encoder = Mistral3VisionEncoder(
        image_size=vision_image_size,
        patch_size=vision_patch_size,
        num_channels=vision_config["num_channels"],
        hidden_dim=vision_hidden_dim,
        num_layers=vision_config["num_hidden_layers"],
        num_heads=vision_num_heads,
        head_dim=vision_head_dim,
        intermediate_dim=vision_config["intermediate_size"],
        rope_theta=_get_rope_theta(vision_config),
        layer_norm_epsilon=vision_config.get("rms_norm_eps", 1e-5),
        activation=vision_config["hidden_act"],
        attention_dropout=vision_config["attention_dropout"],
    )

    multimodal_projector = Mistral3MultiModalProjector(
        vision_hidden_dim=vision_hidden_dim,
        text_hidden_dim=text_config["hidden_size"],
        spatial_merge_size=transformers_config["spatial_merge_size"],
        patch_size=vision_patch_size,
        layer_norm_epsilon=text_config["rms_norm_eps"],
        projector_hidden_act=transformers_config["projector_hidden_act"],
        multimodal_projector_bias=transformers_config[
            "multimodal_projector_bias"
        ],
        image_size=vision_image_size,
    )

    image_token_index = transformers_config["image_token_index"]

    backbone_config.update(
        {
            "vision_encoder": vision_encoder,
            "multimodal_projector": multimodal_projector,
            "image_token_index": image_token_index,
        }
    )
    return backbone_config


def _port_text_weights(backbone, loader, tie_word_embeddings):
    # Embeddings
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="language_model.model.embed_tokens.weight",
        hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float32),
    )
    # When embeddings are tied, `lm_head.weight` is not saved as a separate
    # tensor in the checkpoint; reuse the embedding weights instead.
    lm_head_key = (
        "language_model.model.embed_tokens.weight"
        if tie_word_embeddings
        else "lm_head.weight"
    )
    loader.port_weight(
        keras_variable=backbone.token_embedding.reverse_embeddings,
        hf_weight_key=lm_head_key,
        hook_fn=lambda hf_tensor, _: np.transpose(
            hf_tensor.astype(np.float32), axes=(1, 0)
        ),
    )

    # Attention blocks
    for index in range(backbone.num_layers):
        decoder_layer = backbone.transformer_layers[index]

        # Norm layers
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layernorm.scale,
            hf_weight_key=f"language_model.model.layers.{index}.input_layernorm.weight",
            hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float32),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layernorm.scale,
            hf_weight_key=(
                f"language_model.model.layers.{index}.post_attention_layernorm.weight"
            ),
            hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float32),
        )

        # Attention layers
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._query_dense.kernel,
            hf_weight_key=f"language_model.model.layers.{index}.self_attn.q_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor.astype(np.float32)), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._key_dense.kernel,
            hf_weight_key=f"language_model.model.layers.{index}.self_attn.k_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor.astype(np.float32)), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._value_dense.kernel,
            hf_weight_key=f"language_model.model.layers.{index}.self_attn.v_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor.astype(np.float32)), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._output_dense.kernel,
            hf_weight_key=f"language_model.model.layers.{index}.self_attn.o_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor.astype(np.float32)), keras_shape
            ),
        )

        # MLP layers
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_gate_dense.kernel,
            hf_weight_key=f"language_model.model.layers.{index}.mlp.gate_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor.astype(np.float32), axes=(1, 0)
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_intermediate_dense.kernel,
            hf_weight_key=f"language_model.model.layers.{index}.mlp.up_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor.astype(np.float32), axes=(1, 0)
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_output_dense.kernel,
            hf_weight_key=f"language_model.model.layers.{index}.mlp.down_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor.astype(np.float32), axes=(1, 0)
            ),
        )

    # Normalization
    loader.port_weight(
        keras_variable=backbone.layer_norm.scale,
        hf_weight_key="language_model.model.norm.weight",
        hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float32),
    )


def _port_vision_weights(backbone, loader):
    vision_encoder = backbone.vision_encoder
    projector = backbone.multimodal_projector

    loader.port_weight(
        keras_variable=vision_encoder.patch_conv.kernel,
        hf_weight_key="vision_tower.patch_conv.weight",
        hook_fn=lambda hf_tensor, _: np.transpose(
            hf_tensor.astype(np.float32), axes=(2, 3, 1, 0)
        ),
    )
    loader.port_weight(
        keras_variable=vision_encoder.ln_pre.scale,
        hf_weight_key="vision_tower.ln_pre.weight",
        hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float32),
    )

    for index in range(vision_encoder.num_layers):
        layer = vision_encoder.transformer_layers[index]
        layer_prefix = f"vision_tower.transformer.layers.{index}"

        loader.port_weight(
            keras_variable=layer.attention_norm.scale,
            hf_weight_key=f"{layer_prefix}.attention_norm.weight",
            hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float32),
        )
        loader.port_weight(
            keras_variable=layer.ffn_norm.scale,
            hf_weight_key=f"{layer_prefix}.ffn_norm.weight",
            hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float32),
        )

        loader.port_weight(
            keras_variable=layer.attention.q_proj.kernel,
            hf_weight_key=f"{layer_prefix}.attention.q_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor.astype(np.float32), axes=(1, 0)
            ),
        )
        loader.port_weight(
            keras_variable=layer.attention.k_proj.kernel,
            hf_weight_key=f"{layer_prefix}.attention.k_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor.astype(np.float32), axes=(1, 0)
            ),
        )
        loader.port_weight(
            keras_variable=layer.attention.v_proj.kernel,
            hf_weight_key=f"{layer_prefix}.attention.v_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor.astype(np.float32), axes=(1, 0)
            ),
        )
        loader.port_weight(
            keras_variable=layer.attention.o_proj.kernel,
            hf_weight_key=f"{layer_prefix}.attention.o_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor.astype(np.float32), axes=(1, 0)
            ),
        )

        loader.port_weight(
            keras_variable=layer.feed_forward.gate_proj.kernel,
            hf_weight_key=f"{layer_prefix}.feed_forward.gate_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor.astype(np.float32), axes=(1, 0)
            ),
        )
        loader.port_weight(
            keras_variable=layer.feed_forward.up_proj.kernel,
            hf_weight_key=f"{layer_prefix}.feed_forward.up_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor.astype(np.float32), axes=(1, 0)
            ),
        )
        loader.port_weight(
            keras_variable=layer.feed_forward.down_proj.kernel,
            hf_weight_key=f"{layer_prefix}.feed_forward.down_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor.astype(np.float32), axes=(1, 0)
            ),
        )

    # Multimodal projector
    loader.port_weight(
        keras_variable=projector.norm.scale,
        hf_weight_key="multi_modal_projector.norm.weight",
        hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float32),
    )
    loader.port_weight(
        keras_variable=projector.patch_merger.merging_layer.kernel,
        hf_weight_key="multi_modal_projector.patch_merger.merging_layer.weight",
        hook_fn=lambda hf_tensor, _: np.transpose(
            hf_tensor.astype(np.float32), axes=(1, 0)
        ),
    )
    loader.port_weight(
        keras_variable=projector.linear_1.kernel,
        hf_weight_key="multi_modal_projector.linear_1.weight",
        hook_fn=lambda hf_tensor, _: np.transpose(
            hf_tensor.astype(np.float32), axes=(1, 0)
        ),
    )
    loader.port_weight(
        keras_variable=projector.linear_2.kernel,
        hf_weight_key="multi_modal_projector.linear_2.weight",
        hook_fn=lambda hf_tensor, _: np.transpose(
            hf_tensor.astype(np.float32), axes=(1, 0)
        ),
    )
    if projector.linear_1.use_bias:
        loader.port_weight(
            keras_variable=projector.linear_1.bias,
            hf_weight_key="multi_modal_projector.linear_1.bias",
            hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float32),
        )
        loader.port_weight(
            keras_variable=projector.linear_2.bias,
            hf_weight_key="multi_modal_projector.linear_2.bias",
            hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float32),
        )


def convert_weights(backbone, loader, transformers_config):
    tie_word_embeddings = transformers_config.get("tie_word_embeddings", False)
    _port_text_weights(
        backbone,
        loader,
        tie_word_embeddings=tie_word_embeddings,
    )
    _port_vision_weights(backbone, loader)


def convert_tokenizer(cls, preset, **kwargs):
    # Mistral3 checkpoints always ship a Tekken (byte-level BPE) `tekken.json`
    # tokenizer; there is currently no SentencePiece Mistral3 preset to
    # support.
    if not check_file_exists(preset, "tekken.json"):
        raise ValueError(
            f"Could not find a `tekken.json` file for preset '{preset}'. "
            "Mistral3 checkpoint conversion currently only supports Tekken "
            "(byte-level BPE) tokenizers."
        )
    tekken_path = get_file(preset, "tekken.json")
    vocabulary, merges, split_pattern, control_tokens = (
        _convert_tekken_tokenizer(tekken_path)
    )
    return cls(
        vocabulary=vocabulary,
        merges=merges,
        split_pattern=split_pattern,
        control_tokens=control_tokens,
        **kwargs,
    )
