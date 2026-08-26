import numpy as np

from keras_hub.src.models.mistral.mistral_backbone import MistralBackbone
from keras_hub.src.models.mistral.mistral_tokenizer import (
    MistralTekkenTokenizer,
)
from keras_hub.src.models.mistral.mistral_vision_encoder import (
    Mistral3MultiModalProjector,
)
from keras_hub.src.models.mistral.mistral_vision_encoder import (
    Mistral3VisionEncoder,
)
from keras_hub.src.utils.preset_utils import check_file_exists
from keras_hub.src.utils.preset_utils import get_file
from keras_hub.src.utils.preset_utils import load_json

backbone_cls = MistralBackbone


_PIXTRAL_DEFAULT_RESCALE_FACTOR = 1 / 255


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
    if "vision_config" not in transformers_config:
        return None
    vision_config = transformers_config["vision_config"]
    if check_file_exists(preset, "preprocessor_config.json"):
        preprocessor_config = load_json(preset, "preprocessor_config.json")
        mean = preprocessor_config.get("image_mean")
        std = preprocessor_config.get("image_std")
        rescale_factor = preprocessor_config.get("rescale_factor", 1 / 255)
        patch_size = preprocessor_config.get("patch_size")
        if isinstance(patch_size, dict):
            patch_size = patch_size.get("height") or patch_size.get("width")
        size = preprocessor_config.get("size")
        longest_edge = (
            size.get("longest_edge") if isinstance(size, dict) else None
        )
    else:
        mean, std = _load_pixtral_defaults_from_mistral_common()
        rescale_factor = _PIXTRAL_DEFAULT_RESCALE_FACTOR
        patch_size = vision_config.get("patch_size")
        longest_edge = vision_config.get("image_size")

    config = {}
    if mean is not None and std is not None:
        config["offset"] = [-m / s for m, s in zip(mean, std)]
        config["scale"] = [rescale_factor / s for s in std]
    if patch_size is not None:
        config["patch_size"] = patch_size
    if longest_edge is not None:
        config["longest_edge"] = longest_edge
    config["spatial_merge_size"] = transformers_config.get(
        "spatial_merge_size", 2
    )
    return config


def _get_rope_theta(config, default=10000.0):
    rope_theta = config.get("rope_parameters", {}).get("rope_theta")
    if rope_theta is None:
        rope_theta = config.get("rope_theta", default)
    return rope_theta


def _convert_text_backbone_config(text_config):
    return {
        "vocabulary_size": text_config["vocab_size"],
        "num_layers": text_config["num_hidden_layers"],
        "num_query_heads": text_config["num_attention_heads"],
        "hidden_dim": text_config["hidden_size"],
        "intermediate_dim": text_config["intermediate_size"],
        "num_key_value_heads": text_config["num_key_value_heads"],
        "rope_max_wavelength": _get_rope_theta(text_config),
        "layer_norm_epsilon": text_config["rms_norm_eps"],
        "sliding_window": text_config.get("sliding_window"),
        "head_dim": text_config.get("head_dim"),
    }


def _convert_mistral3_backbone_config(transformers_config):
    text_config = transformers_config["text_config"]
    vision_config = transformers_config["vision_config"]
    backbone_config = _convert_text_backbone_config(text_config)

    vision_hidden_dim = vision_config["hidden_size"]
    vision_num_heads = vision_config["num_attention_heads"]
    vision_head_dim = vision_config.get("head_dim") or (
        vision_hidden_dim // vision_num_heads
    )
    vision_image_size = vision_config.get("image_size", 1540)
    vision_patch_size = vision_config.get("patch_size", 14)
    vision_encoder = Mistral3VisionEncoder(
        image_size=vision_image_size,
        patch_size=vision_patch_size,
        num_channels=vision_config.get("num_channels", 3),
        hidden_dim=vision_hidden_dim,
        num_layers=vision_config["num_hidden_layers"],
        num_heads=vision_num_heads,
        head_dim=vision_head_dim,
        intermediate_dim=vision_config["intermediate_size"],
        rope_theta=_get_rope_theta(vision_config),
        layer_norm_epsilon=vision_config.get("rms_norm_eps", 1e-5),
        activation=vision_config.get("hidden_act", "gelu"),
        attention_dropout=vision_config.get("attention_dropout", 0.0),
    )

    multimodal_projector = Mistral3MultiModalProjector(
        vision_hidden_dim=vision_hidden_dim,
        text_hidden_dim=text_config["hidden_size"],
        spatial_merge_size=transformers_config.get("spatial_merge_size", 2),
        patch_size=vision_patch_size,
        layer_norm_epsilon=text_config.get("rms_norm_eps", 1e-6),
        projector_hidden_act=transformers_config.get(
            "projector_hidden_act", "gelu"
        ),
        multimodal_projector_bias=transformers_config.get(
            "multimodal_projector_bias", False
        ),
        image_size=vision_image_size,
    )

    image_token_index = transformers_config.get(
        "image_token_index", transformers_config.get("image_token_id", 10)
    )

    backbone_config.update(
        {
            "vision_encoder": vision_encoder,
            "multimodal_projector": multimodal_projector,
            "image_token_index": image_token_index,
        }
    )
    return backbone_config


def convert_backbone_config(transformers_config):
    if "vision_config" in transformers_config:
        return _convert_mistral3_backbone_config(transformers_config)
    return _convert_text_backbone_config(transformers_config)


def _port_text_weights(backbone, loader, prefix, tie_word_embeddings):
    # Embeddings
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key=f"{prefix}.embed_tokens.weight",
        hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float32),
    )
    # When embeddings are tied, `lm_head.weight` is not saved as a separate
    # tensor in the checkpoint; reuse the embedding weights instead.
    lm_head_key = (
        f"{prefix}.embed_tokens.weight"
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
            hf_weight_key=f"{prefix}.layers.{index}.input_layernorm.weight",
            hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float32),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layernorm.scale,
            hf_weight_key=(
                f"{prefix}.layers.{index}.post_attention_layernorm.weight"
            ),
            hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float32),
        )

        # Attention layers
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._query_dense.kernel,
            hf_weight_key=f"{prefix}.layers.{index}.self_attn.q_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor.astype(np.float32)), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._key_dense.kernel,
            hf_weight_key=f"{prefix}.layers.{index}.self_attn.k_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor.astype(np.float32)), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._value_dense.kernel,
            hf_weight_key=f"{prefix}.layers.{index}.self_attn.v_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor.astype(np.float32)), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._output_dense.kernel,
            hf_weight_key=f"{prefix}.layers.{index}.self_attn.o_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor.astype(np.float32)), keras_shape
            ),
        )

        # MLP layers
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_gate_dense.kernel,
            hf_weight_key=f"{prefix}.layers.{index}.mlp.gate_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor.astype(np.float32), axes=(1, 0)
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_intermediate_dense.kernel,
            hf_weight_key=f"{prefix}.layers.{index}.mlp.up_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor.astype(np.float32), axes=(1, 0)
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_output_dense.kernel,
            hf_weight_key=f"{prefix}.layers.{index}.mlp.down_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor.astype(np.float32), axes=(1, 0)
            ),
        )

    # Normalization
    loader.port_weight(
        keras_variable=backbone.layer_norm.scale,
        hf_weight_key=f"{prefix}.norm.weight",
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
    if backbone.text_only_model:
        _port_text_weights(
            backbone,
            loader,
            prefix="model",
            tie_word_embeddings=tie_word_embeddings,
        )
    else:
        _port_text_weights(
            backbone,
            loader,
            prefix="language_model.model",
            tie_word_embeddings=tie_word_embeddings,
        )
        _port_vision_weights(backbone, loader)


def _bytes_to_unicode():
    """Return the GPT-2 reversible byte-to-unicode mapping.

    This is the same mapping used by `BytePairTokenizer`. It maps every byte
    value to a printable unicode character so byte sequences can be stored as
    strings in a vocabulary and merges file.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _bpe_split(mergeable_ranks, token, max_rank):
    """Replay tiktoken BPE on `token` using only lower-ranked merges.

    Returns the pieces the byte sequence `token` decomposes into when only
    pairs with rank strictly less than `max_rank` are allowed to merge. For a
    token that is itself the result of a single merge, this returns the two
    pieces that were merged to create it.
    """
    parts = [bytes([b]) for b in token]
    while True:
        min_rank = None
        min_idx = None
        for i in range(len(parts) - 1):
            rank = mergeable_ranks.get(parts[i] + parts[i + 1])
            if rank is not None and rank < max_rank:
                if min_rank is None or rank < min_rank:
                    min_rank = rank
                    min_idx = i
        if min_idx is None:
            break
        parts = (
            parts[:min_idx]
            + [parts[min_idx] + parts[min_idx + 1]]
            + parts[min_idx + 2 :]
        )
    return parts


def _recover_merges(mergeable_ranks):
    """Reconstruct BPE merge rules from a rank-ordered byte vocabulary.

    tiktoken/Tekken vocabularies do not store merge rules; they only store the
    final rank of each token. The standard reconstruction re-derives the merge
    that produced each multi-byte token by replaying BPE with the lower-ranked
    tokens. See `tiktoken.load.data_gym_to_mergeable_bpe_ranks`.
    """
    merges = []
    for token, rank in sorted(mergeable_ranks.items(), key=lambda kv: kv[1]):
        if len(token) == 1:
            # Single bytes are the base alphabet, not the result of a merge.
            continue
        pair = _bpe_split(mergeable_ranks, token, max_rank=rank)
        if len(pair) != 2:
            raise ValueError(
                f"Could not reconstruct a merge for token {token!r} with rank "
                f"{rank}; expected two pieces but got {len(pair)}."
            )
        merges.append((pair[0], pair[1]))
    return merges


def _convert_tekken_tokenizer(path):
    """Convert a `tekken.json` file into `BytePairTokenizer` arguments.

    Tekken is a tiktoken-style byte-level BPE tokenizer: a rank-ordered list of
    raw byte sequences with no explicit merges. The file is parsed with
    Mistral's own `mistral_common` library (the same backend Hugging Face
    delegates to), then the byte vocabulary is turned into `BytePairTokenizer`
    arguments: merges are reconstructed from the ranks and every token is
    re-encoded with the GPT-2 byte-to-unicode mapping.
    """
    try:
        from mistral_common.tokens.tokenizers.tekken import Tekkenizer
    except ImportError:
        raise ImportError(
            "Converting a Tekken (`tekken.json`) tokenizer requires the "
            "`mistral_common` package. Please install it with "
            "`pip install mistral-common`."
        )

    tokenizer = Tekkenizer.from_file(path)
    # `Tekkenizer` wraps a `tiktoken.Encoding`, which exposes both the byte
    # vocabulary (as bytes -> rank) and the pre-tokenization pattern.
    encoding = tokenizer._model
    mergeable_ranks = encoding._mergeable_ranks
    num_special_tokens = tokenizer.num_special_tokens
    split_pattern = encoding._pat_str

    byte_encoder = _bytes_to_unicode()

    def encode(token_bytes):
        return "".join(byte_encoder[b] for b in token_bytes)

    # Regular token ids are offset by the reserved special-token block, so the
    # id of a token with rank `r` is `r + num_special_tokens`.
    vocabulary = {
        encode(token_bytes): rank + num_special_tokens
        for token_bytes, rank in mergeable_ranks.items()
    }
    merges = [
        f"{encode(a)} {encode(b)}" for a, b in _recover_merges(mergeable_ranks)
    ]

    # Special tokens occupy the reserved block of ids
    # `[0, num_special_tokens)`; their id is simply their rank. These are not
    # reachable through BPE merges (they are not part of `mergeable_ranks`),
    # so every one of them must be registered as an unsplittable/special
    # token on the `tokenizers` backend, or literal occurrences in a prompt
    # (e.g. `"[INST]"` from a chat template) get shredded into several
    # regular byte-level tokens instead of mapping to their single reserved
    # id.
    control_tokens = []
    for rank in range(num_special_tokens):
        piece = tokenizer.id_to_piece(rank)
        vocabulary[piece] = rank
        control_tokens.append(piece)

    return vocabulary, merges, split_pattern, control_tokens


def convert_tokenizer(cls, preset, **kwargs):
    # Vision-enabled checkpoints (e.g. Mistral Small 3.x) set `vision_config`
    # in `config.json`; the tokenizer needs to know this so it can register the
    # `[IMG]`/`[IMG_BREAK]`/`[IMG_END]` special tokens used to expand image
    # placeholders during preprocessing.
    if check_file_exists(preset, "config.json"):
        transformers_config = load_json(preset, "config.json")
        kwargs.setdefault(
            "has_vision_tokens", "vision_config" in transformers_config
        )

    # Newer Mistral checkpoints (e.g. Magistral) ship a Tekken (byte-level BPE)
    # `tekken.json` instead of a SentencePiece `tokenizer.model`.
    if check_file_exists(preset, "tekken.json"):
        tekken_path = get_file(preset, "tekken.json")
        vocabulary, merges, split_pattern, control_tokens = (
            _convert_tekken_tokenizer(tekken_path)
        )
        return MistralTekkenTokenizer(
            vocabulary=vocabulary,
            merges=merges,
            split_pattern=split_pattern,
            control_tokens=control_tokens,
            **kwargs,
        )
    return cls(get_file(preset, "tokenizer.model"), **kwargs)
