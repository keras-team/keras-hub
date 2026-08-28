import numpy as np

from keras_hub.src.models.mistral.mistral_backbone import MistralBackbone
from keras_hub.src.models.mistral.mistral_tokenizer import (
    MistralTekkenTokenizer,
)
from keras_hub.src.utils.preset_utils import check_file_exists
from keras_hub.src.utils.preset_utils import get_file

backbone_cls = MistralBackbone


def convert_backbone_config(transformers_config):
    rope_theta = transformers_config.get("rope_parameters", {}).get(
        "rope_theta"
    )
    if rope_theta is None:
        rope_theta = transformers_config["rope_theta"]
    return {
        "vocabulary_size": transformers_config["vocab_size"],
        "num_layers": transformers_config["num_hidden_layers"],
        "num_query_heads": transformers_config["num_attention_heads"],
        "hidden_dim": transformers_config["hidden_size"],
        "intermediate_dim": transformers_config["intermediate_size"],
        "num_key_value_heads": transformers_config["num_key_value_heads"],
        "rope_max_wavelength": rope_theta,
        "layer_norm_epsilon": transformers_config["rms_norm_eps"],
        "sliding_window": transformers_config.get("sliding_window"),
        "head_dim": transformers_config.get("head_dim"),
    }


def convert_weights(backbone, loader, transformers_config):
    # Embeddings
    loader.port_weight(
        keras_variable=backbone.token_embedding.embeddings,
        hf_weight_key="model.embed_tokens.weight",
        hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float16),
    )
    loader.port_weight(
        keras_variable=backbone.token_embedding.reverse_embeddings,
        hf_weight_key="lm_head.weight",
        hook_fn=lambda hf_tensor, _: np.transpose(
            hf_tensor.astype(np.float16), axes=(1, 0)
        ),
    )

    # Attention blocks
    for index in range(backbone.num_layers):
        decoder_layer = backbone.transformer_layers[index]

        # Norm layers
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layernorm.scale,
            hf_weight_key=f"model.layers.{index}.input_layernorm.weight",
            hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float16),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_layernorm.scale,
            hf_weight_key=f"model.layers.{index}.post_attention_layernorm.weight",
            hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float16),
        )

        # Attention layers
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._query_dense.kernel,
            hf_weight_key=f"model.layers.{index}.self_attn.q_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor.astype(np.float16)), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._key_dense.kernel,
            hf_weight_key=f"model.layers.{index}.self_attn.k_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor.astype(np.float16)), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._value_dense.kernel,
            hf_weight_key=f"model.layers.{index}.self_attn.v_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor.astype(np.float16)), keras_shape
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._self_attention_layer._output_dense.kernel,
            hf_weight_key=f"model.layers.{index}.self_attn.o_proj.weight",
            hook_fn=lambda hf_tensor, keras_shape: np.reshape(
                np.transpose(hf_tensor.astype(np.float16)), keras_shape
            ),
        )

        # MLP layers
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_gate_dense.kernel,
            hf_weight_key=f"model.layers.{index}.mlp.gate_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor.astype(np.float16), axes=(1, 0)
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_intermediate_dense.kernel,
            hf_weight_key=f"model.layers.{index}.mlp.up_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor.astype(np.float16), axes=(1, 0)
            ),
        )
        loader.port_weight(
            keras_variable=decoder_layer._feedforward_output_dense.kernel,
            hf_weight_key=f"model.layers.{index}.mlp.down_proj.weight",
            hook_fn=lambda hf_tensor, _: np.transpose(
                hf_tensor.astype(np.float16), axes=(1, 0)
            ),
        )

    # Normalization
    loader.port_weight(
        keras_variable=backbone.layer_norm.scale,
        hf_weight_key="model.norm.weight",
        hook_fn=lambda hf_tensor, _: hf_tensor.astype(np.float16),
    )


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
    # `[0, num_special_tokens)`; their id is simply their rank.
    for rank in range(num_special_tokens):
        vocabulary[tokenizer.id_to_piece(rank)] = rank

    return vocabulary, merges, split_pattern


def convert_tokenizer(cls, preset, **kwargs):
    # Newer Mistral checkpoints (e.g. Magistral) ship a Tekken (byte-level BPE)
    # `tekken.json` instead of a SentencePiece `tokenizer.model`.
    if check_file_exists(preset, "tekken.json"):
        tekken_path = get_file(preset, "tekken.json")
        vocabulary, merges, split_pattern = _convert_tekken_tokenizer(
            tekken_path
        )
        return MistralTekkenTokenizer(
            vocabulary=vocabulary,
            merges=merges,
            split_pattern=split_pattern,
            **kwargs,
        )
    return cls(get_file(preset, "tokenizer.model"), **kwargs)
