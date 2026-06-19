"""Convert KerasHub BytePair tokenizers to HuggingFace tokenizer.json format.

The converter is intentionally dependency-free at runtime: it builds the JSON
structure that ``tokenizers.Tokenizer.from_file()`` expects without importing
the ``tokenizers`` library.  The ``tokenizers`` library is only used in tests
to validate the output.
"""

import json
import os

from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer

# Mapping from (module, class_name) of known KerasHub BytePair tokenizers to
# the corresponding HuggingFace tokenizer family.
_TOKENIZER_FAMILY_MAP = {
    ("keras_hub.src.models.gpt2.gpt2_tokenizer", "GPT2Tokenizer"): "gpt2",
    (
        "keras_hub.src.models.llama3.llama3_tokenizer",
        "Llama3Tokenizer",
    ): "llama3",
    ("keras_hub.src.models.qwen3.qwen3_tokenizer", "Qwen3Tokenizer"): "qwen3",
}


def infer_hf_tokenizer_family(tokenizer):
    """Infer the HF tokenizer family from a KerasHub tokenizer instance.

    Args:
        tokenizer: A KerasHub tokenizer instance.

    Returns:
        One of ``"gpt2"``, ``"llama3"``, ``"qwen3"``, or ``None`` if the
        tokenizer class is not a known BytePair family.
    """
    cls = type(tokenizer)
    return _TOKENIZER_FAMILY_MAP.get((cls.__module__, cls.__name__))


def convert_byte_pair_to_hf(tokenizer, family):
    """Convert a KerasHub BytePairTokenizer to a HF tokenizer.json dict.

    Args:
        tokenizer: A
            ``keras_hub.src.tokenizers.byte_pair_tokenizer.BytePairTokenizer``
            instance.
        family: str. One of ``"gpt2"``, ``"llama3"``, ``"qwen3"``.

    Returns:
        dict: A ``tokenizer.json``-compatible dictionary that can be written to
        disk and loaded with ``tokenizers.Tokenizer.from_file(path)``.
    """
    if not isinstance(tokenizer, BytePairTokenizer):
        raise TypeError(
            "`convert_byte_pair_to_hf` expects a BytePairTokenizer instance. "
            f"Received: {type(tokenizer).__name__}."
        )

    if family not in ("gpt2", "llama3", "qwen3"):
        raise ValueError(
            "`family` must be one of 'gpt2', 'llama3', 'qwen3'. "
            f"Received: {family!r}."
        )

    vocab = dict(tokenizer.vocabulary)
    merges = _clean_merges(tokenizer.merges)

    added_tokens = _build_added_tokens(tokenizer, family)
    pre_tokenizer = _pre_tokenizer_config(family)
    decoder = _decoder_config(family)
    post_processor = _post_processor_config(tokenizer, family)
    unk_token = _unk_token(tokenizer, family)

    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": None,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": post_processor,
        "decoder": decoder,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": unk_token,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": [merge.split(" ") for merge in merges],
        },
    }


def _clean_merges(merges):
    """Return merge rules, skipping ``#version:`` header lines."""
    cleaned = []
    for merge in merges:
        if "#version:" in merge.lstrip():
            continue
        cleaned.append(merge)
    return cleaned


def _build_added_tokens(tokenizer, family):
    """Build the ``added_tokens`` list for the tokenizer family."""
    if family == "gpt2":
        special_tokens = ["<|endoftext|>"]
    else:
        # For llama3 and qwen3, include every registered special token that is
        # present in the vocabulary.
        special_tokens = [
            token
            for token in tokenizer.special_tokens
            if token is not None and token in tokenizer.vocabulary
        ]

    added = []
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        added.append(
            {
                "id": token_id,
                "content": token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )
    # Sort by id to match the canonical tokenizer.json ordering.
    added.sort(key=lambda entry: entry["id"])
    return added


def _pre_tokenizer_config(family):
    """Return the family-specific pre-tokenizer configuration."""
    # GPT-2 style ByteLevel pre-tokenizer is appropriate for all three families
    # because KerasHub's BytePairTokenizer shares the same bytes-to-unicode
    # whitespace mapping.
    return {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": True,
        "use_regex": True,
    }


def _decoder_config(family):
    """Return the family-specific decoder configuration."""
    return {
        "type": "ByteLevel",
        "add_prefix_space": True,
        "trim_offsets": False,
        "use_regex": True,
    }


def _post_processor_config(tokenizer, family):
    """Return the family-specific post-processor configuration."""
    if family == "gpt2":
        return {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        }

    # llama3 / qwen3: wrap the sequence with BOS/EOS special tokens using a
    # TemplateProcessing post-processor when those tokens are present in the
    # vocabulary.
    special_token_entries = []
    single_template = []

    start_token = getattr(tokenizer, "start_token", None)
    start_token_id = getattr(tokenizer, "start_token_id", None)
    if start_token is not None and start_token_id is not None:
        single_template.append(
            {"SpecialToken": {"id": start_token, "type_id": 0}}
        )
        special_token_entries.append((start_token, start_token_id))

    single_template.append({"Sequence": {"id": "A", "type_id": 0}})

    end_token = getattr(tokenizer, "end_token", None)
    end_token_id = getattr(tokenizer, "end_token_id", None)
    if end_token is not None and end_token_id is not None:
        single_template.append(
            {"SpecialToken": {"id": end_token, "type_id": 0}}
        )
        special_token_entries.append((end_token, end_token_id))

    special_tokens_map = {
        token: {
            "id": token,
            "ids": [token_id],
            "tokens": [token],
        }
        for token, token_id in special_token_entries
    }

    return {
        "type": "TemplateProcessing",
        "single": single_template,
        "pair": [
            {"Sequence": {"id": "A", "type_id": 0}},
            {"Sequence": {"id": "B", "type_id": 1}},
        ],
        "special_tokens": special_tokens_map,
    }


def _unk_token(tokenizer, family):
    """Return the ``unk_token`` for the BPE model block."""
    if family == "gpt2":
        return "<|endoftext|>"

    if family == "llama3":
        token = "<|end_of_text|>"
        if token in tokenizer.vocabulary:
            return token
        return None

    # qwen3
    token = "<|endoftext|>"
    if token in tokenizer.vocabulary:
        return token
    return None


def materialize_hf_tokenizer_json(tokenizer, temp_dir):
    """Convert a KerasHub tokenizer and write ``tokenizer.json`` to disk.

    Args:
        tokenizer: A KerasHub tokenizer instance.
        temp_dir: str. Directory where ``tokenizer.json`` will be written.

    Returns:
        str: Path to the written ``tokenizer.json`` file.
    """
    family = infer_hf_tokenizer_family(tokenizer)
    if family is None:
        raise ValueError(
            "Cannot infer HuggingFace tokenizer family from "
            f"{type(tokenizer).__module__}.{type(tokenizer).__name__}. "
            "Supported families are 'gpt2', 'llama3', and 'qwen3'."
        )

    hf_tokenizer_dict = convert_byte_pair_to_hf(tokenizer, family)
    tokenizer_path = os.path.join(temp_dir, "tokenizer.json")
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        json.dump(hf_tokenizer_dict, f, ensure_ascii=False, indent=2)
    return tokenizer_path
