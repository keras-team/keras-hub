"""Convert KerasHub BytePair tokenizers to HuggingFace tokenizer.json format.

The converter is intentionally dependency-free at runtime: it builds the JSON
structure that ``tokenizers.Tokenizer.from_file()`` expects without importing
the ``tokenizers`` library.  The ``tokenizers`` library is only used in tests
to validate the output.
"""

import json
import os

from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer


def convert_byte_pair_to_hf(tokenizer):
    """Convert any KerasHub BytePairTokenizer subclass to HF tokenizer.json.

    Args:
        tokenizer: A
            ``keras_hub.src.tokenizers.byte_pair_tokenizer.BytePairTokenizer``
            instance.

    Returns:
        dict: A ``tokenizer.json``-compatible dictionary that can be written to
        disk and loaded with ``tokenizers.Tokenizer.from_file(path)``.
    """
    if not isinstance(tokenizer, BytePairTokenizer):
        raise TypeError(
            "`convert_byte_pair_to_hf` expects a BytePairTokenizer instance. "
            f"Received: {type(tokenizer).__name__}."
        )

    vocab = dict(tokenizer.vocabulary)
    merges = _clean_merges(tokenizer.merges)

    added_tokens = _build_added_tokens(tokenizer)
    pre_tokenizer = _pre_tokenizer_config(tokenizer)
    decoder = _decoder_config(tokenizer)
    post_processor = _post_processor_config(tokenizer)
    unk_token = _unk_token(tokenizer)

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


def _build_added_tokens(tokenizer):
    """Build the ``added_tokens`` list from the tokenizer's special tokens."""
    special_tokens = [
        token
        for token in tokenizer.special_tokens
        if token is not None and token in tokenizer.vocabulary
    ]
    added = []
    seen_ids = set()
    for token in special_tokens:
        token_id = tokenizer.token_to_id(token)
        if token_id in seen_ids:
            continue
        seen_ids.add(token_id)
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
    added.sort(key=lambda entry: entry["id"])
    return added


def _pre_tokenizer_config(tokenizer):
    """Return the ByteLevel pre-tokenizer configuration."""
    add_prefix_space = getattr(tokenizer, "add_prefix_space", False)
    return {
        "type": "ByteLevel",
        "add_prefix_space": add_prefix_space,
        "trim_offsets": True,
        "use_regex": True,
    }


def _decoder_config(tokenizer):
    """Return the ByteLevel decoder configuration."""
    return {
        "type": "ByteLevel",
        "add_prefix_space": True,
        "trim_offsets": False,
        "use_regex": True,
    }


def _post_processor_config(tokenizer):
    """Return the post-processor configuration from tokenizer attributes."""
    start_token = getattr(tokenizer, "start_token", None)
    start_token_id = getattr(tokenizer, "start_token_id", None)
    end_token = getattr(tokenizer, "end_token", None)
    end_token_id = getattr(tokenizer, "end_token_id", None)

    if (
        start_token is not None
        and end_token is not None
        and start_token == end_token
        and start_token_id is not None
    ):
        return {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        }

    special_token_entries = []
    single_template = []

    if start_token is not None and start_token_id is not None:
        single_template.append(
            {"SpecialToken": {"id": start_token, "type_id": 0}}
        )
        special_token_entries.append((start_token, start_token_id))

    single_template.append({"Sequence": {"id": "A", "type_id": 0}})

    if end_token is not None and end_token_id is not None:
        single_template.append(
            {"SpecialToken": {"id": end_token, "type_id": 0}}
        )
        special_token_entries.append((end_token, end_token_id))

    if not special_token_entries:
        return {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        }

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


def _unk_token(tokenizer):
    """Return the ``unk_token`` for the BPE model block."""
    for attr in ("end_token", "pad_token"):
        token = getattr(tokenizer, attr, None)
        if token is not None and token in tokenizer.vocabulary:
            return token
    return None


def materialize_hf_tokenizer_json(tokenizer, temp_dir):
    """Convert a KerasHub tokenizer and write ``tokenizer.json`` to disk.

    Args:
        tokenizer: A KerasHub ``BytePairTokenizer`` instance.
        temp_dir: str. Directory where ``tokenizer.json`` will be written.

    Returns:
        str: Path to the written ``tokenizer.json`` file.
    """
    if not isinstance(tokenizer, BytePairTokenizer):
        raise TypeError(
            "`materialize_hf_tokenizer_json` expects a BytePairTokenizer "
            f"instance. Received: {type(tokenizer).__name__}."
        )

    hf_tokenizer_dict = convert_byte_pair_to_hf(tokenizer)
    tokenizer_path = os.path.join(temp_dir, "tokenizer.json")
    with open(tokenizer_path, "w", encoding="utf-8") as f:
        json.dump(hf_tokenizer_dict, f, ensure_ascii=False, indent=2)
    return tokenizer_path
