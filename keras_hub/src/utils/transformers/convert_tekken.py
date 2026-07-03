"""Utilities for converting Mistral `tekken.json` tokenizers.

Newer Mistral checkpoints (e.g. Magistral) ship a `tekken.json` file instead
of a SentencePiece `tokenizer.model`. Tekken is a tiktoken-style byte-level
BPE tokenizer: the vocabulary is a rank-ordered list of raw byte sequences and
there are no explicit merge rules. This module reconstructs the merge rules
from the ranks and re-encodes the vocabulary using the GPT-2 byte-to-unicode
mapping so the tokenizer can be represented as a `BytePairTokenizer`.
"""

import base64


def bytes_to_unicode():
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

    Returns the list of pieces the byte sequence `token` decomposes into when
    only pairs with rank strictly less than `max_rank` are allowed to merge.
    For a token that is itself the result of a single merge, this returns the
    two pieces that were merged to create it.
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

    tiktoken/tekken vocabularies do not store merge rules; they only store the
    final rank of each token. The standard reconstruction re-derives the merge
    that produced each multi-byte token by replaying BPE with the lower-ranked
    tokens. See the reference implementation in
    `tiktoken.load.data_gym_to_mergeable_bpe_ranks`.
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


def convert_tekken_tokenizer(tekken_config, vocabulary_size):
    """Convert a parsed `tekken.json` into `BytePairTokenizer` arguments.

    Args:
        tekken_config: dict. The parsed contents of a `tekken.json` file.
        vocabulary_size: int. The model's vocabulary size (from the HF
            `config.json`). Tekken files list more tokens than the model
            actually uses, so we keep only the first
            `vocabulary_size - num_special_tokens` regular tokens.

    Returns:
        A tuple `(vocabulary, merges, special_tokens)` where `vocabulary` maps
        GPT-2-unicode encoded token strings to integer ids, `merges` is a list
        of `"a b"` merge rules, and `special_tokens` is the ordered list of
        control token strings.
    """
    config = tekken_config["config"]
    num_special_tokens = config["default_num_special_tokens"]
    num_regular_tokens = vocabulary_size - num_special_tokens

    # Map each used regular token's rank to its raw bytes.
    rank_to_bytes = {}
    for entry in tekken_config["vocab"][:num_regular_tokens]:
        rank_to_bytes[entry["rank"]] = base64.b64decode(entry["token_bytes"])
    mergeable_ranks = {
        token_bytes: rank for rank, token_bytes in rank_to_bytes.items()
    }

    merges = _recover_merges(mergeable_ranks)

    byte_encoder = bytes_to_unicode()

    def encode(token_bytes):
        return "".join(byte_encoder[b] for b in token_bytes)

    # Regular token ids are offset by the reserved special-token block, so the
    # id of a token with rank `r` is `r + num_special_tokens`.
    vocabulary = {
        encode(rank_to_bytes[rank]): rank + num_special_tokens
        for rank in rank_to_bytes
    }
    merges = [f"{encode(a)} {encode(b)}" for a, b in merges]

    # Special tokens occupy the reserved block of ids `[0, num_special_tokens)`
    # and their id is simply their rank.
    sorted_special = sorted(
        tekken_config["special_tokens"], key=lambda e: e["rank"]
    )
    special_tokens = []
    for entry in sorted_special:
        token_str = entry["token_str"]
        vocabulary[token_str] = entry["rank"]
        special_tokens.append(token_str)

    return vocabulary, merges, special_tokens
