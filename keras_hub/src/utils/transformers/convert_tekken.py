"""Utilities for converting Mistral `tekken.json` tokenizers.

Newer Mistral checkpoints (e.g. Magistral) ship a `tekken.json` file instead
of a SentencePiece `tokenizer.model`. Tekken is a tiktoken-style byte-level
BPE tokenizer: the vocabulary is a rank-ordered list of raw byte sequences and
there are no explicit merge rules.

The file is parsed with Mistral's own `mistral_common` library (the same
backend Hugging Face delegates to for Tekken), and the resulting byte
vocabulary is turned into `BytePairTokenizer` arguments: merge rules are
reconstructed from the ranks and every token is re-encoded with the GPT-2
byte-to-unicode mapping so it can be stored as a string.
"""


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


def convert_tekken_tokenizer(path):
    """Convert a `tekken.json` file into `BytePairTokenizer` arguments.

    Args:
        path: str. Path to a `tekken.json` file.

    Returns:
        A tuple `(vocabulary, merges, special_tokens, split_pattern)` where
        `vocabulary` maps GPT-2-unicode encoded token strings to integer ids,
        `merges` is a list of `"a b"` merge rules, `special_tokens` is the
        ordered list of control token strings, and `split_pattern` is the
        pre-tokenization regex.
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
    # vocabulary (as rank -> bytes) and the pre-tokenization pattern.
    encoding = tokenizer._model
    mergeable_ranks = encoding._mergeable_ranks
    num_special_tokens = tokenizer.num_special_tokens
    split_pattern = encoding._pat_str

    byte_encoder = bytes_to_unicode()

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
    special_tokens = []
    for rank in range(num_special_tokens):
        token_str = tokenizer.id_to_piece(rank)
        vocabulary[token_str] = rank
        special_tokens.append(token_str)

    return vocabulary, merges, special_tokens, split_pattern
