"""Randomness helpers for TensorFlow-free preprocessing layers.

Grain runs preprocessing inside worker processes and pickles the layer into
each of them. Any random state stored on the layer is therefore *copied*, not
shared, so a generator created in `__init__` would replay the exact same
stream in every worker. The helpers here derive randomness from the record
being augmented instead, which keeps outputs independent of how many workers
are used and of the order records are seen in.
"""

import hashlib

import numpy as np


def stable_hash(tokens):
    """Return a stable 64 bit integer hash for a sequence of tokens.

    Unlike the builtin `hash()`, which is salted per process via
    `PYTHONHASHSEED`, this hash is identical across processes and across runs.
    That makes it safe to derive random seeds from inside Grain workers.

    Args:
        tokens: an iterable of `str`, `bytes`, or numbers.
    """
    hasher = hashlib.blake2b(digest_size=8)
    for token in tokens:
        if isinstance(token, bytes):
            hasher.update(token)
        elif isinstance(token, str):
            hasher.update(token.encode("utf-8"))
        else:
            hasher.update(repr(token).encode("utf-8"))
        # Separator, so `["ab", "c"]` and `["a", "bc"]` hash differently.
        hasher.update(b"\x00")
    return int.from_bytes(hasher.digest(), "big")


def record_rng(seed, tokens):
    """Create a `np.random.Generator` keyed to a seed and a record's content.

    The returned generator depends only on `seed` and on the content of
    `tokens`. It does not depend on the position of the record in the dataset,
    on how many Grain workers are running, or on which worker happens to pick
    the record up. Augmenting the same dataset with `num_workers=1` and
    `num_workers=8` therefore gives identical results.

    The tradeoff is that identical records are augmented identically. Callers
    that want augmentation to vary per epoch should pass their own generator,
    for example via `grain.RandomMapTransform`, which hands out a generator
    derived from the element index.

    Args:
        seed: int. The layer level seed.
        tokens: an iterable of `str`, `bytes`, or numbers.
    """
    return np.random.default_rng([seed, stable_hash(tokens)])
