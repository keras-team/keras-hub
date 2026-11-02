"""Randomness helpers for TensorFlow-free preprocessing layers.

Grain runs preprocessing inside worker processes and pickles the layer into
each of them. Any random state stored on the layer is therefore *copied*, not
shared, so a generator created in `__init__` would replay the exact same
stream in every worker. The helpers here derive randomness from the record
being augmented instead, which keeps outputs independent of how many workers
are used and of the order records are seen in.
"""

import hashlib
import operator
import struct

import numpy as np

# `np.random.SeedSequence` rejects negative entropy, so seeds are folded into
# this range before use. Folding is the identity for `0 <= seed < 2**32`.
_SEED_MASK = 0xFFFFFFFF


def _token_bytes(token):
    """Return a canonical byte encoding for a single token.

    Numbers are encoded by value rather than by `repr()`, so the encoding
    does not depend on the python container the number arrived in, nor on
    the NumPy version (`repr(np.int64(1))` is `'1'` on NumPy 1 and
    `'np.int64(1)'` on NumPy 2).
    """
    # `np.str_` and `np.bytes_` subclass `str` and `bytes`, so they are
    # handled by the two branches below and never reach the numeric ones.
    if isinstance(token, bytes):
        return token
    if isinstance(token, str):
        return token.encode("utf-8")
    if isinstance(token, np.generic):
        # Unwrap numpy scalars, e.g. `np.int64(1)` -> `1`. Iterating a
        # `np.ndarray` yields these, so this is what makes an array and a
        # list of the same numbers hash alike.
        token = token.item()
    if isinstance(token, int):
        # `bool` is a subclass of `int`, and `np.bool_.item()` returns a
        # `bool`, so `True`/`False` hash as `1`/`0`.
        return str(int(token)).encode("utf-8")
    if isinstance(token, float):
        # Fixed width IEEE 754, so the encoding does not depend on `repr()`
        # formatting. `np.float32` widens to the double of the same value,
        # so `np.float32(0.5)` and `0.5` agree, while `np.float32(0.1)` and
        # `0.1` do not -- they are different numbers.
        return struct.pack("<d", token)
    # Note: `np.longdouble` returns itself from `.item()`, falling through
    # here. Since its formatting changed in NumPy 2.0, it remains sensitive
    # to the NumPy version.
    return repr(token).encode("utf-8")


def stable_hash(tokens):
    """Return a stable 64 bit integer hash for a sequence of tokens.

    Unlike the builtin `hash()`, which is salted per process via
    `PYTHONHASHSEED`, this hash is identical across processes and across runs.
    That makes it safe to derive random seeds from inside Grain workers.

    `str` and `bytes` tokens hash as their UTF-8 bytes. Numbers hash by
    value, so `[1, 2, 3]`, `np.array([1, 2, 3])` and
    `[np.int64(1), np.int64(2), np.int64(3)]` all give the same hash.
    Booleans hash as the integers `1` and `0`. Any other type falls back to
    `repr()`, which is only stable for types whose `repr` is; objects using
    the default `object.__repr__` embed a memory address and will not hash
    the same way twice.

    Args:
        tokens: an iterable of `str`, `bytes`, or numbers.
    """
    hasher = hashlib.blake2b(digest_size=8)
    for token in tokens:
        hasher.update(_token_bytes(token))
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
        seed: int. The layer level seed. It is folded into the unsigned 32 bit
            range, since `np.random.SeedSequence` rejects negative entropy.
            Seeds in `[0, 2**32)` are used as is.
        tokens: an iterable of `str`, `bytes`, or numbers.
    """
    return np.random.default_rng(
        [operator.index(seed) & _SEED_MASK, stable_hash(tokens)]
    )
