import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export

# Sentinel used instead of literal `-inf` to avoid NaN propagation (e.g. from
# `inf - inf`) on some backends, matching the convention already used for
# masked-out log probabilities in `keras_hub.samplers.BeamSampler`.
_MASKED_SCORE = -1e9
# Mirror of `_MASKED_SCORE`, for positions that must survive eviction
# regardless of what `score()` returned for them.
_FORCED_SCORE = 1e9


@keras_hub_export("keras_hub.press.KVCachePress")
class KVCachePress:
    """Base class for KV cache compression strategies.

    A `KVCachePress` reduces the memory footprint of the key/value cache used
    during autoregressive generation by evicting less important tokens right
    after the prompt has been prefilled. Unlike a simple truncation, eviction
    is computed independently per transformer layer and per attention head:
    each layer/head keeps the same *number* of tokens (so the cache stays a
    dense, statically-shaped tensor), but may keep *different* tokens.

    This is a port of the core idea behind the
    [KVPress](https://github.com/NVIDIA/kvpress) library. To add a new
    compression strategy, subclass `KVCachePress` and implement `score()`.

    Args:
        compression_ratio: float. The fraction of prompt tokens to evict,
            in `[0, 1)`. For example, `0.5` keeps half of the prompt's KV
            cache entries. Defaults to `0.5`.

    `compression_ratio` is applied against the *real* (non-padding) prompt
    length, matching the reference
    [KVPress](https://github.com/NVIDIA/kvpress) library. Note that the
    compressed buffer is not itself `compression_ratio` smaller: it must
    also reserve one slot per token `generate()` is about to produce, and
    that reserve is not compressible. For a 716-token prompt in a 1024-slot
    buffer, `compression_ratio=0.75` yields `179 + 308 = 487` slots -- a 52%
    saving, not 75%.

    Compression during `generate()` currently requires the `"torch"`
    backend. Sizing the buffer correctly depends on the real prompt length,
    which is only known at runtime, whereas the buffer's shape must be
    static at trace time under `jax.jit`/XLA-compiled `tf.function`s.
    Rather than silently substitute the padded buffer length -- which
    makes the decode loop overwrite the very tokens that were just retained
    -- `compress()` raises `NotImplementedError` on those backends when a
    `padding_mask` is supplied.
    """

    def __init__(self, compression_ratio=0.5):
        if not 0.0 <= compression_ratio < 1.0:
            raise ValueError(
                "`compression_ratio` must be in the range [0, 1). Received: "
                f"compression_ratio={compression_ratio}"
            )
        self.compression_ratio = compression_ratio

    def score(self, keys, values, keep_len, padding_mask=None):
        """Compute per-token importance scores.

        Args:
            keys: a tensor of shape
                `(batch_size, num_layers, seq_len, num_heads, head_dim)`,
                the cached key projections.
            values: a tensor of the same shape as `keys`, the cached value
                projections.
            keep_len: int. The number of tokens that will be kept per layer
                and head, as determined by `compression_ratio`.
            padding_mask: an optional boolean tensor of shape
                `(batch_size, seq_len)`, `True` for positions eligible to
                be retained. This is the *effective* mask: real
                (non-padding) tokens that also fall inside the prompt
                region shared by every row of the batch. Subclasses whose
                scoring depends on a token's position relative to that
                region (rather than its raw physical slot in the padded
                buffer) should use it to compute per-row lengths -- see
                `StreamingLLMPress` for an example. Ineligible positions
                are masked out separately by `compress()` regardless of
                what `score()` returns for them, so subclasses may ignore
                this argument entirely.

        Returns:
            A tensor of shape `(batch_size, num_layers, num_heads, seq_len)`
            with higher scores indicating tokens that are more important to
            keep.
        """
        raise NotImplementedError(
            "`KVCachePress` subclasses must implement `score()`."
        )

    def compress(self, cache, padding_mask=None):
        """Evict the lowest-scoring tokens from `cache`.

        The returned buffer is laid out as `keep_len` retained prompt
        positions, packed chronologically into slots `[0, keep_len)`,
        followed by `seq_len - prompt_len` **free slots reserved for the
        tokens generation is about to produce**. That reserve is what makes
        the caller's physical-slot mapping (`logical_position -
        cache_index_offset`, where `cache_index_offset` works out to
        `prompt_len - keep_len`) land on the free region instead of
        overwriting the tokens that were just retained.

        Args:
            cache: a dense float tensor of shape
                `(batch_size, num_layers, 2, seq_len, num_heads, head_dim)`.
            padding_mask: an optional boolean tensor of shape
                `(batch_size, seq_len)`, `True` for real (non-padding)
                tokens. Padding positions are never retained.

        Returns:
            A tensor of shape
            `(batch_size, num_layers, 2, buffer_len, num_heads, head_dim)`,
            where `buffer_len = keep_len + (seq_len - prompt_len)` and
            `keep_len` is derived from `compression_ratio`.
        """
        # `compression_ratio == 0.0` must stay a hard, exact no-op on every
        # backend regardless of the `keep_len` basis below -- generation
        # with a zero-ratio press is required to be byte-identical to an
        # uncompressed run.
        if self.compression_ratio == 0.0:
            return cache

        seq_len = cache.shape[3]
        # Both `keep_len` and the generation reserve determine the returned
        # buffer's *shape*, which must be static at trace time for
        # jax.jit/XLA-compiled tf.function -- neither can depend on
        # padding_mask's actual values there. Eager PyTorch has no such
        # restriction; see the class docstring for why the other backends
        # are rejected outright rather than silently approximated.
        if padding_mask is not None and keras.config.backend() == "torch":
            row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
            # Decoding for the whole batch starts at the *shortest* row's
            # real length, so that -- not the longest row's -- is the
            # prompt region every row shares. Positions at or beyond it
            # belong to longer rows and are rewritten by the decode loop
            # anyway (the sampler re-feeds their real prompt token), so
            # excluding them here loses nothing and keeps a single
            # batch-wide slot mapping valid.
            prompt_len = int(ops.convert_to_numpy(ops.min(row_lengths)))
            prompt_len = max(1, min(prompt_len, seq_len))
        elif padding_mask is not None:
            raise NotImplementedError(
                "KV cache compression during `generate()` is currently "
                "supported only on the PyTorch backend. Sizing the "
                "compressed cache correctly requires the real prompt "
                "length, which is only known at runtime, but the buffer's "
                "shape must be static under jax.jit / XLA-compiled "
                "tf.function. Received: "
                f"keras.config.backend()={keras.config.backend()!r}. Set "
                "`press=None`, or run with `KERAS_BACKEND=torch`."
            )
        else:
            # No padding mask: treat the whole buffer as prompt. Nothing is
            # generated afterwards, so no reserve is needed.
            prompt_len = seq_len

        keep_ratio = 1.0 - self.compression_ratio
        keep_len = max(1, int(round(prompt_len * keep_ratio)))
        if keep_len >= prompt_len:
            return cache
        # Slots the decode loop still needs for the tokens it will write.
        reserve = seq_len - prompt_len

        # Only positions inside the shared prompt region are eligible; a
        # retained position outside it would be double-written by the
        # decode loop and corrupt the slot mapping.
        positions = ops.arange(seq_len)
        eligible = ops.reshape(positions < prompt_len, (1, 1, 1, seq_len))
        if padding_mask is not None:
            eligible = ops.logical_and(
                eligible, ops.cast(padding_mask, "bool")[:, None, None, :]
            )
            # Hand `score()` the *effective* mask so position-aware presses
            # (e.g. StreamingLLMPress's "most recent" window) measure
            # against the same prompt region eviction uses.
            score_mask = ops.squeeze(ops.squeeze(eligible, axis=1), axis=1)
        else:
            score_mask = None

        keys = cache[:, :, 0, ...]
        values = cache[:, :, 1, ...]
        scores = self.score(keys, values, keep_len, padding_mask=score_mask)
        scores = ops.where(
            eligible, scores, ops.full_like(scores, _MASKED_SCORE)
        )
        if reserve > 0:
            # A reserve means decoding follows, and its first step maps to
            # slot `keep_len - 1` and rewrites it with exactly the last
            # prompt position's key/value. Force that position to survive
            # so the slot isn't spent on a token about to be clobbered.
            # With no reserve nothing is decoded afterwards, so leave the
            # press's own ranking untouched.
            forced = ops.reshape(
                ops.equal(positions, prompt_len - 1), (1, 1, 1, seq_len)
            )
            scores = ops.where(
                forced, ops.full_like(scores, _FORCED_SCORE), scores
            )

        _, keep_indices = ops.top_k(scores, k=keep_len, sorted=False)
        # Restore chronological order. This doesn't affect correctness for
        # models like GPT2 (whose causal mask only cares about physical slot
        # vs. write pointer, not stored order), but keeps the compressed
        # cache well-formed for future backends that are order-sensitive.
        keep_indices = ops.sort(keep_indices, axis=-1)
        compressed = _gather_cache_positions(cache, keep_indices)
        if reserve > 0:
            reserve_shape = list(compressed.shape)
            reserve_shape[3] = reserve
            compressed = ops.concatenate(
                [compressed, ops.zeros(reserve_shape, dtype=compressed.dtype)],
                axis=3,
            )
        return compressed

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {"compression_ratio": self.compression_ratio}


def _gather_cache_positions(cache, keep_indices):
    """Gather `cache` along the sequence axis with per layer/head indices.

    Args:
        cache: `(batch_size, num_layers, 2, seq_len, num_heads, head_dim)`.
        keep_indices: `(batch_size, num_layers, num_heads, keep_len)` int
            tensor of positions to keep, distinct per layer and head.

    Returns:
        `(batch_size, num_layers, 2, keep_len, num_heads, head_dim)`.
    """
    batch_size, num_layers = cache.shape[0], cache.shape[1]
    num_heads = cache.shape[4]
    keep_len = keep_indices.shape[-1]
    # (b, l, h, keep_len) -> (b, l, keep_len, h), matching the seq/head axis
    # order of `cache`, then broadcast explicitly (rather than relying on
    # implicit broadcasting in `take_along_axis`, which has known dynamic-
    # shape issues on the TensorFlow backend) up to -- but not including --
    # the trailing `head_dim` axis.
    indices = ops.transpose(keep_indices, axes=(0, 1, 3, 2))
    indices = ops.expand_dims(indices, axis=2)
    indices = ops.expand_dims(indices, axis=-1)
    # `head_dim` deliberately stays size-1 for the backend to expand
    # internally. Selecting a position selects its whole head_dim-length
    # key/value vector, so `head_dim` copies of each index carry no
    # information -- and they are far from free: `take_along_axis`
    # normalizes negative indices with a `where()`, which densifies the
    # otherwise stride-0 broadcast into a real int64 tensor. At full cache
    # rank that temporary is twice the size of the compressed cache (604 MB
    # for a 12-layer, batch-8, 512-slot float32 cache) and it peaks while
    # the *uncompressed* cache is still live. Keeping the last axis at 1
    # makes it 64x smaller here, for bit-identical output.
    indices = ops.cast(indices, "int64")
    indices = ops.broadcast_to(
        indices, (batch_size, num_layers, 2, keep_len, num_heads, 1)
    )
    return ops.take_along_axis(cache, indices, axis=3)
