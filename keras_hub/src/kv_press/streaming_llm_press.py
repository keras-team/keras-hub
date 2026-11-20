from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.kv_press.press import _MASKED_SCORE
from keras_hub.src.kv_press.press import KVCachePress


@keras_hub_export("keras_hub.press.StreamingLLMPress")
class StreamingLLMPress(KVCachePress):
    """Keeps only the first "sink" tokens and the most recent tokens.

    Follows [StreamingLLM](https://arxiv.org/abs/2309.17453): the first
    `n_sink` tokens (which tend to receive disproportionately high attention
    regardless of content) and the most recent tokens are kept; everything
    in between is evicted. This press ignores content entirely, so its
    behavior is identical across layers and heads.

    "Most recent" is relative to each row's real prompt content, not its
    physical slot in the (generally longer, right-padded) cache buffer --
    a `padding_mask` is required to identify that content correctly.

    Args:
        compression_ratio: float. The fraction of prompt tokens to evict,
            in `[0, 1)`. Defaults to `0.5`.
        n_sink: int. The number of leading "attention sink" tokens to always
            keep. Defaults to `4`.

    Example:
    ```python
    causal_lm = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")
    causal_lm.compile(
        press=keras_hub.press.StreamingLLMPress(
            compression_ratio=0.5, n_sink=4
        ),
    )
    causal_lm.generate("Keras is a", max_length=64)
    ```
    """

    def __init__(self, compression_ratio=0.5, n_sink=4, **kwargs):
        super().__init__(compression_ratio=compression_ratio, **kwargs)
        self.n_sink = n_sink

    def score(self, keys, values, keep_len, padding_mask=None):
        shape = ops.shape(keys)
        batch_size, num_layers, seq_len, num_heads = (
            shape[0],
            shape[1],
            shape[2],
            shape[3],
        )
        n_sink = min(self.n_sink, keep_len)
        n_recent = max(keep_len - n_sink, 0)
        positions = ops.arange(seq_len)
        if padding_mask is None:
            is_kept = ops.logical_or(
                positions < n_sink, positions >= (seq_len - n_recent)
            )
            score = ops.where(
                is_kept,
                ops.ones_like(positions, dtype="float32"),
                ops.full_like(ops.cast(positions, "float32"), _MASKED_SCORE),
            )
            score = ops.reshape(score, (1, 1, 1, seq_len))
            return ops.broadcast_to(
                score, (batch_size, num_layers, num_heads, seq_len)
            )

        # "Recent" means the tail of each row's *real* prompt, not the tail
        # of the padded buffer -- otherwise, for any prompt shorter than
        # the buffer, the "recent" window falls entirely on not-yet-written
        # padding, and `compress()`'s later padding mask discards it,
        # silently reducing this press to "sink tokens only" plus
        # arbitrary tie-broken real tokens. Padding positions still get
        # masked out afterwards regardless of what we compute for them
        # here, so it's fine that `is_recent` may also be true for them.
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        row_lengths = ops.reshape(row_lengths, (batch_size, 1))
        positions_row = ops.reshape(positions, (1, seq_len))
        is_kept = ops.logical_or(
            positions_row < n_sink,
            positions_row >= (row_lengths - n_recent),
        )
        score = ops.where(
            is_kept,
            ops.ones_like(is_kept, dtype="float32"),
            ops.full_like(ops.cast(is_kept, "float32"), _MASKED_SCORE),
        )
        score = ops.reshape(score, (batch_size, 1, 1, seq_len))
        return ops.broadcast_to(
            score, (batch_size, num_layers, num_heads, seq_len)
        )

    def get_config(self):
        config = super().get_config()
        config.update({"n_sink": self.n_sink})
        return config
