from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.kv_press.press import KVCachePress


@keras_hub_export("keras_hub.press.KnormPress")
class KnormPress(KVCachePress):
    """Evicts tokens with the largest key-vector norm, per layer and head.

    Follows the observation from the KVPress line of work that KV pairs
    whose key embeddings have a smaller L2 norm tend to be more important to
    retain.

    Args:
        compression_ratio: float. The fraction of prompt tokens to evict,
            in `[0, 1)`. Defaults to `0.5`.

    Example:
    ```python
    causal_lm = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")
    causal_lm.compile(
        press=keras_hub.press.KnormPress(compression_ratio=0.5)
    )
    causal_lm.generate("Keras is a", max_length=64)
    ```
    """

    def score(self, keys, values, keep_len, padding_mask=None):
        # `keys` has shape (batch, num_layers, seq_len, num_heads, head_dim).
        norm = ops.norm(keys, axis=-1)
        # Lower norm is more important, so score is the negative norm.
        # Move to (batch, num_layers, num_heads, seq_len).
        return ops.transpose(-norm, axes=(0, 1, 3, 2))
