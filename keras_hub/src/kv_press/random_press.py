from keras import ops
from keras import random

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.kv_press.press import KVCachePress


@keras_hub_export("keras_hub.press.RandomPress")
class RandomPress(KVCachePress):
    """Evicts a random subset of tokens per layer and head.

    Mainly useful as a baseline for comparing other `KVCachePress` methods.

    Args:
        compression_ratio: float. The fraction of prompt tokens to evict,
            in `[0, 1)`. Defaults to `0.5`.
        seed: int. Optional random seed.

    Example:
    ```python
    causal_lm = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")
    causal_lm.compile(
        press=keras_hub.press.RandomPress(compression_ratio=0.5)
    )
    causal_lm.generate("Keras is a", max_length=64)
    ```
    """

    def __init__(self, compression_ratio=0.5, seed=None, **kwargs):
        super().__init__(compression_ratio=compression_ratio, **kwargs)
        self.seed = seed
        self.seed_generator = random.SeedGenerator(seed)

    def score(self, keys, values, keep_len, padding_mask=None):
        shape = ops.shape(keys)
        batch_size, num_layers, seq_len, num_heads = (
            shape[0],
            shape[1],
            shape[2],
            shape[3],
        )
        return random.uniform(
            (batch_size, num_layers, num_heads, seq_len),
            seed=self.seed_generator,
        )

    def get_config(self):
        config = super().get_config()
        config.update({"seed": self.seed})
        return config
