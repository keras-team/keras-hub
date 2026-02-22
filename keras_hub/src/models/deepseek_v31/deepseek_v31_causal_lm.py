"""DeepSeek V31 Causal Language Model."""

import keras
from keras import ops
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.deepseek_v31.deepseek_v31_backbone import (
    DeepSeekV31Backbone,
)
from keras_hub.src.models.deepseek_v31.deepseek_v31_causal_lm_preprocessor import (
    DeepSeekV31CausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.DeepSeekV31CausalLM")
class DeepSeekV31CausalLM(CausalLM):
    """DeepSeek V31 Causal Language Model.

    Pairs `DeepSeekV31Backbone` with a language model head for next-token
    prediction. The LM head reuses the token embedding weights via
    `ReversibleEmbedding` (weight tying is off by default for DeepSeek V31,
    but the same layer is used for both embedding lookup and logit projection).

    Autoregressive generation uses an MLA-compatible KV cache. Rather than
    storing full per-head K and V tensors at each step, only the compressed
    latents `c_kv` (shape `[batch, max_len, kv_lora_rank]`) and decoupled RoPE
    keys `k_rope` (shape `[batch, max_len, qk_rope_head_dim]`) are cached per
    layer. This significantly reduces memory usage compared to standard MHA
    caching.

    Args:
        backbone: `DeepSeekV31Backbone` instance. The core transformer model.
        preprocessor: Optional `DeepSeekV31CausalLMPreprocessor`. Used for
            tokenizing inputs before passing to the model.

    Example:

    ```python
    backbone = keras_hub.models.DeepSeekV31Backbone(...)
    lm = keras_hub.models.DeepSeekV31CausalLM(backbone=backbone)
    output = lm.generate("Once upon a time")
    ```

    Reference:
     - [DeepSeek-AI et al., 2024](https://arxiv.org/abs/2412.19437)
    """

    backbone_cls = DeepSeekV31Backbone
    preprocessor_cls = DeepSeekV31CausalLMPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        inputs = backbone.input
        hidden_states = backbone(inputs)
        outputs = backbone.token_embedding(hidden_states, reverse=True)

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        self.backbone = backbone
        self.preprocessor = preprocessor

    def _build_cache(self, token_ids):
        """Build an empty MLA KV cache for a given token_ids tensor.

        Returns a list of `(c_kv, k_rope)` tuples, one per transformer layer,
        each pre-allocated to the full sequence length. The cache is written
        to incrementally during autoregressive generation.
        """
        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]
        cache = []
        for _ in range(self.backbone.num_layers):
            cache.append(
                (
                    ops.zeros(
                        [batch_size, max_length, self.backbone.kv_lora_rank],
                        dtype=self.compute_dtype,
                    ),
                    ops.zeros(
                        [
                            batch_size,
                            max_length,
                            self.backbone.qk_rope_head_dim,
                        ],
                        dtype=self.compute_dtype,
                    ),
                )
            )
        return cache

    def call_with_cache(self, token_ids, cache, cache_update_index):
        """Forward pass with explicit KV cache read/write.

        Threads the cache through each transformer layer by calling sub-layers
        directly, bypassing the Keras functional model graph (which does not
        support dynamic cache arguments). This is the standard KerasHub pattern
        for cached autoregressive generation.

        Args:
            token_ids: int32 tensor of shape `(batch, 1)` for single-step
                decoding or `(batch, seq_len)` for prefill.
            cache: list of `(c_kv, k_rope)` tuples from `_build_cache`.
            cache_update_index: int. Position index into the cache to write the
                current token's KV entries.

        Returns:
            A `(logits, hidden_states, new_cache)` tuple where `logits` has
            shape `(batch, seq_len, vocabulary_size)`, `hidden_states` has
            shape `(batch, seq_len, hidden_dim)`, and `new_cache` is the
            updated list of `(c_kv, k_rope)` tuples.
        """
        x = self.backbone.token_embedding(token_ids)
        new_cache = []
        for i, layer in enumerate(self.backbone.transformer_layers):
            x, updated_cache = layer(
                x,
                cache=cache[i],
                cache_update_index=cache_update_index,
            )
            new_cache.append(updated_cache)

        x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, x, new_cache

    def generate_step(self, inputs, stop_token_ids=None):
        """XLA-compilable single-batch generation step.

        Args:
            inputs: dict with keys `"token_ids"` (int32 tensor) and
                `"padding_mask"` (bool tensor).
            stop_token_ids: tuple of int token ids. Generation stops when all
                sequences have produced at least one stop token.

        Returns:
            Updated `inputs` dict with the same keys.
        """
        token_ids, padding_mask = inputs["token_ids"], inputs["padding_mask"]
        cache = self._build_cache(token_ids)

        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            cache_update_index = index - 1
            batch_size = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, cache_update_index], [batch_size, 1])
            logits, hidden_states, cache = self.call_with_cache(
                prompt, cache, cache_update_index
            )
            return (
                ops.squeeze(logits, axis=1),
                ops.squeeze(hidden_states, axis=1),
                cache,
            )

        token_ids = self.sampler(
            next=next,
            prompt=token_ids,
            cache=cache,
            index=index,
            mask=padding_mask,
            stop_token_ids=stop_token_ids,
            hidden_states=ops.zeros(
                [ops.shape(token_ids)[0], self.backbone.hidden_dim],
                dtype=self.compute_dtype,
            ),
            model=self,
        )

        if stop_token_ids is not None:
            end_locations = any_equal(
                token_ids, stop_token_ids, ops.logical_not(padding_mask)
            )
            end_locations = ops.cast(end_locations, "int32")
            cumsum = ops.cast(ops.cumsum(end_locations, axis=-1), "int32")
            overflow = cumsum - end_locations
            padding_mask = ops.logical_not(ops.cast(overflow, "bool"))
        else:
            padding_mask = ops.ones_like(token_ids, dtype="bool")

        return {"token_ids": token_ids, "padding_mask": padding_mask}
