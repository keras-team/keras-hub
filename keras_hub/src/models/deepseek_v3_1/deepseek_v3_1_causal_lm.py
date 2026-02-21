"""DeepSeek V3.1 Causal Language Model."""

import keras
from keras import ops
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.deepseek_v3_1.deepseek_v3_1_backbone import (
    DeepSeekV3_1Backbone,
)
from keras_hub.src.models.deepseek_v3_1.deepseek_v3_1_causal_lm_preprocessor import (
    DeepSeekV3_1CausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.DeepSeekV3_1CausalLM")
class DeepSeekV3_1CausalLM(CausalLM):
    """DeepSeek V3.1 Causal Language Model.

    Wraps `DeepSeekV3_1Backbone` with a language model head (tied to the
    token embedding via `ReversibleEmbedding`) for next-token prediction.

    Generation uses an MLA-compatible KV cache: instead of storing full
    per-head K/V tensors, only the compressed latents `c_kv` (shape:
    [batch, seq, kv_lora_rank]) and decoupled RoPE keys `k_rope` (shape:
    [batch, seq, qk_rope_head_dim]) are cached. This significantly reduces
    memory usage during autoregressive generation.

    Args:
        backbone: `DeepSeekV3_1Backbone` instance.
        preprocessor: Optional `DeepSeekV3_1CausalLMPreprocessor` instance.
    """

    backbone_cls = DeepSeekV3_1Backbone
    preprocessor_cls = DeepSeekV3_1CausalLMPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        # Reuse the backbone's exact input specification and call it
        # symbolically to build the connected functional graph.
        inputs = backbone.input
        hidden_states = backbone(inputs)
        outputs = backbone.token_embedding(hidden_states, reverse=True)

        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        self.backbone = backbone
        self.preprocessor = preprocessor

    def _build_cache(self, token_ids):
        """Build an empty MLA KV cache sized to the input sequence length.

        Returns:
            cache: list of (c_kv, k_rope) tuples, one per transformer layer.
                - c_kv shape:   (batch, max_len, kv_lora_rank)
                - k_rope shape: (batch, max_len, qk_rope_head_dim)

        Note: Unlike standard MHA which caches (K, V) per head, MLA only
        needs to cache the compressed latents. This is the key efficiency
        advantage of the MLA architecture.
        """
        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]

        cache = []
        for _ in range(self.backbone.num_layers):
            c_kv = ops.zeros(
                [batch_size, max_length, self.backbone.kv_lora_rank],
                dtype=self.compute_dtype,
            )
            k_rope = ops.zeros(
                [batch_size, max_length, self.backbone.qk_rope_head_dim],
                dtype=self.compute_dtype,
            )
            cache.append((c_kv, k_rope))

        return cache

    def call_with_cache(self, token_ids, cache, cache_update_index):
        """Run a single forward pass with KV cache read/write.

        Directly accesses backbone sub-layers to thread the cache through
        each transformer layer. This bypasses the Keras functional model
        (which doesn't support dynamic cache arguments) but is the standard
        pattern for cached generation in KerasHub.

        Args:
            token_ids: int32 tensor of shape (batch, 1) during decoding,
                or (batch, seq_len) for the initial prefill.
            cache: list of (c_kv, k_rope) tuples from `_build_cache`.
            cache_update_index: int. Position in the cache to write to.

        Returns:
            logits: float tensor of shape (batch, seq_len, vocab_size).
            hidden_states: float tensor of shape (batch, seq_len, hidden_dim).
            new_cache: updated list of (c_kv, k_rope) tuples.
        """
        x = self.backbone.token_embedding(token_ids)
        new_cache = []

        for i in range(self.backbone.num_layers):
            layer_cache = cache[i] if cache is not None else None
            x, updated_layer_cache = self.backbone.transformer_layers[i](
                x, cache=layer_cache, cache_update_index=cache_update_index
            )
            new_cache.append(updated_layer_cache)

        x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, x, new_cache

    def generate_step(self, inputs, stop_token_ids=None):
        """A compilable generation function for a single batch of inputs.

        This function represents the inner, XLA-compilable generation function
        for a single batch of inputs. Inputs should have the same structure as
        model inputs, a dictionary with keys `"token_ids"` and `"padding_mask"`.

        Args:
            inputs: A dictionary with two keys `"token_ids"` and
                `"padding_mask"` and batched tensor values.
            stop_token_ids: Tuple of id's of the end token to stop on. If all
                sequences have produced a new stop token, generation
                will stop.
        """
        token_ids, padding_mask = inputs["token_ids"], inputs["padding_mask"]

        # Build empty cache sized to the full sequence length
        cache = self._build_cache(token_ids)

        # Compute the lengths of all user inputted token ids.
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        # Start at the first index that has no user inputted id.
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            # cache_update_index is the position of the token being generated
            cache_update_index = index - 1
            batch_size = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, cache_update_index], [batch_size, 1])
            logits, hidden_states, cache = self.call_with_cache(
                prompt,
                cache,
                cache_update_index,
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

        # Compute an output padding mask with the token ids we updated.
        if stop_token_ids is not None:
            # Build a mask of stop token locations not in the original
            # prompt (not in locations where `padding_mask` is True).
            end_locations = any_equal(
                token_ids, stop_token_ids, ops.logical_not(padding_mask)
            )
            end_locations = ops.cast(end_locations, "int32")
            # Use cumsum to get ones in all locations after end_locations.
            cumsum = ops.cast(ops.cumsum(end_locations, axis=-1), "int32")
            overflow = cumsum - end_locations
            # Our padding mask is the inverse of these overflow locations.
            padding_mask = ops.logical_not(ops.cast(overflow, "bool"))
        else:
            # Without early stopping, all locations will have been updated.
            padding_mask = ops.ones_like(token_ids, dtype="bool")

        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
