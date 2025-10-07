from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.rwkv7.rwkv7_backbone import RWKV7Backbone
from keras_hub.src.models.rwkv7.rwkv7_causal_lm_preprocessor import (
    RWKV7CausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.RWKV7CausalLM")
class RWKV7CausalLM(CausalLM):
    backbone_cls = RWKV7Backbone
    preprocessor_cls = RWKV7CausalLMPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        super().__init__(
            inputs=backbone.inputs,
            outputs=backbone.outputs,
            **kwargs,
        )
        self.call(ops.ones([1, 16], "int32"))

    def call_with_cache(
        self,
        token_ids,
        cache,
        compute_head=True,
        padding_mask=None,
        rnn_mode=True,
    ):
        state_cachce, last_token_cache = cache
        x = self.backbone.token_embedding(token_ids)
        if padding_mask is None:
            padding_mask = ops.not_equal(token_ids, 0)
        v_first = None
        updated_state_cachce = []
        updated_last_token_cache = []

        for i in range(self.backbone.num_layers):
            current_state_cache = state_cachce[:, i, ...]
            current_token_cache = last_token_cache[:, i, ...]
            x, v_first, new_cache_state, cache_tmix_x, cache_cmix_x = (
                self.backbone.rwkv_layers[i].call(
                    x,
                    v_first=v_first,
                    padding_mask=padding_mask,
                    cache_state=current_state_cache,
                    cache_tmix_x=current_token_cache[:, 0],
                    cache_cmix_x=current_token_cache[:, 1],
                    rnn_mode=rnn_mode,
                    train_mode=False,
                )
            )
            new_token_cache = ops.stack([cache_tmix_x, cache_cmix_x], axis=1)
            updated_state_cachce.append(new_cache_state)
            updated_last_token_cache.append(new_token_cache)
        cache = [
            ops.stack(updated_state_cachce, axis=1),
            ops.stack(updated_last_token_cache, axis=1),
        ]
        hidden_states = x = self.backbone.output_layer_norm(x)
        if compute_head:
            logits = self.backbone.head(x)
        else:
            logits = None
        return logits, hidden_states, cache

    def _build_cache(self, token_ids):
        """Build an empty cache for use with `call_with_cache()`."""
        batch_size = ops.shape(token_ids)[0]
        num_layers = self.backbone.num_layers
        head_dim = self.backbone.head_size
        hidden_size = self.backbone.hidden_size
        num_heads = hidden_size // head_dim

        state_cachce = ops.zeros(
            [batch_size, num_layers, num_heads, head_dim, head_dim],
            dtype=self.compute_dtype,
        )
        last_token_cache = ops.zeros(
            [batch_size, num_layers, 2, 1, hidden_size],
            dtype=self.compute_dtype,
        )
        cache = [state_cachce, last_token_cache]

        # Seed the cache.
        # prefill阶段可以使用kernel,要快一点
        _, hidden_states, cache = self.call_with_cache(
            token_ids,
            cache,
            rnn_mode=False,
            compute_head=False,
        )

        return hidden_states, cache

    def generate_step(
        self,
        inputs,
        stop_token_ids=None,
    ):
        """A compilable generation function for a single batch of inputs.

        This function represents the inner, XLA-compilable, generation function
        for a single batch of inputs. Inputs should have the same structure as
        model inputs, a dictionary with keys `"token_ids"` and `"padding_mask"`.

        Args:
            inputs: A dictionary with two keys `"token_ids"` and
                `"padding_mask"` and batched tensor values.
            stop_token_ids: Tuple of id's of the end token to stop on. If all
                sequences have produced a new stop token, generation
                will stop.
        """
        token_ids, padding_mask, predict_token_ids = (
            inputs["token_ids"],
            inputs["padding_mask"],
            inputs["predict_token_ids"],
        )
        # Create and seed cache with a single forward pass.

        hidden_states, cache = self._build_cache(token_ids)

        def next(prompt, cache, index):
            # The cache index is the index of our previous token.
            cache_update_index = index - 1
            batch_size = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, cache_update_index], [batch_size, 1])
            logits, hidden_states, cache = self.call_with_cache(
                prompt,
                cache,
            )
            return (
                ops.squeeze(logits, axis=1),
                ops.squeeze(hidden_states, axis=1),
                cache,
            )

        output_ids = self.sampler(
            next=next,
            prompt=predict_token_ids,
            cache=cache,
            index=1,
            mask=padding_mask,
            stop_token_ids=stop_token_ids,
            hidden_states=hidden_states,
            model=self,
        )
        padding_mask = ops.concatenate(
            [
                ops.cast(ops.not_equal(token_ids, 0), padding_mask.dtype),
                padding_mask,
            ],
            axis=1,
        )
        token_ids = ops.concatenate([token_ids, output_ids], axis=1)

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
