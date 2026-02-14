"""Qwen2-VL Causal Language Model.

End-to-end multimodal model for causal language modelling, supporting
both image+text and text-only inputs.
"""

import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import Qwen2VLBackbone
from keras_hub.src.models.qwen2_vl.qwen2_vl_causal_lm_preprocessor import (
    Qwen2VLCausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.Qwen2VLCausalLM")
class Qwen2VLCausalLM(CausalLM):
    """An end-to-end multimodal Qwen2-VL model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens.  This model supports both image+text and text-only inputs.

    This model has a ``generate()`` method, which generates text based on a
    prompt. The generation strategy used is controlled by an additional
    ``sampler`` argument on ``compile()``. You can recompile the model with
    different ``keras_hub.samplers`` objects to control the generation. By
    default, ``"greedy"`` sampling will be used.

    This model can optionally be configured with a ``preprocessor`` layer, in
    which case it will automatically apply preprocessing to string inputs during
    ``fit()``, ``predict()``, ``evaluate()`` and ``generate()``.

    Args:
        preprocessor: A ``keras_hub.models.Qwen2VLCausalLMPreprocessor`` or
            ``None``. If ``None``, this model will not apply preprocessing
            and inputs should be preprocessed before calling the model.
        backbone: A ``keras_hub.models.Qwen2VLBackbone`` instance.
    """

    backbone_cls = Qwen2VLBackbone
    preprocessor_cls = Qwen2VLCausalLMPreprocessor

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        inputs = backbone.input
        hidden_states = backbone(inputs)
        outputs = backbone.token_embedding(hidden_states, reverse=True)

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    def compile(
        self,
        optimizer="auto",
        loss="auto",
        *,
        weighted_metrics="auto",
        sampler="greedy",
        **kwargs,
    ):
        super().compile(
            optimizer=optimizer,
            loss=loss,
            weighted_metrics=weighted_metrics,
            sampler=sampler,
            **kwargs,
        )

    def _normalize_generate_inputs(self, inputs):
        """Handle unbatched image inputs for generation."""
        if tf and isinstance(inputs, tf.data.Dataset):
            return inputs.as_numpy_iterator(), False

        if self.preprocessor is None:
            return [inputs], False

        def normalize(x):
            if isinstance(x, str):
                return [x], True
            if tf and isinstance(x, tf.Tensor) and x.shape.rank == 0:
                return x[tf.newaxis], True
            return x, False

        if isinstance(inputs, dict):
            inputs["prompts"], input_is_scalar = normalize(inputs["prompts"])
            # If prompt is scalar, images can be a single 3D array.
            if input_is_scalar and "images" in inputs:
                x = inputs["images"]
                if isinstance(x, np.ndarray) and len(x.shape) == 3:
                    inputs["images"] = [x]
                elif tf and isinstance(x, tf.Tensor) and x.shape.rank == 3:
                    inputs["images"] = x[tf.newaxis]
                elif isinstance(x, list):
                    inputs["images"] = [x]
            if "responses" in inputs:
                inputs["responses"], _ = normalize(inputs["responses"])
        else:
            inputs, input_is_scalar = normalize(inputs)

        return [inputs], input_is_scalar

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
        padding_mask=None,
        mrope_position_ids=None,
    ):
        """Forward pass with cache for autoregressive inference.

        Args:
            token_ids: Dense int tensor  ``(batch, seq_len)``.
            cache: Dense float tensor with cached key/value states.
            cache_update_index: int or int tensor. Current index in sequence.
            padding_mask: Optional mask ``(batch, seq_len)``.
            mrope_position_ids: Optional tensor ``(batch, seq_len, 3)``
                for M-RoPE position IDs.

        Returns:
            Tuple of ``(logits, hidden_states, cache)``.
        """
        from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import (
            _compute_mrope_embeddings,
        )

        x = self.backbone.token_embedding(token_ids)

        # Compute position embeddings if position ids provided.
        position_embeddings = None
        if mrope_position_ids is not None:
            head_dim = (
                self.backbone.hidden_dim // self.backbone.num_query_heads
            )
            position_embeddings = _compute_mrope_embeddings(
                mrope_position_ids,
                head_dim,
                self.backbone.rope_max_wavelength,
                self.backbone.mrope_section,
            )

        # Each decoder layer has a cache; we update them separately.
        caches = []
        for i, transformer_layer in enumerate(
            self.backbone.transformer_layers
        ):
            current_cache = cache[:, i, ...]
            x, next_cache = transformer_layer(
                x,
                attention_mask=padding_mask,
                position_embeddings=position_embeddings,
                cache=current_cache,
                cache_update_index=cache_update_index,
            )
            caches.append(next_cache)

        cache = ops.stack(caches, axis=1)
        hidden_states = x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, hidden_states, cache

    def _build_cache(self, token_ids, padding_mask=None,
                     mrope_position_ids=None):
        """Build an empty cache for use with ``call_with_cache()``."""
        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_key_value_heads
        head_dim = (
            self.backbone.hidden_dim // self.backbone.num_query_heads
        )
        shape = [batch_size, num_layers, 2, max_length, num_heads, head_dim]
        cache = ops.zeros(shape, dtype=self.compute_dtype)
        # Seed the cache.
        logits, hidden_states, cache = self.call_with_cache(
            token_ids=token_ids,
            cache=cache,
            cache_update_index=0,
            padding_mask=padding_mask,
            mrope_position_ids=mrope_position_ids,
        )
        return hidden_states, cache

    def generate_step(self, inputs, stop_token_ids=None):
        """A compilable generation function for a single batch of inputs.

        Args:
            inputs: A dictionary with keys ``"token_ids"``,
                ``"padding_mask"``, and optionally ``"mrope_position_ids"``.
            stop_token_ids: Tuple of end token IDs to stop on.
        """
        token_ids = inputs["token_ids"]
        padding_mask = inputs["padding_mask"]
        mrope_position_ids = inputs.get("mrope_position_ids", None)

        # Create and seed cache.
        hidden_states, cache = self._build_cache(
            token_ids,
            padding_mask=padding_mask,
            mrope_position_ids=mrope_position_ids,
        )

        # Compute the lengths of all user inputted token ids.
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        # Start at the first index that has no user inputted id.
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            # The cache index is the index of our previous token.
            cache_update_index = index - 1
            batch_size = ops.shape(prompt)[0]
            prompt = ops.slice(
                prompt, [0, cache_update_index], [batch_size, 1]
            )
            logits, hidden_states, cache = self.call_with_cache(
                token_ids=prompt,
                cache=cache,
                cache_update_index=cache_update_index,
                padding_mask=padding_mask,
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
            hidden_states=hidden_states,
            model=self,
        )

        # Compute an output padding mask with the token ids we updated.
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

        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
