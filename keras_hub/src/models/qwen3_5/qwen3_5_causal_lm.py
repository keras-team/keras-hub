import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.qwen3_5.qwen3_5_backbone import Qwen3_5Backbone
from keras_hub.src.models.qwen3_5.qwen3_5_causal_lm_preprocessor import (
    Qwen3_5CausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.Qwen3_5CausalLM")
class Qwen3_5CausalLM(CausalLM):
    """An end-to-end Qwen3.5 model for causal language modeling.

    This model predicts the next token based on previous tokens using the
    Qwen3.5 hybrid architecture (full attention + GatedDeltaNet linear
    attention layers). It optionally supports multimodal (image + text)
    inputs when the backbone has a ``vision_encoder`` attached.

    This model has a ``generate()`` method for autoregressive text
    generation.

    Args:
        backbone: A ``keras_hub.models.Qwen3_5Backbone`` instance.
        preprocessor: A ``keras_hub.models.Qwen3_5CausalLMPreprocessor``
            or ``None``.
    """

    backbone_cls = Qwen3_5Backbone
    preprocessor_cls = Qwen3_5CausalLMPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
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

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
        padding_mask=None,
        img_embeddings=None,
        vision_indices=None,
        position_ids=None,
    ):
        """Forward pass with cache for autoregressive decoding.

        Only full_attention layers use the KV cache. Linear attention
        layers (GatedDeltaNet) do not cache and process tokens
        independently per step.

        Args:
            token_ids: Dense int tensor (batch_size, max_length).
            cache: Tuple of (kv_cache, conv_cache, recurrent_cache).
            cache_update_index: Int or int tensor, current step index.
            padding_mask: Optional padding mask.
            img_embeddings: Optional vision embeddings tensor
                (total_vision_tokens, hidden_dim). Only used on the
                first (prefill) call to interleave into text embeddings.
            vision_indices: Optional int tensor (total_vision_tokens,).
                Flat indices for scattering vision tokens.
            position_ids: Optional M-RoPE position IDs tensor
                (batch, 4, seq_len). Only provided for multimodal inputs.

        Returns:
            (logits, hidden_states, cache) tuple.
        """
        x = self.backbone.token_embedding(token_ids)

        # Interleave vision embeddings on the prefill step.
        if img_embeddings is not None and vision_indices is not None:
            if hasattr(self.backbone, "interleave_embeddings"):
                x = self.backbone.interleave_embeddings(
                    image_embeddings=img_embeddings,
                    text_embeddings=x,
                    vision_indices=vision_indices,
                )

        # We need three separate lists because XLA requires tensors to
        # have consistent shapes. KV cache, Conv cache, and Recurrent cache
        # all have completely different shapes and cannot be stacked.
        kv_cache = cache[0]
        conv_cache = cache[1]
        recurrent_cache = cache[2]

        next_kv_cache = []
        next_conv_cache = []
        next_recurrent_cache = []

        for i in range(self.backbone.num_layers):
            layer = self.backbone.transformer_layers[i]
            x, next_kv, next_conv, next_recurrent = layer.call_and_update_cache(
                x,
                kv_cache=kv_cache[:, i, ...],
                conv_cache=conv_cache[:, i, ...],
                recurrent_cache=recurrent_cache[:, i, ...],
                cache_update_index=cache_update_index,
                decoder_padding_mask=padding_mask,
                position_ids=position_ids,
            )
            next_kv_cache.append(next_kv)
            next_conv_cache.append(next_conv)
            next_recurrent_cache.append(next_recurrent)

        # Stack caches along the layer dimension
        next_cache = (
            ops.stack(next_kv_cache, axis=1),
            ops.stack(next_conv_cache, axis=1),
            ops.stack(next_recurrent_cache, axis=1),
        )

        hidden_states = x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, hidden_states, next_cache

    def _build_cache(
        self,
        token_ids,
        padding_mask,
        img_embeddings=None,
        vision_indices=None,
        position_ids=None,
    ):
        """Build an empty cache for use with ``call_with_cache()``."""
        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]
        num_layers = self.backbone.num_layers
        num_kv_heads = self.backbone.num_key_value_heads
        head_dim = self.backbone.head_dim

        # KV Cache shape: (B, num_layers, 2, seq_len, num_kv_heads, head_dim)
        kv_shape = [
            batch_size,
            num_layers,
            2,
            max_length,
            num_kv_heads,
            head_dim,
        ]
        kv_cache = ops.zeros(kv_shape, dtype=self.compute_dtype)

        # Conv cache shape: (B, num_layers, conv_dim, conv_kernel_size - 1)
        linear_key_dim = (
            self.backbone.linear_num_key_heads
            * self.backbone.linear_key_head_dim
        )
        linear_val_dim = (
            self.backbone.linear_num_value_heads
            * self.backbone.linear_value_head_dim
        )
        conv_dim = linear_key_dim * 2 + linear_val_dim
        conv_shape = [
            batch_size,
            num_layers,
            conv_dim,
            self.backbone.linear_conv_kernel_dim - 1,
        ]
        conv_cache = ops.zeros(conv_shape, dtype=self.compute_dtype)
        recurrent_shape = [
            batch_size,
            num_layers,
            self.backbone.linear_num_value_heads,
            self.backbone.linear_key_head_dim,
            self.backbone.linear_value_head_dim,
        ]
        recurrent_cache = ops.zeros(recurrent_shape, dtype="float32")

        cache = (kv_cache, conv_cache, recurrent_cache)
        # Seed the cache with a full forward pass, including vision
        # embeddings on the first call.
        _, hidden_states, cache = self.call_with_cache(
            token_ids,
            cache,
            0,
            padding_mask=padding_mask,
            img_embeddings=img_embeddings,
            vision_indices=vision_indices,
            position_ids=position_ids,
        )
        return hidden_states, cache

    def generate_step(self, inputs, stop_token_ids=None):
        """A compilable generation function for a single batch.

        For multimodal inputs, the preprocessor populates extra keys like
        ``pixel_values``, ``image_grid_thw``, ``vision_indices``, and
        ``position_ids``. These are consumed on the first forward pass
        (cache prefill) and then dropped for subsequent autoregressive
        steps.
        """
        token_ids = inputs["token_ids"]
        padding_mask = inputs["padding_mask"]

        # Check for multimodal inputs.
        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)
        vision_indices = inputs.get("vision_indices", None)
        position_ids = inputs.get("position_ids", None)

        # Run vision encoder if present and we have pixel data.
        img_embeddings = None
        if (
            self.backbone.vision_encoder is not None
            and pixel_values is not None
        ):
            img_embeddings = self.backbone.vision_encoder(
                pixel_values, image_grid_thw
            )

        hidden_states, cache = self._build_cache(
            token_ids,
            padding_mask,
            img_embeddings=img_embeddings,
            vision_indices=vision_indices,
            position_ids=position_ids,
        )
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            cache_update_index = index - 1
            batch_size = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, cache_update_index], [batch_size, 1])
            # No vision inputs during autoregressive generation — they
            # are already baked into the KV cache from the prefill step.
            logits, hidden_states, cache = self.call_with_cache(
                prompt, cache, cache_update_index, padding_mask=None
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

        if stop_token_ids is not None:
            end_locations = any_equal(
                token_ids,
                stop_token_ids,
                ops.logical_not(padding_mask),
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

    def score(
        self,
        token_ids,
        padding_mask=None,
        scoring_mode="logits",
        layer_intercept_fn=None,
        target_ids=None,
    ):
        """Score a generation represented by the provided token ids.

        Accepts either a plain ``token_ids`` tensor (text-only) or a dict
        with ``token_ids``, ``padding_mask``, and optional multimodal keys
        (``pixel_values``, ``image_grid_thw``, ``vision_indices``,
        ``position_ids``).
        """
        if scoring_mode not in ("logits", "loss"):
            raise ValueError(
                "Unsupported scoring_mode. Must be 'logits' or 'loss'."
            )
        if scoring_mode == "loss" and target_ids is None:
            raise ValueError(
                "Cannot compute loss without targets. Please provide "
                "target token ids via the target_ids parameter."
            )

        # Unpack multimodal dict inputs if provided.
        pixel_values = None
        image_grid_thw = None
        vision_indices = None
        if isinstance(token_ids, dict):
            padding_mask = token_ids.get("padding_mask", padding_mask)
            pixel_values = token_ids.get("pixel_values", None)
            image_grid_thw = token_ids.get("image_grid_thw", None)
            vision_indices = token_ids.get("vision_indices", None)
            token_ids = token_ids["token_ids"]

        batch_shape = ops.shape(token_ids)[:2]
        assert len(batch_shape) == 2

        if padding_mask is None:
            padding_mask = ops.ones(shape=batch_shape)

        if layer_intercept_fn is None:

            def default_layer_intercept_fn(x, unused_i):
                return x

            layer_intercept_fn = default_layer_intercept_fn

        token_embeddings = self.backbone.token_embedding(token_ids)

        # Interleave vision embeddings if multimodal inputs are present.
        if (
            self.backbone.vision_encoder is not None
            and pixel_values is not None
        ):
            img_embeddings = self.backbone.vision_encoder(
                pixel_values, image_grid_thw
            )
            token_embeddings = self.backbone.interleave_embeddings(
                image_embeddings=img_embeddings,
                text_embeddings=token_embeddings,
                vision_indices=vision_indices,
            )

        x = layer_intercept_fn(token_embeddings, -1)

        for i, transformer_layer in enumerate(self.backbone.transformer_layers):
            x = transformer_layer(x, decoder_padding_mask=padding_mask)
            x = layer_intercept_fn(x, i)

        x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)

        if scoring_mode == "logits":
            return logits

        per_token_loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        per_token_loss = per_token_loss_fn(target_ids, logits)
        return per_token_loss
