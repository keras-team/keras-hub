from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.block_diffusion_lm import BlockDiffusionLM
from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.models.gemma4.gemma4_block_diffusion_lm_preprocessor import (
    Gemma4BlockDiffusionLMPreprocessor,
)


@keras_hub_export("keras_hub.models.Gemma4BlockDiffusionLM")
class Gemma4BlockDiffusionLM(BlockDiffusionLM):
    """Gemma4-based discrete block-diffusion language model.

    Wraps a `Gemma4Backbone` with the block-diffusion generation loop from
    `BlockDiffusionLM`.  The backbone is called twice per generation iteration:
    once as a causal encoder to freeze prompt KV caches, and up to
    `max_denoising_steps` times as a bidirectional decoder over a fixed-length
    canvas of tokens.

    Supports both text-only and multimodal (image/video) prompts.  Vision
    embeddings are pre-scaled by `1/sqrt(hidden_dim)` before interleaving so
    that the global `embed_scale` factor does not distort them.

    Args:
        backbone: A `keras_hub.models.Gemma4Backbone` instance.
        preprocessor: A `keras_hub.models.Gemma4BlockDiffusionLMPreprocessor`
            or `None`.
        canvas_length: int. Number of tokens in the denoising canvas.
            Defaults to `256`.
        max_denoising_steps: int. Maximum number of denoising iterations per
            canvas block. Defaults to `48`.
        t_min: float. Minimum sampling temperature applied at the last
            denoising step. Defaults to `0.4`.
        t_max: float. Maximum sampling temperature applied at the first
            denoising step. Defaults to `0.8`.
        **kwargs: Additional keyword arguments passed to the parent class.

    Example:

    >>> backbone = keras_hub.models.Gemma4Backbone.from_preset(
    ...     "gemma4_2b_en"
    ... )
    >>> preprocessor = (
    ...     keras_hub.models.Gemma4BlockDiffusionLMPreprocessor.from_preset(
    ...         "gemma4_2b_en"
    ...     )
    ... )
    >>> model = keras_hub.models.Gemma4BlockDiffusionLM(
    ...     backbone=backbone,
    ...     preprocessor=preprocessor,
    ... )
    >>> model.generate("The quick brown fox")
    """

    backbone_cls = Gemma4Backbone
    preprocessor_cls = Gemma4BlockDiffusionLMPreprocessor

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        self.backbone = backbone
        self.preprocessor = preprocessor
        super().__init__(**kwargs)

    def _encode_prompt(self, inputs):
        token_ids = inputs["token_ids"]
        padding_mask = inputs.get("padding_mask", None)

        pixel_values = inputs.get("pixel_values", None)
        pixel_position_ids = inputs.get("pixel_position_ids", None)
        vision_indices = inputs.get("vision_indices", None)
        vision_mask = inputs.get("vision_mask", None)

        # Text embeddings — kept unscaled until after vision interleaving so
        # that pre-scaled vision embeddings land at the correct magnitude after
        # the global `x = x * embed_scale` step below.
        x = self.backbone.token_embedding(token_ids)
        embed_scale = ops.cast(
            ops.sqrt(ops.cast(self.backbone.hidden_dim, "float32")), x.dtype
        )

        # Interleave vision embeddings (images or video frames).
        # Pre-scale by 1/sqrt(hidden_dim) so that after the global scale the
        # vision positions stay at their natural embed_vision magnitude,
        # matching the pattern in Gemma4CausalLM.call_with_cache().
        num_images = 0
        if (
            pixel_values is not None
            and hasattr(pixel_values, "shape")
            and len(pixel_values.shape) > 1
        ):
            num_images = pixel_values.shape[1]

        if not self.backbone.text_only_model and num_images:
            img_embeddings = self.backbone.vision_encoder(
                {
                    "pixel_values": pixel_values,
                    "pixel_position_ids": pixel_position_ids,
                }
            )
            scaled_img_embeddings = img_embeddings * ops.cast(
                float(self.backbone.hidden_dim) ** -0.5, img_embeddings.dtype
            )
            x = self.backbone.interleave_embeddings(
                image_embeddings=scaled_img_embeddings,
                text_embeddings=x,
                vision_indices=vision_indices,
            )
            vision_mask = ops.cast(vision_mask, "bool")
        else:
            vision_mask = None

        # Global scale applied after interleaving: text positions get
        # sqrt(hidden_dim), vision positions keep their pre-scaled magnitude.
        x = x * embed_scale

        batch_size = ops.shape(token_ids)[0]
        prompt_length = ops.shape(token_ids)[1]
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_key_value_heads
        head_dim = self.backbone.head_dim
        global_head_dim = self.backbone.global_head_dim
        max_head_dim = (
            max(head_dim, global_head_dim) if global_head_dim else head_dim
        )
        cache_shape = [
            batch_size,
            num_layers,
            2,
            prompt_length,
            num_heads,
            max_head_dim,
        ]
        cache = ops.zeros(cache_shape, dtype=self.compute_dtype)

        # The HF encoder does not inject per-layer embeddings — it is a plain
        # residual transformer loop (see `DiffusionGemmaEncoderTextLayer`).
        caches = []
        for i, layer in enumerate(self.backbone.transformer_layers):
            current_cache = cache[:, i, ...]
            shared_kv = None
            if (
                layer.is_kv_shared_layer
                and layer.kv_shared_layer_index is not None
            ):
                idx = layer.kv_shared_layer_index
                if idx < len(caches):
                    shared_kv = caches[idx]
                else:
                    shared_kv = cache[:, idx, ...]

            x, next_cache = layer(
                x,
                cache=current_cache,
                cache_update_index=0,
                padding_mask=padding_mask,
                vision_mask=vision_mask,
                shared_kv=shared_kv,
                use_encoder_scalar=True,
            )
            caches.append(next_cache)

        encoder_kv_cache = ops.stack(caches, axis=1)
        return encoder_kv_cache, prompt_length

    def _prepare_canvas_embeds(self, canvas, prev_logits):
        x = self.backbone.token_embedding(canvas)
        embed_scale = ops.cast(
            ops.sqrt(ops.cast(self.backbone.hidden_dim, "float32")), x.dtype
        )
        x = x * embed_scale

        x = self.backbone.diffusion_self_conditioning(
            x,
            prev_logits,
            self.backbone.token_embedding.embeddings,
            embed_scale,
        )
        return x

    def _decode_canvas_step(
        self, canvas_embeds, encoder_kv_cache, prompt_length
    ):
        x = canvas_embeds
        batch_size = ops.shape(x)[0]
        canvas_length = ops.shape(x)[1]

        # Build a combined cache: the encoder slice is pre-filled; the canvas
        # slice will be written during this forward pass.  Only the sequence
        # axis (axis 3) needs padding; all other dims match the encoder cache.

        # Pad encoder cache to cover prompt + canvas along the sequence axis
        # (axis 3).  encoder_kv_cache shape: (B, L, 2, prompt_len, heads, hd)
        pad_len = canvas_length
        # Pad: (before, after) for each dimension — only pad axis 3.
        paddings = [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, pad_len],
            [0, 0],
            [0, 0],
        ]
        combined_cache = ops.pad(encoder_kv_cache, paddings)

        # canvas_mask marks every canvas position as bidirectional.
        canvas_mask = ops.ones((batch_size, canvas_length), dtype="bool")

        # The HF decoder does not inject per-layer embeddings during the canvas
        # pass — it is a plain residual transformer loop.
        caches = []
        for i, layer in enumerate(self.backbone.transformer_layers):
            current_cache = combined_cache[:, i, ...]
            shared_kv = None
            if (
                layer.is_kv_shared_layer
                and layer.kv_shared_layer_index is not None
            ):
                idx = layer.kv_shared_layer_index
                if idx < len(caches):
                    shared_kv = caches[idx]
                else:
                    shared_kv = combined_cache[:, idx, ...]

            x, next_cache = layer(
                x,
                cache=current_cache,
                cache_update_index=prompt_length,
                canvas_mask=canvas_mask,
                shared_kv=shared_kv,
            )
            caches.append(next_cache)

        return self.backbone.layer_norm(x)

    def _canvas_logits(self, hidden):
        logits = self.backbone.token_embedding(hidden, reverse=True)
        soft_cap = self.backbone.final_logit_soft_cap
        if soft_cap is not None:
            logits = ops.tanh(logits / soft_cap) * soft_cap
        return logits

    def call(self, x, training=False):
        token_ids = x["token_ids"]
        padding_mask = x.get("padding_mask", None)

        backbone_inputs = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "position_ids": ops.expand_dims(
                ops.arange(ops.shape(token_ids)[1], dtype="int32"), axis=0
            ),
        }
        # Pass vision fields to the backbone when a vision encoder is present.
        if not self.backbone.text_only_model:
            for key in (
                "pixel_values",
                "pixel_position_ids",
                "vision_indices",
                "vision_mask",
            ):
                if key in x:
                    backbone_inputs[key] = x[key]
        hidden = self.backbone(backbone_inputs, training=training)
        return self._canvas_logits(hidden)
