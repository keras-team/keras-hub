import numpy as np
from keras import ops

try:
    import tensorflow as tf
except ImportError:
    tf = None

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

    Supports both text-only and multimodal (image) prompts.  Vision
    embeddings are pre-scaled by `1/sqrt(hidden_dim)` before interleaving so
    that the global `embed_scale` factor does not distort them.

    Args:
        preprocessor: A `keras_hub.models.Gemma4BlockDiffusionLMPreprocessor`
            or `None`.
        backbone: A `keras_hub.models.Gemma4Backbone` instance.
        canvas_length: int. Number of tokens in the denoising canvas.
            Defaults to `256`.
        max_denoising_steps: int. Maximum number of denoising iterations per
            canvas block. Defaults to `48`.
        t_min: float. Minimum sampling temperature applied at the last
            denoising step. Defaults to `0.4`.
        t_max: float. Maximum sampling temperature applied at the first
            denoising step. Defaults to `0.8`.
        **kwargs: Additional keyword arguments passed to the parent class.

    Examples:

    Text generation from a text prompt.
    ```python
    model = keras_hub.models.Gemma4BlockDiffusionLM.from_preset(
        "diffusion_gemma_26b_a4b_it",
    )
    model.generate("The quick brown fox")
    ```

    Image + text generation.
    ```python
    model = keras_hub.models.Gemma4BlockDiffusionLM.from_preset(
        "diffusion_gemma_26b_a4b_it",
    )
    model.generate({
        "prompts": "Describe this image: <|image|>",
        "images": image_array,  # np.ndarray of shape (H, W, 3)
    })
    ```
    """

    backbone_cls = Gemma4Backbone
    preprocessor_cls = Gemma4BlockDiffusionLMPreprocessor

    def __init__(
        self,
        preprocessor,
        backbone,
        **kwargs,
    ):
        # === Layers ===
        self.preprocessor = preprocessor
        self.backbone = backbone

        # === Functional Model ===
        inputs = backbone.input
        hidden = backbone(inputs)
        outputs = self._canvas_logits(hidden)

        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    def _normalize_generate_inputs(self, inputs):
        """Overrides the base class to handle unbatched multimodal inputs."""
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

            # If prompt is scalar, images can be either a 3-D NumPy array /
            # Tensor, or a list of 3-D arrays. Uprank images accordingly.
            if input_is_scalar and "images" in inputs:
                x = inputs["images"]
                if isinstance(x, np.ndarray) and len(x.shape) == 3:
                    inputs["images"] = [x]
                elif tf and isinstance(x, tf.Tensor) and x.shape.rank == 3:
                    inputs["images"] = x[tf.newaxis]
                elif isinstance(x, list):
                    inputs["images"] = [x]
        else:
            inputs, input_is_scalar = normalize(inputs)

        return [inputs], input_is_scalar

    def get_config(self):
        config = super().get_config()
        return config

    def _encode_prompt(self, inputs):
        token_ids = inputs["token_ids"]
        padding_mask = inputs.get("padding_mask", None)

        pixel_values = inputs.get("pixel_values", None)
        pixel_position_ids = inputs.get("pixel_position_ids", None)
        vision_indices = inputs.get("vision_indices", None)
        vision_mask = inputs.get("vision_mask", None)

        # `generate_preprocess` appends `canvas_length` mask tokens to
        # `token_ids` for the generation loop.  Strip them here so the encoder
        # computes KV cache over prompt tokens only.
        if (
            hasattr(self, "canvas_length")
            and self.canvas_length
            and ops.shape(token_ids)[1] > self.canvas_length
        ):
            canvas_len = self.canvas_length
            token_ids = token_ids[:, :-canvas_len]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :-canvas_len]
            if vision_mask is not None:
                vision_mask = vision_mask[:, :-canvas_len]

        # Text embeddings are unscaled until after vision interleaving.
        x = self.backbone.token_embedding(token_ids)
        embed_scale = ops.cast(
            ops.sqrt(ops.cast(self.backbone.hidden_dim, "float32")), x.dtype
        )

        # Interleave vision embeddings (images).
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
            max(head_dim, global_head_dim)
            if global_head_dim is not None
            else head_dim
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

    def _encode_canvas_as_context(
        self, canvas_token_ids, encoder_kv_cache, context_length
    ):
        """Incrementally extend the encoder KV cache with canvas tokens.

        Encodes only the new `canvas_length` tokens — not the full growing
        prompt — by starting from the existing KV cache at
        `cache_update_index=context_length`.  This reduces the per-canvas
        encoder cost from O(context_length) to O(canvas_length), converting
        the multi-canvas generation loop from O(n²) to O(n · canvas_length).

        No vision processing is performed: image embeddings are consumed once
        in `_encode_prompt` and never re-injected on subsequent canvas blocks,
        matching the HuggingFace DiffusionGemmaGenerationMixin behaviour.

        Args:
            canvas_token_ids: int tensor of shape `(B, canvas_length)`.
            encoder_kv_cache: float tensor of shape
                `(B, num_layers, 2, context_length, num_heads, head_dim)`.
            context_length: int scalar; number of tokens already encoded.

        Returns:
            Extended KV cache of shape
            `(B, num_layers, 2, context_length + canvas_length, ...)`.
        """
        x = self.backbone.token_embedding(canvas_token_ids)
        embed_scale = ops.cast(
            ops.sqrt(ops.cast(self.backbone.hidden_dim, "float32")), x.dtype
        )
        x = x * embed_scale

        canvas_length = ops.shape(canvas_token_ids)[1]

        # Extend the existing encoder KV cache to make room for canvas KVs.
        paddings = [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, canvas_length],
            [0, 0],
            [0, 0],
        ]
        extended_cache = ops.pad(encoder_kv_cache, paddings)

        caches = []
        for i, layer in enumerate(self.backbone.transformer_layers):
            current_cache = extended_cache[:, i, ...]
            shared_kv = None
            if (
                layer.is_kv_shared_layer
                and layer.kv_shared_layer_index is not None
            ):
                idx = layer.kv_shared_layer_index
                if idx < len(caches):
                    shared_kv = caches[idx]
                else:
                    shared_kv = extended_cache[:, idx, ...]

            x, next_cache = layer(
                x,
                cache=current_cache,
                cache_update_index=context_length,
                shared_kv=shared_kv,
                use_encoder_scalar=True,
            )
            caches.append(next_cache)

        return ops.stack(caches, axis=1)

    def _prepare_canvas_embeds(self, canvas, prev_logits):
        x = self.backbone.token_embedding(canvas)
        embed_scale = ops.cast(
            ops.sqrt(ops.cast(self.backbone.hidden_dim, "float32")), x.dtype
        )
        x = x * embed_scale

        sc = getattr(self.backbone, "diffusion_self_conditioning", None)
        if sc is not None:
            x = sc(
                x,
                prev_logits,
                self.backbone.token_embedding.embeddings,
                embed_scale,
            )
        return x

    def _prepare_encoder_cache_for_decoding(self, encoder_cache):
        paddings = [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, self.canvas_length],
            [0, 0],
            [0, 0],
        ]
        return ops.pad(encoder_cache, paddings)

    def _decode_canvas_step(
        self, canvas_embeds, encoder_kv_cache, prompt_length
    ):
        x = canvas_embeds
        batch_size = ops.shape(x)[0]
        canvas_length = ops.shape(x)[1]

        # Auto-pad encoder KV cache to prompt + canvas length if not pre-padded.
        cache_seq_len = ops.shape(encoder_kv_cache)[3]
        if cache_seq_len < prompt_length + canvas_length:
            pad_len = (prompt_length + canvas_length) - cache_seq_len
            paddings = [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, pad_len],
                [0, 0],
                [0, 0],
            ]
            combined_cache = ops.pad(encoder_kv_cache, paddings)
        else:
            combined_cache = encoder_kv_cache

        # canvas_mask marks every canvas position as bidirectional.
        canvas_mask = ops.ones((batch_size, canvas_length), dtype="bool")

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
        return self.backbone.token_embedding(hidden, reverse=True)
