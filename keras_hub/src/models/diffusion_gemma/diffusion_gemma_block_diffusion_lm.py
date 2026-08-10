import keras
import numpy as np
from keras import ops

try:
    import tensorflow as tf
except ImportError:
    tf = None

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.block_diffusion_lm import BlockDiffusionLM
from keras_hub.src.models.diffusion_gemma.diffusion_gemma_backbone import (
    DiffusionGemmaBackbone,
)
from keras_hub.src.models.diffusion_gemma.diffusion_gemma_block_diffusion_lm_preprocessor import (  # noqa: E501
    DiffusionGemmaBlockDiffusionLMPreprocessor,
)
from keras_hub.src.samplers.serialization import get as get_sampler


@keras_hub_export("keras_hub.models.DiffusionGemmaBlockDiffusionLM")
class DiffusionGemmaBlockDiffusionLM(BlockDiffusionLM):
    """DiffusionGemma discrete block-diffusion language model.

    Wraps a `DiffusionGemmaBackbone` with the block-diffusion generation loop
    from `BlockDiffusionLM`.  The backbone is called twice per generation
    iteration: once as a causal encoder to freeze prompt KV caches, and up to
    `max_denoising_steps` times as a bidirectional decoder over a fixed-length
    canvas of tokens.

    Supports both text-only and multimodal (image) prompts.  Vision
    embeddings are pre-scaled by `1/sqrt(hidden_dim)` before interleaving so
    that the global `embed_scale` factor does not distort them.

    Args:
        preprocessor: A
            `keras_hub.models.DiffusionGemmaBlockDiffusionLMPreprocessor`
            or `None`.
        backbone: A `keras_hub.models.DiffusionGemmaBackbone` instance.
        canvas_length: int. Number of tokens in the denoising canvas.
            Defaults to `256`.
        max_denoising_steps: int. Maximum number of denoising iterations per
            canvas block. Defaults to `48`.
        t_min: float. Minimum sampling temperature applied at the last
            denoising step. Defaults to `0.4`.
        t_max: float. Maximum sampling temperature applied at the first
            denoising step. Defaults to `0.8`.
        sampler: `"entropy_bound"` or a compatible diffusion sampler.
            Defaults to `"entropy_bound"`.
        stop_token_ids: Optional tuple of token IDs that finish generation.
            Defaults to `None`.
        pad_token_id: Optional int token ID used after the first stop token.
            Defaults to `None`.
        **kwargs: Additional keyword arguments passed to the parent class.

    Examples:

    Text generation from a text prompt.
    ```python
    model = keras_hub.models.DiffusionGemmaBlockDiffusionLM.from_preset(
        "diffusion_gemma_26b_a4b_it",
    )
    model.generate("The quick brown fox")
    ```

    Image + text generation.
    ```python
    model = keras_hub.models.DiffusionGemmaBlockDiffusionLM.from_preset(
        "diffusion_gemma_26b_a4b_it",
    )
    model.generate({
        "prompts": "Describe this image: <|image|>",
        "images": image_array,  # np.ndarray of shape (H, W, 3)
    })
    ```
    """

    backbone_cls = DiffusionGemmaBackbone
    preprocessor_cls = DiffusionGemmaBlockDiffusionLMPreprocessor

    def __init__(
        self,
        preprocessor,
        backbone,
        canvas_length=256,
        max_denoising_steps=48,
        t_min=0.4,
        t_max=0.8,
        sampler="entropy_bound",
        stop_token_ids=None,
        pad_token_id=None,
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
        self.canvas_length = canvas_length
        self.max_denoising_steps = max_denoising_steps
        self.t_min = t_min
        self.t_max = t_max
        self.stop_token_ids = (
            tuple(stop_token_ids) if stop_token_ids is not None else None
        )
        if pad_token_id is None and preprocessor is not None:
            pad_token_id = preprocessor.tokenizer.pad_token_id
        self.pad_token_id = pad_token_id
        self.sampler = get_sampler(sampler)
        self.generate_function = None

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
        config.update(
            {
                "canvas_length": self.canvas_length,
                "max_denoising_steps": self.max_denoising_steps,
                "t_min": self.t_min,
                "t_max": self.t_max,
                "sampler": self._serialize_sampler(),
                "stop_token_ids": self.stop_token_ids,
                "pad_token_id": self.pad_token_id,
            }
        )
        return config

    def _serialize_sampler(self):
        from keras_hub.src.samplers.serialization import serialize

        return serialize(self.sampler)

    def _init_canvas(self, batch_size):
        """Create the initial random-token canvas `(B, canvas_length)`."""
        vocab_size = self.backbone.vocabulary_size
        return keras.random.randint(
            shape=(batch_size, self.canvas_length),
            minval=0,
            maxval=vocab_size,
            seed=getattr(self.sampler, "seed_generator", None),
            dtype="int32",
        )

    def _forward_step(
        self,
        canvas,
        encoder_cache,
        prompt_length,
        prev_logits,
        temperature,
        prompt_padding_mask=None,
    ):
        """Run a single denoising forward pass."""
        canvas_embeds = self._prepare_canvas_embeds(canvas, prev_logits)
        hidden = self._decode_canvas_step(
            canvas_embeds,
            encoder_cache,
            prompt_length,
            prompt_padding_mask=prompt_padding_mask,
        )
        logits = self._canvas_logits(hidden)
        return ops.cast(logits, "float32") / temperature

    def generate_step(
        self,
        inputs,
        max_length=None,
        stop_token_ids=None,
    ):
        """Generate one or more denoised canvases for a single batch.

        Args:
            inputs: dict. Pre-processed inputs containing at minimum
                `"token_ids"` and `"padding_mask"`.

        Returns:
            A `(B, max_length)` int tensor of final denoised tokens. If
            `max_length` is `None`, returns one canvas.
        """
        output_length = self.canvas_length if max_length is None else max_length
        num_canvases = (
            output_length + self.canvas_length - 1
        ) // self.canvas_length

        encoder_cache, prompt_length = self._encode_prompt(inputs)
        prompt_padding_mask = inputs.get("padding_mask", None)
        if stop_token_ids is not None and prompt_padding_mask is None:
            prompt_padding_mask = ops.ones_like(
                inputs["token_ids"], dtype="bool"
            )

        batch_size = ops.shape(inputs["token_ids"])[0]
        generated_canvases = []
        generated_masks = []
        finished_sequences = ops.zeros((batch_size,), dtype="bool")

        for canvas_index in range(num_canvases):
            canvas = self._init_canvas(batch_size)

            def next(canvas, prev_logits, step):
                step_float = ops.cast(step, "float32")
                temperature = self.t_max - (
                    (self.t_max - self.t_min)
                    * step_float
                    / self.max_denoising_steps
                )
                return self._forward_step(
                    canvas,
                    encoder_cache,
                    prompt_length,
                    prev_logits,
                    temperature,
                    prompt_padding_mask=prompt_padding_mask,
                )

            argmax_canvas = self.sampler(
                next=next,
                canvas=canvas,
                max_steps=self.max_denoising_steps,
                model=self,
            )
            argmax_canvas = ops.cast(argmax_canvas, "int32")

            if stop_token_ids is not None:
                stop_token_ids_tensor = ops.convert_to_tensor(
                    stop_token_ids, dtype="int32"
                )
                stop_locations = ops.any(
                    ops.equal(
                        ops.expand_dims(argmax_canvas, axis=-1),
                        stop_token_ids_tensor,
                    ),
                    axis=-1,
                )
                stop_locations = ops.logical_and(
                    stop_locations,
                    ops.logical_not(
                        ops.expand_dims(finished_sequences, axis=-1)
                    ),
                )
                stop_count = ops.cumsum(
                    ops.cast(stop_locations, "int32"), axis=-1
                )
                after_first_stop = ops.greater(
                    stop_count - ops.cast(stop_locations, "int32"), 0
                )
                canvas_padding_mask = ops.logical_not(after_first_stop)
                canvas_padding_mask = ops.logical_and(
                    canvas_padding_mask,
                    ops.logical_not(
                        ops.expand_dims(finished_sequences, axis=-1)
                    ),
                )
                if self.pad_token_id is not None:
                    argmax_canvas = ops.where(
                        canvas_padding_mask,
                        argmax_canvas,
                        ops.cast(self.pad_token_id, "int32"),
                    )
                finished_sequences = ops.logical_or(
                    finished_sequences,
                    ops.any(stop_locations, axis=-1),
                )
            else:
                canvas_padding_mask = ops.ones(
                    (batch_size, self.canvas_length), dtype="bool"
                )

            generated_canvases.append(argmax_canvas)
            generated_masks.append(canvas_padding_mask)

            if canvas_index < num_canvases - 1:
                if prompt_padding_mask is not None:
                    prompt_padding_mask = ops.concatenate(
                        [
                            ops.cast(prompt_padding_mask, "bool"),
                            canvas_padding_mask,
                        ],
                        axis=1,
                    )
                encoder_cache = self._encode_canvas_as_context(
                    argmax_canvas,
                    encoder_cache,
                    prompt_length,
                    padding_mask=prompt_padding_mask,
                )
                prompt_length += self.canvas_length

        generated = ops.concatenate(generated_canvases, axis=1)
        generated = generated[:, :output_length]
        if stop_token_ids is None:
            return generated
        padding_mask = ops.concatenate(generated_masks, axis=1)
        return {
            "token_ids": generated,
            "padding_mask": padding_mask[:, :output_length],
        }

    def _encode_prompt(self, inputs):
        token_ids = inputs["token_ids"]
        padding_mask = inputs.get("padding_mask", None)

        pixel_values = inputs.get("pixel_values", None)
        pixel_position_ids = inputs.get("pixel_position_ids", None)
        vision_indices = inputs.get("vision_indices", None)
        vision_mask = inputs.get("vision_mask", None)

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
        prompt_length = token_ids.shape[1]
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
            x, next_cache = layer(
                x,
                cache=cache[:, i, ...],
                cache_update_index=0,
                padding_mask=padding_mask,
                vision_mask=vision_mask,
                is_encoder=True,
            )
            caches.append(next_cache)

        encoder_kv_cache = ops.stack(caches, axis=1)
        return encoder_kv_cache, prompt_length

    def _encode_canvas_as_context(
        self,
        canvas_token_ids,
        encoder_kv_cache,
        context_length,
        padding_mask=None,
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
            padding_mask: Optional bool tensor covering the existing context
                and the appended canvas.

        Returns:
            Extended KV cache of shape
            `(B, num_layers, 2, context_length + canvas_length, ...)`.
        """
        x = self.backbone.token_embedding(canvas_token_ids)
        embed_scale = ops.cast(
            ops.sqrt(ops.cast(self.backbone.hidden_dim, "float32")), x.dtype
        )
        x = x * embed_scale

        # Extend the existing encoder KV cache to make room for canvas KVs.
        paddings = [
            [0, 0],
            [0, 0],
            [0, 0],
            [0, self.canvas_length],
            [0, 0],
            [0, 0],
        ]
        extended_cache = ops.pad(encoder_kv_cache, paddings)

        caches = []
        for i, layer in enumerate(self.backbone.transformer_layers):
            x, next_cache = layer(
                x,
                cache=extended_cache[:, i, ...],
                cache_update_index=context_length,
                padding_mask=padding_mask,
                is_encoder=True,
            )
            caches.append(next_cache)

        return ops.stack(caches, axis=1)

    def _prepare_canvas_embeds(self, canvas, prev_logits):
        x = self.backbone.token_embedding(canvas)
        embed_scale = ops.cast(
            ops.sqrt(ops.cast(self.backbone.hidden_dim, "float32")), x.dtype
        )
        x = x * embed_scale

        return self.backbone.diffusion_self_conditioning(x, prev_logits)

    def _decode_canvas_step(
        self,
        canvas_embeds,
        encoder_kv_cache,
        prompt_length,
        prompt_padding_mask=None,
    ):
        x = canvas_embeds
        batch_size = ops.shape(x)[0]
        canvas_length = x.shape[1]

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

        # Build a combined key-side padding mask so canvas queries do not
        # attend to padding positions in the encoder KV cache.
        if prompt_padding_mask is not None:
            canvas_real = ops.ones((batch_size, canvas_length), dtype="bool")
            combined_padding_mask = ops.concatenate(
                [
                    ops.cast(prompt_padding_mask, "bool"),
                    canvas_real,
                ],
                axis=1,
            )
        else:
            combined_padding_mask = None

        canvas_positions = ops.arange(
            prompt_length,
            prompt_length + canvas_length,
            dtype="int32",
        )
        canvas_positions = ops.broadcast_to(
            ops.expand_dims(canvas_positions, axis=0),
            (batch_size, canvas_length),
        )

        caches = []
        for i, layer in enumerate(self.backbone.transformer_layers):
            current_cache = combined_cache[:, i, ...]
            current_padding_mask = combined_padding_mask
            cache_update_index = prompt_length
            positions = None

            if (
                layer.use_sliding_window_attention
                and not layer.is_global_attention
            ):
                # HF exposes only the rolling encoder prefix to local decoder
                # layers, then appends the current canvas read-only.
                prefix_length = min(
                    prompt_length, layer.sliding_window_size - 1
                )
                cache_start = prompt_length - prefix_length
                local_cache_length = prefix_length + canvas_length
                cache_shape = ops.shape(current_cache)
                current_cache = ops.slice(
                    current_cache,
                    (0, 0, cache_start, 0, 0),
                    (
                        cache_shape[0],
                        cache_shape[1],
                        local_cache_length,
                        cache_shape[3],
                        cache_shape[4],
                    ),
                )
                if current_padding_mask is not None:
                    current_padding_mask = ops.slice(
                        current_padding_mask,
                        (0, cache_start),
                        (batch_size, local_cache_length),
                    )
                cache_update_index = prefix_length
                positions = canvas_positions

            x, next_cache = layer(
                x,
                cache=current_cache,
                cache_update_index=cache_update_index,
                canvas_mask=canvas_mask,
                padding_mask=current_padding_mask,
                positions=positions,
            )
            caches.append(next_cache)

        return self.backbone.layer_norm(x)

    def _canvas_logits(self, hidden):
        return self.backbone.token_embedding(hidden, reverse=True)
