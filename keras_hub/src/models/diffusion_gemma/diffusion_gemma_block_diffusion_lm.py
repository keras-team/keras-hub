import keras
import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.block_diffusion_lm import BlockDiffusionLM
from keras_hub.src.models.block_diffusion_lm import get_diffusion_sampler
from keras_hub.src.models.diffusion_gemma.diffusion_gemma_backbone import (
    DiffusionGemmaBackbone,
)
from keras_hub.src.models.diffusion_gemma.diffusion_gemma_block_diffusion_lm_preprocessor import (  # noqa: E501
    DiffusionGemmaBlockDiffusionLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import tf


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

    def generate(
        self,
        inputs,
        max_length=None,
        stop_token_ids="auto",
        sequence_length=None,
        t_min=None,
        t_max=None,
        pad_token_id="auto",
    ):
        """Generate a denoised canvas given prompt inputs.

        Args:
            inputs: python data, tensor data, or a `tf.data.Dataset`. If a
                `preprocessor` is attached to the model, `inputs` should
                match the structure expected by the `preprocessor` layer. If
                a `preprocessor` is not attached, `inputs` should match the
                structure expected by the `backbone` model.
            max_length: Optional int. Maximum length of the generated
                sequence. Defaults to the model's configured canvas length.
            stop_token_ids: Optional. `None`, `"auto"`, or tuple of token
                IDs. Defaults to `"auto"`, which uses stop IDs configured on
                the model, or the preprocessor tokenizer's end token plus
                `<turn|>` (DiffusionGemma's end-of-turn token). `None`
                generates until `max_length`.
            sequence_length: Optional int. Overrides the preprocessor's
                prompt packing length. Raises `ValueError` if the prompt does
                not fit. Defaults to `None`.
            t_min: Optional float. Sampling temperature minimum. Defaults to
                `self.t_min`.
            t_max: Optional float. Sampling temperature maximum. Defaults to
                `self.t_max`.
            pad_token_id: Optional int. Overrides the model's configured
                padding token. `None` disables padding. Defaults to
                `"auto"`, which uses `self.pad_token_id`.

        Returns:
            Decoded string(s) or integer token arrays, depending on whether
            a `preprocessor` is attached.
        """
        resolved_stop_token_ids = stop_token_ids
        if stop_token_ids == "auto" and self.preprocessor is not None:
            if getattr(self, "stop_token_ids", None) is None:
                resolved_stop_token_ids = (
                    self.preprocessor.tokenizer.end_token_id,
                    self.preprocessor.tokenizer.token_to_id("<turn|>"),
                )
            else:
                resolved_stop_token_ids = self.stop_token_ids
        # Keep the preprocessor's stop-token list in sync with this call.
        if self.preprocessor is not None and resolved_stop_token_ids != "auto":
            self.preprocessor.stop_token_ids = (
                tuple(resolved_stop_token_ids)
                if resolved_stop_token_ids is not None
                else None
            )

        # Pass sampling arguments as tensors so JAX and TF keep them dynamic.
        # Use -1 as the sentinel for no padding token.
        resolved_t_min = self.t_min if t_min is None else t_min
        resolved_t_max = self.t_max if t_max is None else t_max
        resolved_pad_token_id = (
            self.pad_token_id if pad_token_id == "auto" else pad_token_id
        )
        sentinel_pad_token_id = (
            -1 if resolved_pad_token_id is None else resolved_pad_token_id
        )
        return super().generate(
            inputs,
            max_length=max_length,
            stop_token_ids=resolved_stop_token_ids,
            sequence_length=sequence_length,
            t_min=ops.convert_to_tensor(resolved_t_min, dtype="float32"),
            t_max=ops.convert_to_tensor(resolved_t_max, dtype="float32"),
            pad_token_id=ops.convert_to_tensor(
                sentinel_pad_token_id, dtype="int32"
            ),
        )

    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            "DiffusionGemmaBlockDiffusionLM only supports inference for "
            "now. Training the model isn't supported yet."
        )

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
        self.sampler = get_diffusion_sampler(sampler)
        self.generate_function = None

    # Reset compiled generation when shape-affecting settings change.
    @property
    def canvas_length(self):
        return self._canvas_length

    @canvas_length.setter
    def canvas_length(self, value):
        self._canvas_length = value
        self.generate_function = None

    @property
    def max_denoising_steps(self):
        return self._max_denoising_steps

    @max_denoising_steps.setter
    def max_denoising_steps(self, value):
        self._max_denoising_steps = value
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
            seed=self.sampler.seed_generator,
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
        skip_auto_pad=False,
    ):
        """Run a single denoising forward pass."""
        canvas_embeds = self._prepare_canvas_embeds(canvas, prev_logits)
        hidden = self._decode_canvas_step(
            canvas_embeds,
            encoder_cache,
            prompt_length,
            prompt_padding_mask=prompt_padding_mask,
            skip_auto_pad=skip_auto_pad,
        )
        logits = self._canvas_logits(hidden)
        return ops.cast(logits, "float32") / temperature

    def generate_step(
        self,
        inputs,
        max_length=None,
        stop_token_ids=None,
        t_min=None,
        t_max=None,
        pad_token_id=None,
    ):
        """Generate one or more denoised canvases for a single batch.

        Denoises canvases with `ops.while_loop` until all rows finish.

        Args:
            inputs: dict. Pre-processed inputs containing at minimum
                `"token_ids"` and `"padding_mask"`.
            t_min: Optional float. Minimum sampling temperature.
            t_max: Optional float. Maximum sampling temperature.
            pad_token_id: Optional int. Padding token ID. Use `-1` to disable
                padding.

        Returns:
            A dict with `"token_ids"` and `"padding_mask"`, each shaped
            `(B, max_length)`. If `max_length` is `None`, returns one
            canvas.
        """
        output_length = self.canvas_length if max_length is None else max_length
        num_canvases = (
            output_length + self.canvas_length - 1
        ) // self.canvas_length

        encoder_cache, prompt_length = self._encode_prompt(inputs)
        prompt_padding_mask = inputs.get("padding_mask", None)
        if prompt_padding_mask is None:
            prompt_padding_mask = ops.ones_like(
                inputs["token_ids"], dtype="bool"
            )
        prompt_padding_mask = ops.cast(prompt_padding_mask, "bool")
        batch_size = ops.shape(inputs["token_ids"])[0]

        stop_token_ids_tensor = ops.convert_to_tensor(
            stop_token_ids if stop_token_ids else (), dtype="int32"
        )

        if t_min is None:
            t_min = self.t_min
        if t_max is None:
            t_max = self.t_max
        if pad_token_id is None:
            pad_token_id = (
                self.pad_token_id if self.pad_token_id is not None else -1
            )
        t_min = ops.convert_to_tensor(t_min, dtype="float32")
        t_max = ops.convert_to_tensor(t_max, dtype="float32")
        pad_token_id = ops.convert_to_tensor(pad_token_id, dtype="int32")
        has_pad_token = ops.not_equal(pad_token_id, -1)

        # Add space for the largest local attention window.
        max_sliding_prefix = 0
        for layer in self.backbone.transformer_layers:
            if (
                layer.use_sliding_window_attention
                and not layer.is_global_attention
            ):
                max_sliding_prefix = max(
                    max_sliding_prefix, layer.sliding_window_size - 1
                )
        buffer_length = (
            prompt_length
            + num_canvases * self.canvas_length
            + max_sliding_prefix
        )

        encoder_cache_buffer = self._pad_encoder_cache(
            encoder_cache, 0, buffer_length
        )
        padding_mask_buffer = ops.zeros(
            (batch_size, buffer_length), dtype="bool"
        )
        padding_mask_buffer = ops.slice_update(
            padding_mask_buffer, (0, 0), prompt_padding_mask
        )
        pad_value = ops.where(
            has_pad_token, pad_token_id, ops.zeros_like(pad_token_id)
        )
        output_canvases_buffer = (
            ops.ones(
                (batch_size, num_canvases * self.canvas_length),
                dtype="int32",
            )
            * pad_value
        )
        output_masks_buffer = ops.zeros(
            (batch_size, num_canvases * self.canvas_length), dtype="bool"
        )
        finished_sequences = ops.zeros((batch_size,), dtype="bool")
        canvas_index = ops.convert_to_tensor(0, dtype="int32")
        context_length = ops.convert_to_tensor(prompt_length, dtype="int32")

        def cond(
            canvas_index,
            context_length,
            encoder_cache_buffer,
            padding_mask_buffer,
            finished_sequences,
            output_canvases_buffer,
            output_masks_buffer,
        ):
            return ops.logical_and(
                canvas_index < num_canvases,
                ops.logical_not(ops.all(finished_sequences)),
            )

        def body(
            canvas_index,
            context_length,
            encoder_cache_buffer,
            padding_mask_buffer,
            finished_sequences,
            output_canvases_buffer,
            output_masks_buffer,
        ):
            canvas = self._init_canvas(batch_size)

            def next(canvas, prev_logits, step):
                step_float = ops.cast(step, "float32")
                temperature = t_max - (
                    (t_max - t_min) * step_float / self.max_denoising_steps
                )
                return self._forward_step(
                    canvas,
                    encoder_cache_buffer,
                    context_length,
                    prev_logits,
                    temperature,
                    prompt_padding_mask=padding_mask_buffer,
                    skip_auto_pad=True,
                )

            argmax_canvas = self.sampler(
                next=next,
                canvas=canvas,
                max_steps=self.max_denoising_steps,
                model=self,
            )
            argmax_canvas = ops.cast(argmax_canvas, "int32")

            stop_locations = ops.any(
                ops.equal(
                    ops.expand_dims(argmax_canvas, axis=-1),
                    stop_token_ids_tensor,
                ),
                axis=-1,
            )
            stop_locations = ops.logical_and(
                stop_locations,
                ops.logical_not(ops.expand_dims(finished_sequences, axis=-1)),
            )
            stop_count = ops.cumsum(ops.cast(stop_locations, "int32"), axis=-1)
            after_first_stop = ops.greater(
                stop_count - ops.cast(stop_locations, "int32"), 0
            )
            canvas_padding_mask = ops.logical_not(after_first_stop)
            canvas_padding_mask = ops.logical_and(
                canvas_padding_mask,
                ops.logical_not(ops.expand_dims(finished_sequences, axis=-1)),
            )
            # Use a tensor operation because `has_pad_token` can be traced.
            fill_mask = ops.logical_or(
                canvas_padding_mask, ops.logical_not(has_pad_token)
            )
            argmax_canvas = ops.where(fill_mask, argmax_canvas, pad_token_id)
            new_finished_sequences = ops.logical_or(
                finished_sequences, ops.any(stop_locations, axis=-1)
            )

            output_offset = canvas_index * self.canvas_length
            output_canvases_buffer = ops.slice_update(
                output_canvases_buffer,
                (0, output_offset),
                argmax_canvas,
            )
            output_masks_buffer = ops.slice_update(
                output_masks_buffer,
                (0, output_offset),
                canvas_padding_mask,
            )
            padding_mask_buffer = ops.slice_update(
                padding_mask_buffer,
                (0, context_length),
                canvas_padding_mask,
            )
            encoder_cache_buffer = self._encode_canvas_as_context(
                argmax_canvas,
                encoder_cache_buffer,
                context_length,
                padding_mask=padding_mask_buffer,
            )

            return (
                canvas_index + 1,
                context_length + self.canvas_length,
                encoder_cache_buffer,
                padding_mask_buffer,
                new_finished_sequences,
                output_canvases_buffer,
                output_masks_buffer,
            )

        loop_vars = (
            canvas_index,
            context_length,
            encoder_cache_buffer,
            padding_mask_buffer,
            finished_sequences,
            output_canvases_buffer,
            output_masks_buffer,
        )
        (
            _,
            _,
            _,
            _,
            _,
            output_canvases_buffer,
            output_masks_buffer,
        ) = self.sampler.run_loop(
            cond=cond,
            body=body,
            loop_vars=loop_vars,
            maximum_iterations=num_canvases,
            model=self,
        )

        return {
            "token_ids": output_canvases_buffer[:, :output_length],
            "padding_mask": output_masks_buffer[:, :output_length],
        }

    def _encode_prompt(self, inputs):
        token_ids = inputs["token_ids"]
        padding_mask = inputs.get("padding_mask", None)

        pixel_values = inputs.get("pixel_values", None)
        pixel_position_ids = inputs.get("pixel_position_ids", None)
        vision_indices = inputs.get("vision_indices", None)
        vision_mask = inputs.get("vision_mask", None)

        # Add a batch dimension for unbatched image inputs.
        if pixel_values is not None and len(ops.shape(pixel_values)) == 3:
            pixel_values = ops.expand_dims(pixel_values, axis=0)
        if (
            pixel_position_ids is not None
            and len(ops.shape(pixel_position_ids)) == 3
        ):
            pixel_position_ids = ops.expand_dims(pixel_position_ids, axis=0)

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

        # Global scale applied after interleaving: text positions get
        # sqrt(hidden_dim), vision positions keep their pre-scaled magnitude.
        x = x * embed_scale

        batch_size = ops.shape(token_ids)[0]
        # static: preprocessor pads to fixed sequence_length
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
                is_encoder=True,
                vision_mask=vision_mask,
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
        """Extend the encoder KV cache with new canvas tokens.

        The method writes new keys and values into the existing cache.
        Vision embeddings are consumed once in `_encode_prompt`.

        Args:
            canvas_token_ids: int tensor of shape `(B, canvas_length)`.
            encoder_kv_cache: float tensor with a pre-sized sequence axis.
            context_length: int scalar. Index for the new canvas keys and
                values.
            padding_mask: Optional bool tensor covering the cache sequence.

        Returns:
            KV cache with new canvas keys and values.
        """
        x = self.backbone.token_embedding(canvas_token_ids)
        embed_scale = ops.cast(
            ops.sqrt(ops.cast(self.backbone.hidden_dim, "float32")), x.dtype
        )
        x = x * embed_scale

        caches = []
        for i, layer in enumerate(self.backbone.transformer_layers):
            x, next_cache = layer(
                x,
                cache=encoder_kv_cache[:, i, ...],
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

    def _pad_encoder_cache(
        self, encoder_kv_cache, prompt_length, canvas_length
    ):
        """Pad the encoder KV cache to the requested sequence length."""
        # All ints: cache dim 3 is static prompt_length from _encode_prompt,
        # canvas_length is a fixed config attribute, so the comparison is safe.
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
            return ops.pad(encoder_kv_cache, paddings)
        return encoder_kv_cache

    def _decode_canvas_step(
        self,
        canvas_embeds,
        encoder_kv_cache,
        prompt_length,
        prompt_padding_mask=None,
        skip_auto_pad=False,
    ):
        x = canvas_embeds
        batch_size = ops.shape(x)[0]
        canvas_length = x.shape[1]

        if skip_auto_pad:
            # The caller pre-sizes the cache before the while loop.
            combined_cache = encoder_kv_cache
        else:
            combined_cache = self._pad_encoder_cache(
                encoder_kv_cache, prompt_length, canvas_length
            )

        # canvas_mask marks every canvas position as bidirectional.
        canvas_mask = ops.ones((batch_size, canvas_length), dtype="bool")

        # Build a combined key-side padding mask so canvas queries do not
        # attend to padding positions in the encoder KV cache.
        if skip_auto_pad:
            # Mark the current canvas positions in the fixed buffer.
            buffer_length = ops.shape(encoder_kv_cache)[3]
            position_index = ops.arange(buffer_length, dtype="int32")
            is_this_canvas = ops.logical_and(
                position_index >= prompt_length,
                position_index < prompt_length + canvas_length,
            )
            is_this_canvas = ops.broadcast_to(
                ops.expand_dims(is_this_canvas, axis=0),
                (batch_size, buffer_length),
            )
            combined_padding_mask = ops.logical_or(
                ops.cast(prompt_padding_mask, "bool"), is_this_canvas
            )
        elif prompt_padding_mask is not None:
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

        # Local layers slice their own windows inside the transformer layer.
        for i, layer in enumerate(self.backbone.transformer_layers):
            x, _ = layer(
                x,
                cache=combined_cache[:, i, ...],
                cache_update_index=prompt_length,
                canvas_mask=canvas_mask,
                padding_mask=combined_padding_mask,
                return_cache=False,
            )

        return self.backbone.layer_norm(x)

    def _canvas_logits(self, hidden):
        return self.backbone.token_embedding(hidden, reverse=True)
