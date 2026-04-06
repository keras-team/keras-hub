import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.models.gemma4.gemma4_causal_lm_preprocessor import (
    Gemma4CausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.Gemma4CausalLM")
class Gemma4CausalLM(CausalLM):
    """An end-to-end multimodal Gemma4 model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on
    images and plain text inputs, or to autoregressively generate plain text
    similar to the data used for training. Note that the model is
    image-text in, text out.

    This model has a `generate()` method, which generates text based on a
    prompt. The generation strategy used is controlled by an additional
    `sampler` argument on `compile()`. You can recompile the model with
    different `keras_hub.samplers` objects to control the generation. By
    default, `"greedy"` sampling will be used.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to string inputs during
    `fit()`, `predict()`, `evaluate()` and `generate()`. This is done by default
    when creating the model with `from_preset()`.

    Args:
        preprocessor: A `keras_hub.models.Gemma4CausalLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.
        backbone: A `keras_hub.models.Gemma4Backbone` instance.
    """

    backbone_cls = Gemma4Backbone
    preprocessor_cls = Gemma4CausalLMPreprocessor

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
        # This must be "backbone.input" i.e. the full input structure,
        # rather than "backbone.inputs" which is the flattened list of inputs.
        inputs = backbone.input
        hidden_state = backbone(inputs=inputs)
        outputs = backbone.token_embedding(hidden_state, reverse=True)

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

    def _normalize_generate_inputs(
        self,
        inputs,
    ):
        """Overrides the superclass' method to handle unbatched image inputs."""
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

            # If prompt is scalar, images can be either a 3-D NumPy array/
            # Tensor, or a list of 3-D arrays. Uprank images accordingly.
            if input_is_scalar and "images" in inputs:
                x = inputs["images"]
                if isinstance(x, np.ndarray) and len(x.shape) == 3:
                    inputs["images"] = [x]
                elif tf and isinstance(x, tf.Tensor) and x.shape.rank == 3:
                    inputs["images"] = x[tf.newaxis]
                elif isinstance(x, list):
                    inputs["images"] = [x]

            # If prompt is scalar, a raw 1-D audio waveform should be
            # wrapped in a list so _preprocess_audio sees a batch of 1.
            if input_is_scalar and "audio" in inputs:
                x = inputs["audio"]
                if isinstance(x, np.ndarray) and len(x.shape) == 1:
                    inputs["audio"] = [x]
                elif tf and isinstance(x, tf.Tensor) and x.shape.rank == 1:
                    inputs["audio"] = x[tf.newaxis]
                elif isinstance(x, list) and not isinstance(x[0], list):
                    inputs["audio"] = [x]

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
        img_embeddings=None,
        vision_mask=None,
        padding_mask=None,
        vision_indices=None,
        audio_embeddings=None,
        audio_indices=None,
        audio_mask=None,
        cache_update_mask=None,
    ):
        """Forward pass of `Gemma4CausalLM` with cache.

        `call_with_cache` adds an additional forward pass for the model for
        autoregressive inference. Unlike calling the model directly, this method
        allows caching previous key/value Tensors in multi-head attention layer,
        and avoids recomputing the outputs of seen tokens.

        Args:
            token_ids: a dense int Tensor with shape `(batch_size, max_length)`.
            cache: a dense float Tensor, the cache of key and value.
            cache_update_index: int, or int Tensor. The index of current inputs
                in the whole sequence.
            img_embeddings: a dense float Tensor with shape
                `(batch_size, num_images, image_sequence_length, hidden_dim)`.
            vision_mask: a dense bool Tensor with shape
                `(batch_size, max_length)`. True at positions occupied by
                vision tokens.
            padding_mask: a dense int Tensor with shape
                `(batch_size, max_length)`.
            vision_indices: a dense int Tensor with shape
                `(batch_size, num_images * num_vision_tokens_per_image)`.
                Positions in the text sequence where image embeddings are
                inserted.
            audio_embeddings: a dense float Tensor with shape
                `(batch_size, num_clips, audio_sequence_length, hidden_dim)`.
            audio_indices: a dense int Tensor with shape
                `(batch_size, num_clips * num_audio_tokens_per_clip)`.
                Positions in the text sequence where audio embeddings are
                inserted.
            cache_update_mask: a dense bool Tensor with shape
                `(batch_size, 1)`. Controls which cache positions are updated
                during decoding (used to skip positions already filled by the
                prompt).

        Returns:
            A (logits, hidden_states, cache) tuple. Where `logits` is the
            language model logits for the input token_ids, `hidden_states` is
            the final hidden representation of the input tokens, and `cache` is
            the decoding cache.
        """

        text_embeddings = self.backbone.token_embedding(token_ids)

        # Interleave image embeddings. Pre-scale by 1/sqrt(hidden_dim) so that
        # after the global x *= sqrt(hidden_dim) below, vision positions remain
        # at their natural (unscaled) embed_vision magnitude.
        if img_embeddings is not None:
            scaled_img_embeddings = img_embeddings * ops.cast(
                float(self.backbone.hidden_dim) ** -0.5, img_embeddings.dtype
            )
            x = self.backbone.interleave_embeddings(
                image_embeddings=scaled_img_embeddings,
                text_embeddings=text_embeddings,
                vision_indices=vision_indices,
            )
        else:
            x = text_embeddings

        # Interleave audio embeddings (same pre-scaling as vision).
        if audio_embeddings is not None:
            scaled_audio_embeddings = audio_embeddings * ops.cast(
                float(self.backbone.hidden_dim) ** -0.5, audio_embeddings.dtype
            )
            x = self.backbone.audio_interleave_embeddings(
                image_embeddings=scaled_audio_embeddings,
                text_embeddings=x,
                vision_indices=audio_indices,
            )

        # Per-layer token embeddings. Vision positions use pad_token_id (0),
        # mirroring HF's llm_input_ids masking before embed_tokens_per_layer.
        _hpl = self.backbone.hidden_size_per_layer_input
        if _hpl > 0:
            _per_layer_ids = token_ids
            if vision_mask is not None or audio_mask is not None:
                mask_to_zero = ops.zeros_like(_per_layer_ids, dtype="bool")
                if vision_mask is not None:
                    mask_to_zero = ops.logical_or(
                        mask_to_zero, ops.cast(vision_mask, "bool")
                    )
                if audio_mask is not None:
                    mask_to_zero = ops.logical_or(
                        mask_to_zero, ops.cast(audio_mask, "bool")
                    )
                _per_layer_ids = ops.where(
                    mask_to_zero,
                    ops.zeros_like(_per_layer_ids),
                    _per_layer_ids,
                )
            _per_emb = self.backbone.per_layer_token_embedding(_per_layer_ids)
            _per_emb = ops.cast(_per_emb, text_embeddings.dtype)
            _per_emb = _per_emb * ops.cast(float(_hpl) ** 0.5, _per_emb.dtype)
            per_layer_emb_flat = _per_emb
        else:
            per_layer_emb_flat = None

        # Global scale: text positions → token_embedding * sqrt(hidden_dim);
        # vision/audio positions remain at their pre-scaled embed magnitude.
        x = x * ops.cast(ops.sqrt(self.backbone.hidden_dim), x.dtype)

        # Per-layer model projection, computed after the global scale.
        if _hpl > 0:
            _per_proj = self.backbone.per_layer_model_projection(x)
            _per_proj = _per_proj * ops.cast(
                float(self.backbone.hidden_dim) ** -0.5, _per_proj.dtype
            )
            per_layer_proj_flat = _per_proj
        else:
            per_layer_proj_flat = None

        caches = []
        for i, transformer_layer in enumerate(self.backbone.transformer_layers):
            current_cache = cache[:, i, ...]
            # KV-shared layers (E4B) borrow K/V from an earlier layer's cache.
            shared_kv = None
            if (
                transformer_layer.is_kv_shared_layer
                and transformer_layer.kv_shared_layer_index is not None
            ):
                idx = transformer_layer.kv_shared_layer_index
                if idx < len(caches):
                    shared_kv = caches[idx]
                else:
                    shared_kv = cache[:, idx, ...]
            if per_layer_proj_flat is not None:
                proj_i = per_layer_proj_flat[:, :, i * _hpl : (i + 1) * _hpl]
                emb_i = per_layer_emb_flat[:, :, i * _hpl : (i + 1) * _hpl]
                proj_i_normed = self.backbone.per_layer_projection_norm(proj_i)
                per_layer_input_i = (proj_i_normed + emb_i) * ops.cast(
                    2.0**-0.5, proj_i.dtype
                )
            else:
                per_layer_input_i = None
            x, next_cache = transformer_layer(
                x,
                cache=current_cache,
                cache_update_index=cache_update_index,
                padding_mask=padding_mask,
                vision_mask=vision_mask,
                cache_update_mask=cache_update_mask,
                shared_kv=shared_kv,
                per_layer_input=per_layer_input_i,
            )
            caches.append(next_cache)
        cache = ops.stack(caches, axis=1)
        hidden_states = x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, hidden_states, cache

    def _build_cache(
        self,
        token_ids,
        img_embeddings,
        vision_mask,
        padding_mask,
        vision_indices,
        audio_embeddings=None,
        audio_indices=None,
        audio_mask=None,
    ):
        """Build an empty cache for use with `call_with_cache()`."""
        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_key_value_heads
        head_dim = self.backbone.head_dim
        # Global attention layers may use a larger per-head dimension than
        # local layers (global_head_dim > head_dim).  Allocate the cache
        # with max_head_dim so every layer's cache slot has the same shape
        # and ops.stack() works across all layers.
        global_head_dim = self.backbone.global_head_dim
        max_head_dim = (
            max(head_dim, global_head_dim) if global_head_dim else head_dim
        )
        shape = [batch_size, num_layers, 2, max_length, num_heads, max_head_dim]
        cache = ops.zeros(shape, dtype=self.compute_dtype)
        # Seed the cache.
        logits, hidden_states, cache = self.call_with_cache(
            token_ids=token_ids,
            img_embeddings=img_embeddings,
            vision_mask=vision_mask,
            cache=cache,
            cache_update_index=0,
            padding_mask=padding_mask,
            vision_indices=vision_indices,
            audio_embeddings=audio_embeddings,
            audio_indices=audio_indices,
            audio_mask=audio_mask,
            cache_update_mask=None,
        )
        return hidden_states, cache

    def generate_step(self, inputs, stop_token_ids=[106]):
        """A compilable generation function for a single batch of inputs.

        This function represents the inner, XLA-compilable, generation function
        for a single batch of inputs. Inputs should have the same structure as
        model inputs, a dictionary with keys `"token_ids"` and `"padding_mask"`.
        For multimodal models, inputs can also contain `"pixel_values"`,
        `"pixel_position_ids"`, `"vision_mask"`, `"vision_indices"`,
        `"audio_mel"`, `"audio_mel_mask"`, and `"audio_indices"`.

        Args:
            inputs: A dictionary with keys `"token_ids"` and
                `"padding_mask"` and optional multimodal keys, along with
                their batched tensor values.
            stop_token_ids: Tuple of ids of end tokens to stop on. If all
                sequences have produced a new stop token, generation
                will stop.
        """

        (
            token_ids,
            padding_mask,
            pixel_values,
            pixel_position_ids,
            vision_mask,
            vision_indices,
        ) = (
            inputs["token_ids"],
            inputs["padding_mask"],
            inputs.get("pixel_values", None),
            inputs.get("pixel_position_ids", None),
            inputs.get("vision_mask", None),
            inputs.get("vision_indices", None),
        )
        audio_mel = inputs.get("audio_mel", None)
        audio_mel_mask = inputs.get("audio_mel_mask", None)
        audio_indices = inputs.get("audio_indices", None)
        audio_mask = inputs.get("audio_mask", None)

        # Determine if we have actual images to process.
        # After preprocessing, pixel_values shape is
        # (batch, num_images, n**2, dim).
        # For text-only input, num_images=0 (static shape).
        # We use static shape check which returns a Python int, not a tensor.
        num_images = 0
        if (
            pixel_values is not None
            and hasattr(pixel_values, "shape")
            and len(pixel_values.shape) > 1
        ):
            num_images = pixel_values.shape[
                1
            ]  # Static shape; Python int or None.

        if not self.backbone.text_only_model and num_images:
            # Handle an unbatched image.
            if len(ops.shape(pixel_values)) == 3:
                pixel_values = ops.expand_dims(pixel_values, axis=0)
            if len(ops.shape(pixel_position_ids)) == 3:
                pixel_position_ids = ops.expand_dims(pixel_position_ids, axis=0)
            if len(ops.shape(vision_mask)) == 1:
                vision_mask = ops.expand_dims(vision_mask, axis=0)
            if len(ops.shape(vision_indices)) == 1:
                vision_indices = ops.expand_dims(vision_indices, axis=0)
            img_embeddings = self.backbone.vision_encoder(
                {
                    "pixel_values": pixel_values,
                    "pixel_position_ids": pixel_position_ids,
                }
            )
        else:
            img_embeddings = None
            vision_mask = None
            vision_indices = None

        # Determine if we have actual audio clips to process.
        # After preprocessing, audio_mel shape is (batch, num_clips, T, feat)
        # when batched, or (num_clips, T, feat) when unbatched (batch dim was
        # squeezed out in generate_preprocess). For no-audio input, num_clips=0.
        # We must key on axis 0 for the unbatched (3-D) case because axis 1
        # is the mel-time axis, which is dynamic and would evaluate to None,
        # silently disabling audio.
        num_clips = 0
        if audio_mel is not None and hasattr(audio_mel, "shape"):
            if len(audio_mel.shape) == 4:
                num_clips = audio_mel.shape[1]  # (B, num_clips, T, feat)
            elif len(audio_mel.shape) == 3:
                num_clips = audio_mel.shape[0]  # (num_clips, T, feat)

        if self.backbone.audio_encoder is not None and num_clips:
            # Handle an unbatched audio tensor.
            if len(ops.shape(audio_mel)) == 3:
                audio_mel = ops.expand_dims(audio_mel, axis=0)
            if len(ops.shape(audio_mel_mask)) == 2:
                audio_mel_mask = ops.expand_dims(audio_mel_mask, axis=0)
            if len(ops.shape(audio_indices)) == 1:
                audio_indices = ops.expand_dims(audio_indices, axis=0)

            audio_mel_mask = ops.cast(audio_mel_mask, "bool")
            audio_embeddings = self.backbone.audio_encoder(
                audio_mel, audio_mel_mask
            )
        else:
            audio_embeddings = None
            audio_indices = None

        # Create and seed cache with a single forward pass.
        hidden_states, cache = self._build_cache(
            token_ids,
            img_embeddings,
            vision_mask,
            padding_mask,
            vision_indices,
            audio_embeddings=audio_embeddings,
            audio_indices=audio_indices,
            audio_mask=audio_mask,
        )

        # Compute the lengths of all user-inputted token ids.
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        # Start at the first index that has no user-inputted id.
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            # The cache index is the index of our previous token.
            cache_update_index = index - 1
            batch_size = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, index - 1], [batch_size, 1])
            sliced_cache_update_mask = ops.slice(
                ~padding_mask, [0, index - 1], [batch_size, 1]
            )
            logits, hidden_states, cache = self.call_with_cache(
                token_ids=prompt,
                cache=cache,
                cache_update_index=cache_update_index,
                cache_update_mask=sliced_cache_update_mask,
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

        # Compute an updated padding mask.
        if stop_token_ids is not None:
            end_locations = any_equal(
                token_ids, stop_token_ids, ops.logical_not(padding_mask)
            )
            end_locations = ops.cast(end_locations, "int32")
            # Use cumsum to get ones in all locations after end_locations.
            cumsum = ops.cast(ops.cumsum(end_locations, axis=-1), "int32")
            overflow = cumsum - end_locations
            padding_mask = ops.logical_not(ops.cast(overflow, "bool"))
        else:
            padding_mask = ops.ones_like(token_ids, dtype="bool")
        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "pixel_values": pixel_values,
            "pixel_position_ids": pixel_position_ids,
            "audio_mel": audio_mel,
            "audio_mel_mask": audio_mel_mask,
            "audio_indices": audio_indices,
        }

    def generate(
        self,
        inputs,
        max_length=None,
        stop_token_ids="auto",
        strip_prompt=False,
    ):
        # If `auto`, add `<turn|>` as a stop token too.
        if self.preprocessor is None and stop_token_ids == "auto":
            raise ValueError(
                "A `preprocessor` must be attached to the model if "
                '`stop_token_ids="auto"`. Currently `preprocessor=None`. To '
                "call `generate()` with preprocessing detached, either pass "
                "`stop_token_ids=None` to always generate until `max_length` "
                "or pass a tuple of token ids that should terminate generation "
                "as `stop_token_ids`."
            )
        elif stop_token_ids == "auto":
            stop_token_ids = [
                self.preprocessor.tokenizer.end_token_id,
                self.preprocessor.tokenizer.token_to_id("<turn|>"),
            ]

        return super().generate(
            inputs,
            max_length=max_length,
            stop_token_ids=stop_token_ids,
            strip_prompt=strip_prompt,
        )
