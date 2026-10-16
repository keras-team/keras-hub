import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.gemma4_unified.gemma4_unified_backbone import (
    Gemma4UnifiedBackbone,
)
from keras_hub.src.models.gemma4_unified.gemma4_unified_causal_lm_preprocessor import (  # noqa: E501
    Gemma4UnifiedCausalLMPreprocessor,
)
from keras_hub.src.samplers.greedy_sampler import GreedySampler
from keras_hub.src.utils.tensor_utils import any_equal

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.Gemma4UnifiedCausalLM")
class Gemma4UnifiedCausalLM(CausalLM):
    """Multimodal Gemma4 Unified model for causal language
    modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This unified model uses an encoder-free architecture for
    multimodal inputs (text, images, audio, and video) and produces text
    output.

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
        preprocessor: A `keras_hub.models.Gemma4UnifiedCausalLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.
        backbone: A `keras_hub.models.Gemma4UnifiedBackbone` instance.

    Examples:

    Text generation from a text prompt.
    ```python
    gemma4_lm = keras_hub.models.Gemma4UnifiedCausalLM.from_preset(
        "gemma4_unified_instruct_12b",
    )
    gemma4_lm.generate("What is the capital of France?")
    ```
    """

    backbone_cls = Gemma4UnifiedBackbone
    preprocessor_cls = Gemma4UnifiedCausalLMPreprocessor

    def __init__(
        self,
        preprocessor,
        backbone,
        final_logit_cap=None,
        **kwargs,
    ):
        # === Layers ===
        self.preprocessor = preprocessor
        self.backbone = backbone
        self.final_logit_cap = final_logit_cap

        # === Functional Model ===
        inputs = backbone.input
        hidden_state = backbone(inputs=inputs)
        outputs = backbone.token_embedding(hidden_state, reverse=True)

        if final_logit_cap is not None:
            outputs = outputs / final_logit_cap
            outputs = ops.tanh(outputs)
            outputs = outputs * final_logit_cap

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
        """Forward pass of `Gemma4UnifiedCausalLM` with cache.

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

        # Pre-scale vision/audio by 1/sqrt(hidden_dim) to preserve magnitude.
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

        # Per-layer token embeddings; vision/audio positions use pad_token_id
        # (0) to zero out their per-layer contribution.
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

        # Global scale: multiply all embeddings by sqrt(hidden_dim).
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
        if self.final_logit_cap is not None:
            logits = logits / self.final_logit_cap
            logits = ops.tanh(logits)
            logits = logits * self.final_logit_cap
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
        # Global layers may use larger head_dim; allocate cache with max.
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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "final_logit_cap": self.final_logit_cap,
            }
        )
        return config

    def generate_step(self, inputs, stop_token_ids=None):
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

        # pixel_values: (B, num_images, n**2, dim); 0 for text-only.
        num_images = 0
        if (
            pixel_values is not None
            and hasattr(pixel_values, "shape")
            and len(pixel_values.shape) > 1
        ):
            num_images = pixel_values.shape[1]  # static shape; Python int

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

        # audio_mel shape: (B, clips, T, feat) or (clips, T, feat) unbatched.
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
            audio_mask = None

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

        draft_next = None
        draft_cache = None
        verify_next = None

        # Speculative decoding: build draft_next/verify_next for assistant.
        _assistant = getattr(self, "_assistant_model", None)
        if _assistant is not None:
            # Find the last non-KV-shared layer of each attention type.
            _target_kv_src_full_idx = None
            _target_kv_src_local_idx = None
            _target_layer_types = self.backbone.layer_types or []
            for _i, _tl in enumerate(self.backbone.transformer_layers):
                if not getattr(_tl, "is_kv_shared_layer", False):
                    _lt = (
                        _target_layer_types[_i]
                        if _i < len(_target_layer_types)
                        else None
                    )
                    if _lt == "full_attention":
                        _target_kv_src_full_idx = _i
                    else:
                        _target_kv_src_local_idx = _i

            # Draft model borrows target's KV cache; draft_cache =
            # (last_hidden, target_cache, fixed_pos).
            batch_size_ = ops.shape(token_ids)[0]
            start_pos = ops.cast(index - 1, "int32")
            init_last_hidden = ops.slice(
                hidden_states,
                [0, start_pos, 0],
                [batch_size_, 1, self.backbone.hidden_dim],
            )

            def draft_next(prompt, draft_state, index):
                """One draft step: produce a candidate token from the assistant.

                `draft_state` is a tuple
                `(last_hidden, cur_target_cache, fixed_pos)`:
                  - `last_hidden`: target model hidden state from the
                    previous step.
                  - `cur_target_cache`: the target model's KV cache.
                  - `fixed_pos`: RoPE position shared across all k draft steps.
                """
                last_hidden, cur_target_cache, fixed_pos = draft_state
                batch = ops.shape(prompt)[0]
                # Extract the last token id at position `index - 1`.
                last_token_id = ops.slice(
                    prompt, [0, ops.cast(index - 1, "int32")], [batch, 1]
                )
                last_token_embedding = self.backbone.token_embedding(
                    last_token_id
                )
                # Scale to match Gemma4TextScaledWordEmbedding.
                _embed_scale = ops.cast(
                    ops.sqrt(ops.cast(self.backbone.hidden_dim, "float32")),
                    last_token_embedding.dtype,
                )
                last_token_embedding = last_token_embedding * _embed_scale
                logits, next_hidden = _assistant.call_with_cache(
                    last_token_embedding=last_token_embedding,
                    last_hidden_state=last_hidden,
                    target_cache=cur_target_cache,
                    cache_update_index=fixed_pos,
                    target_kv_source_full_idx=_target_kv_src_full_idx,
                    target_kv_source_local_idx=_target_kv_src_local_idx,
                )
                # Apply final logit soft-cap to keep draft/verify on same scale.
                if self.final_logit_cap is not None:
                    cap = ops.cast(self.final_logit_cap, logits.dtype)
                    logits = ops.tanh(logits / cap) * cap
                return (
                    ops.squeeze(logits, axis=1),
                    next_hidden,
                    (next_hidden, cur_target_cache, fixed_pos),
                )

            def verify_next(prompt, target_cache, index, k):
                """Verify K+1 positions with the target model in parallel.

                ``safe_start`` is clamped so that the k+1-token window never
                overflows the buffer, preventing out-of-bounds in the
                sliding-window mask and key-cache update slices when generation
                nears the end of the pre-allocated sequence.
                """
                batch = ops.shape(prompt)[0]
                max_len = ops.shape(prompt)[1]
                safe_start = ops.maximum(
                    ops.cast(0, "int32"),
                    ops.minimum(
                        ops.cast(index - 1, "int32"),
                        ops.cast(max_len - k - 1, "int32"),
                    ),
                )
                prompt_slice = ops.slice(
                    prompt, [0, safe_start], [batch, k + 1]
                )
                # cache_update_mask: True=write; False=keep existing.
                cache_update_slice = ops.slice(
                    ~padding_mask, [0, safe_start], [batch, k + 1]
                )
                vision_mask_slice = (
                    ops.slice(vision_mask, [0, safe_start], [batch, k + 1])
                    if vision_mask is not None
                    else None
                )
                audio_mask_slice = (
                    ops.slice(audio_mask, [0, safe_start], [batch, k + 1])
                    if audio_mask is not None
                    else None
                )
                logits, hidden_states, updated_cache = self.call_with_cache(
                    token_ids=prompt_slice,
                    cache=target_cache,
                    cache_update_index=safe_start,
                    img_embeddings=None,
                    vision_mask=vision_mask_slice,
                    padding_mask=None,
                    vision_indices=None,
                    audio_embeddings=None,
                    audio_indices=None,
                    audio_mask=audio_mask_slice,
                    cache_update_mask=cache_update_slice,
                )
                # Align position 0 of logits and hidden_states with index - 1.
                start_offset = ops.cast(index - 1, "int32") - safe_start
                indices = ops.arange(k + 1, dtype="int32")
                indices = ops.minimum(
                    indices + start_offset, ops.cast(k, "int32")
                )
                logits = ops.take(logits, indices, axis=1)
                hidden_states = ops.take(hidden_states, indices, axis=1)
                return logits, hidden_states, updated_cache

            # fixed_pos is constant across all k draft steps in a cycle.
            initial_fixed_pos = ops.cast(index - 1, "int32")
            draft_cache = (init_last_hidden, cache, initial_fixed_pos)

        token_ids = self.sampler(
            next=next,
            prompt=token_ids,
            cache=cache,
            index=index,
            mask=padding_mask,
            stop_token_ids=stop_token_ids,
            hidden_states=hidden_states,
            model=self,
            draft_next=draft_next,
            draft_cache=draft_cache,
            verify_next=verify_next,
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
        }

    def generate(
        self,
        inputs,
        max_length=None,
        stop_token_ids="auto",
        strip_prompt=False,
        assistant_model=None,
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

        if assistant_model is not None:
            from keras_hub.src.samplers.speculative_sampler import (
                SpeculativeSampler,
            )

            # Save current (sampler, compiled graph) to restore after.
            original_sampler = self.sampler
            original_generate_function = self.generate_function

            num_spec = getattr(assistant_model, "num_speculative_tokens", 5)

            # Use assistant model's sampler as base_sampler for
            # stochastic rejection sampling.
            spec_base_sampler = getattr(assistant_model, "sampler", None)
            if isinstance(self.sampler, GreedySampler):
                spec_base_sampler = None

            # Reuse cached speculative graph if params match.
            cached_spec_sampler = getattr(self, "_cached_spec_sampler", None)
            cached_spec_fn = getattr(self, "_cached_spec_generate_fn", None)

            if (
                cached_spec_sampler is not None
                and cached_spec_sampler.num_speculative_tokens == num_spec
                and cached_spec_sampler.base_sampler is spec_base_sampler
            ):
                # Reuse compiled speculative graph.
                self.sampler = cached_spec_sampler
                self.generate_function = cached_spec_fn
            else:
                # Compile a new speculative graph.
                self.sampler = SpeculativeSampler(
                    num_speculative_tokens=num_spec,
                    base_sampler=spec_base_sampler,
                    temperature=getattr(original_sampler, "temperature", 1.0),
                )
                self.generate_function = None  # force recompile

            self._assistant_model = assistant_model

        try:
            outputs = super().generate(
                inputs,
                max_length=max_length,
                stop_token_ids=stop_token_ids,
                strip_prompt=strip_prompt,
            )
        finally:
            if assistant_model is not None:
                self._cached_spec_sampler = self.sampler
                self._cached_spec_generate_fn = self.generate_function
                # Restore original sampler/graph without forcing recompile.
                self._assistant_model = None
                self.sampler = original_sampler
                self.generate_function = original_generate_function

        return outputs
