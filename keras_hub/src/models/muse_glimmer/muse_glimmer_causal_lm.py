from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_causal_lm_preprocessor import (  # noqa: E501
    MuseGlimmerCausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


def _set_vision_encoder_padding_caps(backbone, preprocessor):
    """Set vision encoder padding caps from processor limits."""
    vision_encoder = backbone.vision_encoder
    if vision_encoder is None or preprocessor is None:
        return
    window_patches = max(
        vision_encoder.window_size // vision_encoder.patch_size, 1
    )
    image_converter = preprocessor.image_converter
    if image_converter is not None:
        # Derive the window cap from the image token budget.
        raw_side = image_converter.max_image_tokens * image_converter.merge_size
        vision_encoder.max_num_windows = -(-raw_side // window_patches)
    video_converter = preprocessor.video_converter
    if video_converter is not None:
        vision_encoder.max_num_frames = video_converter.num_frames
        # Derive the frame patch cap from the video token budget.
        vision_encoder.max_frame_size = (
            video_converter.max_video_frame_tokens
            * video_converter.merge_size**2
        )


@keras_hub_export("keras_hub.models.MuseGlimmerCausalLM")
class MuseGlimmerCausalLM(CausalLM):
    """An end-to-end MuseGlimmer model for causal language modeling.

    Predicts the next token given previous tokens (and, optionally,
    interleaved image/video features routed through the backbone's vision
    encoder). Applies MuseGlimmer's Gemma-style tanh logit softcap, with
    the extra `output_multiplier` pre-scale:
    `T * tanh((logits * output_multiplier) / T)`.

    Args:
        backbone: A `keras_hub.models.MuseGlimmerBackbone` instance.
        preprocessor: A `keras_hub.models.MuseGlimmerCausalLMPreprocessor`
            or `None`.
    """

    backbone_cls = MuseGlimmerBackbone
    preprocessor_cls = MuseGlimmerCausalLMPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor
        _set_vision_encoder_padding_caps(backbone, preprocessor)

        # === Functional Model ===
        inputs = backbone.input
        hidden_states = backbone(inputs)
        logits = backbone.token_embedding(hidden_states, reverse=True)
        logits = self._apply_logit_softcap(logits)
        super().__init__(inputs=inputs, outputs=logits, **kwargs)

    def _apply_logit_softcap(self, logits):
        logits = ops.cast(logits, "float32")
        logits = logits * self.backbone.output_multiplier
        cap = self.backbone.final_logit_softcapping
        logits = ops.tanh(logits / cap) * cap
        return logits

    def _encode_vision(self, pixel_values, image_grid_thw):
        backbone = self.backbone
        img_embeddings = backbone.vision_encoder(pixel_values, image_grid_thw)
        img_embeddings = backbone.projector_activation(
            backbone.vision_adapter_fc1(img_embeddings)
        )
        img_embeddings = backbone.projector_activation(
            backbone.vision_adapter_fc2(img_embeddings)
        )
        img_embeddings = backbone.vision_projection(img_embeddings)
        img_embeddings = backbone.perception_emb_norm(img_embeddings)
        return img_embeddings

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
        padding_mask=None,
        img_embeddings=None,
        vision_indices=None,
        target_layer_ids=None,
    ):
        """Forward pass with a KV cache, for autoregressive decoding.

        Args:
            target_layer_ids: optional list of int. When given, an extra
                4th return value is produced: the concatenation of this
                pass's hidden states at each of these layer indices (`0`
                is the embedding output, `i` for `i >= 1` is the output of
                decoder layer `i - 1`), matching the `target_layer_ids`
                indexing convention of the DFlash assistant/drafter's
                `context_hidden_states` input. Used only by
                `generate_step()`'s speculative-decoding path when an
                `assistant_model` is attached; `None` (the default) keeps
                the plain 3-tuple return for every other caller.
        """
        x = self.backbone.token_embedding(token_ids)
        x = self.backbone.embed_norm(x)

        if img_embeddings is not None and vision_indices is not None:
            x = self.backbone.interleave_embeddings(
                image_embeddings=img_embeddings,
                text_embeddings=x,
                vision_indices=vision_indices,
            )

        collected_layer_outputs = (
            {0: x} if target_layer_ids is not None else None
        )

        next_cache = []
        for i in range(self.backbone.num_layers):
            layer = self.backbone.transformer_layers[i]
            x, layer_cache = layer(
                x,
                decoder_padding_mask=padding_mask,
                self_attention_cache=cache[:, i, ...],
                self_attention_cache_update_index=cache_update_index,
            )
            next_cache.append(layer_cache)
            if collected_layer_outputs is not None:
                collected_layer_outputs[i + 1] = x
        next_cache = ops.stack(next_cache, axis=1)

        hidden_states = x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        logits = self._apply_logit_softcap(logits)

        if target_layer_ids is not None:
            target_hidden_states = ops.concatenate(
                [collected_layer_outputs[idx] for idx in target_layer_ids],
                axis=-1,
            )
            return logits, hidden_states, next_cache, target_hidden_states
        return logits, hidden_states, next_cache

    def _build_cache(
        self,
        token_ids,
        padding_mask,
        img_embeddings=None,
        vision_indices=None,
        target_layer_ids=None,
    ):
        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]
        num_layers = self.backbone.num_layers
        num_kv_heads = self.backbone.num_key_value_heads
        head_dim = self.backbone.head_dim
        shape = [
            batch_size,
            num_layers,
            2,
            max_length,
            num_kv_heads,
            head_dim,
        ]
        cache = ops.zeros(shape, dtype=self.compute_dtype)
        outputs = self.call_with_cache(
            token_ids,
            cache,
            0,
            padding_mask=padding_mask,
            img_embeddings=img_embeddings,
            vision_indices=vision_indices,
            target_layer_ids=target_layer_ids,
        )
        if target_layer_ids is not None:
            _, hidden_states, cache, target_hidden_states = outputs
            return hidden_states, cache, target_hidden_states
        _, hidden_states, cache = outputs
        return hidden_states, cache

    def generate_step(self, inputs, stop_token_ids=None):
        token_ids, padding_mask = inputs["token_ids"], inputs["padding_mask"]

        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)
        vision_indices = inputs.get("vision_indices", None)

        img_embeddings = None
        if (
            self.backbone.vision_encoder is not None
            and pixel_values is not None
        ):
            img_embeddings = self._encode_vision(pixel_values, image_grid_thw)

        _assistant = getattr(self, "_assistant_model", None)
        target_layer_ids = (
            _assistant.backbone.context_projection_layer_ids
            if _assistant is not None
            else None
        )

        cache_outputs = self._build_cache(
            token_ids,
            padding_mask,
            img_embeddings=img_embeddings,
            vision_indices=vision_indices,
            target_layer_ids=target_layer_ids,
        )
        if target_layer_ids is not None:
            hidden_states, cache, target_hidden_states = cache_outputs
        else:
            hidden_states, cache = cache_outputs
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            cache_update_index = index - 1
            batch_size = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, cache_update_index], [batch_size, 1])
            logits, hidden_states, cache = self.call_with_cache(
                prompt, cache, cache_update_index, padding_mask=None
            )
            return (
                ops.squeeze(logits, axis=1),
                ops.squeeze(hidden_states, axis=1),
                cache,
            )

        draft_next = None
        draft_cache = None
        verify_next = None

        # DFlash block-diffusion speculative decoding: build draft_next and
        # verify_next when an assistant model is attached via generate().
        # Unlike a single-token autoregressive drafter, the DFlash
        # assistant produces its whole candidate block from ONE forward
        # pass. `SpeculativeSampler` still calls `draft_next` once per
        # candidate position, so `draft_next` computes that one-shot
        # block on its first call each cycle and caches the result (a
        # plain Python-level memo — safe because `SpeculativeSampler`'s
        # per-cycle draft loop is an ordinary unrolled Python `for` loop
        # traced once per compiled generate graph, not a symbolic loop),
        # returning a different position's slice on each subsequent call
        # instead of recomputing the block.
        #
        # `draft_cache`'s 4th element is the assistant's own persistent
        # context-cache (see `MuseGlimmerTextAttention`'s docstring): one
        # real target position's key/value cached per cycle, growing
        # across the whole generation, so later cycles reuse earlier
        # context K/V instead of recomputing it from scratch every time
        # (the actual DFlash caching benefit) — only the noise block is
        # ever recomputed fresh.
        if _assistant is not None:
            num_candidates = _assistant.block_size - 1
            mask_token_id = _assistant.mask_token_id
            hidden_dim = self.backbone.hidden_dim
            assistant_backbone = _assistant.backbone

            batch_size_ = ops.shape(token_ids)[0]
            max_length_ = ops.shape(token_ids)[1]
            anchor_pos = ops.cast(index - 1, "int32")
            context_dim = len(target_layer_ids) * hidden_dim
            init_context = ops.slice(
                target_hidden_states,
                [0, anchor_pos, 0],
                [batch_size_, 1, context_dim],
            )
            # Persistent context-cache for the assistant's own attention
            # layers (see `MuseGlimmerTextAttention`'s docstring): holds
            # one real target position's key/value per drafting cycle,
            # growing across the whole generation, so later cycles reuse
            # earlier context K/V instead of recomputing it. The noise
            # block itself is never cached (always fresh, discarded each
            # cycle) — see `MuseGlimmerAssistantCausalLM.call_with_cache`.
            assistant_cache = ops.zeros(
                [
                    batch_size_,
                    assistant_backbone.num_layers,
                    2,
                    max_length_,
                    assistant_backbone.num_key_value_heads,
                    assistant_backbone.head_dim,
                ],
                dtype=self.compute_dtype,
            )
            draft_cache = (init_context, cache, anchor_pos, assistant_cache)

            _block_logits_memo = {}

            def draft_next(prompt, draft_state, call_index):
                context_seed, _, cycle_anchor_pos, assistant_cache_state = (
                    draft_state
                )
                if "logits" not in _block_logits_memo:
                    batch = ops.shape(prompt)[0]
                    last_token_id = ops.slice(
                        prompt, [0, cycle_anchor_pos], [batch, 1]
                    )
                    noise_ids = ops.concatenate(
                        [
                            last_token_id,
                            ops.full(
                                (batch, num_candidates),
                                mask_token_id,
                                dtype=prompt.dtype,
                            ),
                        ],
                        axis=1,
                    )
                    noise_embeds = self.backbone.token_embedding(noise_ids)
                    block_padding_mask = ops.ones(
                        (batch, num_candidates + 1), dtype="int32"
                    )
                    block_hidden, updated_assistant_cache = (
                        _assistant.call_with_cache(
                            noise_embeds=noise_embeds,
                            context_hidden_states=context_seed,
                            cache=assistant_cache_state,
                            cache_update_index=cycle_anchor_pos,
                            padding_mask=block_padding_mask,
                        )
                    )
                    _block_logits_memo["assistant_cache"] = (
                        updated_assistant_cache
                    )
                    candidate_hidden = block_hidden[:, 1:, :]
                    candidate_logits = self.backbone.token_embedding(
                        candidate_hidden, reverse=True
                    )
                    candidate_logits = self._apply_logit_softcap(
                        candidate_logits
                    )
                    _block_logits_memo["logits"] = candidate_logits
                candidate_logits = _block_logits_memo["logits"]
                position = ops.clip(
                    ops.cast(call_index - cycle_anchor_pos - 1, "int32"),
                    0,
                    num_candidates - 1,
                )
                logits_i = ops.take(candidate_logits, position, axis=1)
                dummy_hidden = ops.zeros((ops.shape(prompt)[0], 1))
                new_draft_state = (
                    context_seed,
                    draft_state[1],
                    cycle_anchor_pos,
                    _block_logits_memo["assistant_cache"],
                )
                return logits_i, dummy_hidden, new_draft_state

            def verify_next(prompt, target_cache, call_index, k):
                batch = ops.shape(prompt)[0]
                max_len = ops.shape(prompt)[1]
                safe_start = ops.maximum(
                    ops.cast(0, "int32"),
                    ops.minimum(
                        ops.cast(call_index - 1, "int32"),
                        ops.cast(max_len - k - 1, "int32"),
                    ),
                )
                prompt_slice = ops.slice(
                    prompt, [0, safe_start], [batch, k + 1]
                )
                logits, _, updated_cache, context_hidden = self.call_with_cache(
                    prompt_slice,
                    target_cache,
                    safe_start,
                    padding_mask=None,
                    target_layer_ids=target_layer_ids,
                )
                start_offset = ops.cast(call_index - 1, "int32") - safe_start
                indices = ops.arange(k + 1, dtype="int32")
                indices = ops.minimum(
                    indices + start_offset, ops.cast(k, "int32")
                )
                logits = ops.take(logits, indices, axis=1)
                context_hidden = ops.take(context_hidden, indices, axis=1)
                return logits, context_hidden, updated_cache

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
        return {"token_ids": token_ids, "padding_mask": padding_mask}

    def generate(
        self,
        inputs,
        max_length=None,
        stop_token_ids="auto",
        strip_prompt=False,
        assistant_model=None,
    ):
        """Generate text, optionally accelerated by a DFlash assistant.

        Args:
            assistant_model: optional
                `keras_hub.models.MuseGlimmerAssistantCausalLM`. When set,
                `generate()` performs DFlash speculative decoding: the
                assistant drafts a whole candidate block per cycle and
                this (target) model verifies it in one parallel forward
                pass via `keras_hub.samplers.SpeculativeSampler`. The
                assistant's own `block_size - 1` becomes the sampler's
                `num_speculative_tokens`. Defaults to `None` (plain
                autoregressive decoding with the compiled `sampler`).
        """
        if assistant_model is not None:
            from keras_hub.src.samplers.speculative_sampler import (
                SpeculativeSampler,
            )

            original_sampler = self.sampler
            original_generate_function = self.generate_function
            self.sampler = SpeculativeSampler(
                num_speculative_tokens=assistant_model.block_size - 1,
                temperature=getattr(original_sampler, "temperature", 1.0),
            )
            self.generate_function = None  # Force recompile.
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
                self._assistant_model = None
                self.sampler = original_sampler
                self.generate_function = original_generate_function

        return outputs
