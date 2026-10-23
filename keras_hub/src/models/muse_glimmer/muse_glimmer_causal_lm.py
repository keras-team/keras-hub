import weakref

import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_causal_lm_preprocessor import (  # noqa: E501
    MuseGlimmerCausalLMPreprocessor,
)
from keras_hub.src.samplers.greedy_sampler import GreedySampler
from keras_hub.src.samplers.random_sampler import RandomSampler
from keras_hub.src.samplers.speculative_sampler import SpeculativeSampler
from keras_hub.src.samplers.top_k_sampler import TopKSampler
from keras_hub.src.samplers.top_p_sampler import TopPSampler
from keras_hub.src.utils.tensor_utils import any_equal
from keras_hub.src.utils.tensor_utils import tf


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

    Example:
    ```python
    muse_glimmer_lm = keras_hub.models.MuseGlimmerCausalLM.from_preset(
        "muse_glimmer_30b"
    )

    # Generate text.
    muse_glimmer_lm.generate("What is Keras?", max_length=64)

    # Generate with a different sampler.
    muse_glimmer_lm.compile(sampler="top_k")
    muse_glimmer_lm.generate("What is Keras?", max_length=64)

    # Speculative decoding with the DFlash assistant. The output matches
    # plain greedy decoding when the sampler is greedy.
    assistant = keras_hub.models.MuseGlimmerAssistantCausalLM.from_preset(
        "muse_glimmer_30b_assistant"
    )
    muse_glimmer_lm.compile(sampler="greedy")
    muse_glimmer_lm.generate(
        "What is Keras?", max_length=64, assistant_model=assistant
    )
    ```
    """

    backbone_cls = MuseGlimmerBackbone
    preprocessor_cls = MuseGlimmerCausalLMPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

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

    def _embed_tokens(self, token_ids):
        """Embed token ids as plain text."""
        embeddings = self.backbone.token_embedding(token_ids)
        return self.backbone.embed_norm(embeddings)

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
        padding_mask=None,
        img_embeddings=None,
        vision_indices=None,
        target_layer_ids=None,
        inputs_embeds=None,
    ):
        """Forward pass with a KV cache, for autoregressive decoding.

        Args:
            inputs_embeds: optional float tensor. When given, it replaces
                the embedding of `token_ids`, and the pass skips the vision
                interleave. `generate_step()` uses it to keep the image
                embeddings of prompt positions.
            target_layer_ids: optional list of int. When given, an extra
                4th return value is produced: the concatenation of this
                pass's hidden states at each of these layer indices. Index
                `j` is the output of decoder layer `j`, and the last layer
                gives the final-norm output. The indexing matches the
                Hugging Face `output_hidden_states=target_layer_ids` call
                that feeds the DFlash assistant's
                `context_hidden_states` input. Used only by
                `generate_step()`'s speculative-decoding path when an
                `assistant_model` is attached; `None` (the default) keeps
                the plain 3-tuple return for every other caller.
        """
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self._embed_tokens(token_ids)
            if img_embeddings is not None and vision_indices is not None:
                x = self.backbone.interleave_embeddings(
                    image_embeddings=img_embeddings,
                    text_embeddings=x,
                    vision_indices=vision_indices,
                )

        collected_layer_outputs = {} if target_layer_ids is not None else None

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
                collected_layer_outputs[i] = x
        next_cache = ops.stack(next_cache, axis=1)

        hidden_states = x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        logits = self._apply_logit_softcap(logits)

        if target_layer_ids is not None:
            # Same indexing as Hugging Face `output_hidden_states=[ids]`:
            # id `j` is the output of decoder layer `j`, and the last
            # layer's entry is the final-norm output.
            collected_layer_outputs[self.backbone.num_layers - 1] = (
                hidden_states
            )
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

    def _write_assistant_context(
        self, assistant_model, context_hidden_states, assistant_cache, start
    ):
        """Write target context into the assistant cache at `start`.

        The call runs the drafter on a one-token dummy block to keep the
        cost low. The block output is discarded.
        """
        batch_size = ops.shape(context_hidden_states)[0]
        dummy_block = ops.zeros(
            (batch_size, 1, self.backbone.hidden_dim), dtype=self.compute_dtype
        )
        _, assistant_cache = assistant_model.call_with_cache(
            noise_embeds=dummy_block,
            context_hidden_states=context_hidden_states,
            cache=assistant_cache,
            cache_update_index=start,
            padding_mask=ops.ones((batch_size, 1), dtype="int32"),
        )
        return assistant_cache

    def _draft_block(
        self, assistant_model, prompt, anchor, last_context, assistant_cache
    ):
        """Draft the candidate logits for the positions after `anchor`.

        The assistant cache must already hold the target context for every
        position below `anchor - 1`. `last_context` is the target context
        at `anchor - 1`. The call rewrites that slot, so the valid context
        is `[0, anchor)` and the block starts at `anchor`. This matches
        the HF DFlash layout.
        """
        batch_size = ops.shape(prompt)[0]
        num_candidates = assistant_model.block_size - 1
        anchor_token = ops.slice(prompt, [0, anchor], [batch_size, 1])
        noise_ids = ops.concatenate(
            [
                anchor_token,
                ops.full(
                    (batch_size, num_candidates),
                    assistant_model.mask_token_id,
                    dtype=prompt.dtype,
                ),
            ],
            axis=1,
        )
        noise_embeds = self.backbone.token_embedding(noise_ids)
        # A prompt of length one has no context. Slot 0 then holds the
        # anchor context, and the block starts one position late.
        context_position = ops.maximum(anchor - 1, 0)
        block_hidden, assistant_cache = assistant_model.call_with_cache(
            noise_embeds=noise_embeds,
            context_hidden_states=last_context,
            cache=assistant_cache,
            cache_update_index=context_position,
            padding_mask=ops.ones(
                (batch_size, num_candidates + 1), dtype="int32"
            ),
        )
        logits = self.backbone.token_embedding(
            block_hidden[:, 1:, :], reverse=True
        )
        return self._apply_logit_softcap(logits), assistant_cache

    def generate_step(self, inputs, stop_token_ids=None):
        """Run one compiled generation pass.

        `generate()` attaches the assistant as `_assistant_model` for the
        duration of one call. When it is set, the pass uses DFlash
        speculative decoding.

        Args:
            inputs: dict. The preprocessed generation inputs.
            stop_token_ids: optional tuple of int. Token ids that end
                generation.
        """
        token_ids, padding_mask = inputs["token_ids"], inputs["padding_mask"]
        assistant_model = getattr(self, "_assistant_model", None)

        pixel_values = inputs.get("pixel_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)
        vision_indices = inputs.get("vision_indices", None)

        img_embeddings = None
        if (
            self.backbone.vision_encoder is not None
            and pixel_values is not None
        ):
            img_embeddings = self._encode_vision(pixel_values, image_grid_thw)

        target_layer_ids = (
            assistant_model.backbone.context_projection_layer_ids
            if assistant_model is not None
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

        # Embed the prompt once with the vision features. Every pass over a
        # prompt position reuses these embeddings. A plain text embedding
        # of an image position would rewrite its cached key and value.
        prompt_embeds = None
        if img_embeddings is not None and vision_indices is not None:
            prompt_embeds = self.backbone.interleave_embeddings(
                image_embeddings=img_embeddings,
                text_embeddings=self._embed_tokens(token_ids),
                vision_indices=vision_indices,
            )

        def embed_window(tokens, start):
            """Return the embeddings of a window of `tokens` at `start`.

            Prompt positions keep the prefill embeddings. Other positions
            use the text embedding of the generated token.
            """
            if prompt_embeds is None:
                return None
            batch_size, length = ops.shape(tokens)[0], ops.shape(tokens)[1]
            prompt_batch_size = ops.shape(prompt_embeds)[0]
            is_prompt = ops.slice(
                ops.cast(padding_mask, "bool"),
                [0, start],
                [prompt_batch_size, length],
            )
            window = ops.slice(
                prompt_embeds,
                [0, start, 0],
                [prompt_batch_size, length, self.backbone.hidden_dim],
            )
            # Beam and contrastive search repeat every row. Map each
            # repeated row back to its prompt row.
            source_row = ops.arange(batch_size) // (
                batch_size // prompt_batch_size
            )
            is_prompt = ops.take(is_prompt, source_row, axis=0)
            window = ops.take(window, source_row, axis=0)
            return ops.where(
                is_prompt[:, :, None], window, self._embed_tokens(tokens)
            )

        def next(prompt, cache, index):
            cache_update_index = index - 1
            batch_size = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, cache_update_index], [batch_size, 1])
            logits, hidden_states, cache = self.call_with_cache(
                prompt,
                cache,
                cache_update_index,
                padding_mask=None,
                inputs_embeds=embed_window(prompt, cache_update_index),
            )
            return (
                ops.squeeze(logits, axis=1),
                ops.squeeze(hidden_states, axis=1),
                cache,
            )

        draft_next = None
        draft_cache = None
        verify_next = None

        # DFlash block-diffusion speculative decoding. The assistant drafts
        # the whole candidate block in one forward pass. `SpeculativeSampler`
        # calls `draft_next` once per candidate, so the first call of each
        # cycle computes the block. Later calls return one slice of it.
        #
        # `draft_cache` is `(packed_window, target_cache, anchor, state)`.
        # `packed_window` holds the target context of the last verify
        # window, flattened to one row. `state` is
        # `(assistant_cache, window_start)`. The assistant cache persists
        # the context key/value across cycles. The noise block is never
        # cached.
        if assistant_model is not None:
            num_candidates = assistant_model.block_size - 1
            window_length = num_candidates + 1
            assistant_backbone = assistant_model.backbone
            batch_size = ops.shape(token_ids)[0]
            max_length = ops.shape(token_ids)[1]
            # The shape is a Python int unless the sequence length is
            # dynamic. Each verify window needs `window_length` positions.
            if isinstance(max_length, int) and max_length < window_length:
                raise ValueError(
                    f"`max_length` must be at least {window_length}, the "
                    f"`block_size` of the assistant. Received "
                    f"`max_length={max_length}`. Increase `max_length` or "
                    "call `generate()` without `assistant_model`."
                )
            context_dim = len(target_layer_ids) * self.backbone.hidden_dim

            def get_window_start(anchor):
                # First position of the verify window for `anchor`.
                return ops.clip(anchor, 0, max_length - window_length)

            def pack_window(window):
                return ops.reshape(
                    window, (batch_size, 1, window_length * context_dim)
                )

            # Seed the full prompt context before the first cycle, as HF
            # DFlash does. Slots at or after the anchor hold unverified
            # context. Later cycles overwrite them before they become valid.
            assistant_cache = ops.zeros(
                [
                    batch_size,
                    assistant_backbone.num_layers,
                    2,
                    max_length,
                    assistant_backbone.num_key_value_heads,
                    assistant_backbone.head_dim,
                ],
                dtype=self.compute_dtype,
            )
            assistant_cache = self._write_assistant_context(
                assistant_model, target_hidden_states, assistant_cache, 0
            )
            anchor = ops.cast(index - 1, "int32")
            window_start = ops.clip(anchor - 1, 0, max_length - window_length)
            window = ops.slice(
                target_hidden_states,
                [0, window_start, 0],
                [batch_size, window_length, context_dim],
            )
            draft_cache = (
                pack_window(window),
                cache,
                anchor,
                (assistant_cache, window_start),
            )

            # `SpeculativeSampler` calls `draft_next` exactly
            # `num_candidates` times per cycle. Recompute the block on the
            # first call of each cycle, so eager backends (torch) do not
            # reuse a stale block from an earlier cycle.
            _block_logits_memo = {"num_calls": 0}

            def draft_next(prompt, draft_state, call_index):
                packed_window, target_cache, anchor, assistant_state = (
                    draft_state
                )
                is_first_call = _block_logits_memo["num_calls"] == 0
                _block_logits_memo["num_calls"] = (
                    _block_logits_memo["num_calls"] + 1
                ) % num_candidates
                if is_first_call:
                    assistant_cache, window_start = assistant_state
                    window = ops.reshape(
                        packed_window, (batch_size, window_length, context_dim)
                    )
                    # Write every position of the last verify window. This
                    # covers the old anchor and all accepted candidates.
                    # Rejected positions lie at or after the new anchor.
                    assistant_cache = self._write_assistant_context(
                        assistant_model, window, assistant_cache, window_start
                    )
                    offset = ops.clip(
                        ops.maximum(anchor - 1, 0) - window_start,
                        0,
                        window_length - 1,
                    )
                    last_context = ops.slice(
                        window, [0, offset, 0], [batch_size, 1, context_dim]
                    )
                    logits, assistant_cache = self._draft_block(
                        assistant_model,
                        prompt,
                        anchor,
                        last_context,
                        assistant_cache,
                    )
                    _block_logits_memo["logits"] = logits
                    _block_logits_memo["state"] = (
                        assistant_cache,
                        get_window_start(anchor),
                    )
                position = ops.clip(
                    ops.cast(call_index - anchor - 1, "int32"),
                    0,
                    num_candidates - 1,
                )
                logits_i = ops.take(_block_logits_memo["logits"], position, 1)
                dummy_hidden = ops.zeros((batch_size, 1))
                new_draft_state = (
                    packed_window,
                    target_cache,
                    anchor,
                    _block_logits_memo["state"],
                )
                return logits_i, dummy_hidden, new_draft_state

            def verify_next(prompt, target_cache, call_index, k):
                start = get_window_start(ops.cast(call_index - 1, "int32"))
                prompt_slice = ops.slice(
                    prompt, [0, start], [batch_size, k + 1]
                )
                logits, _, updated_cache, context_hidden = self.call_with_cache(
                    prompt_slice,
                    target_cache,
                    start,
                    padding_mask=None,
                    target_layer_ids=target_layer_ids,
                    inputs_embeds=embed_window(prompt_slice, start),
                )
                start_offset = ops.cast(call_index - 1, "int32") - start
                indices = ops.arange(k + 1, dtype="int32")
                indices = ops.minimum(
                    indices + start_offset, ops.cast(k, "int32")
                )
                logits = ops.take(logits, indices, axis=1)
                # The sampler passes one row of these hidden states to the
                # next cycle. Every row holds the whole verify window, so
                # the drafter can write all accepted positions.
                packed_window = ops.broadcast_to(
                    pack_window(context_hidden),
                    (batch_size, k + 1, window_length * context_dim),
                )
                return logits, packed_window, updated_cache

        # Only `SpeculativeSampler` accepts the draft arguments.
        speculative_kwargs = {}
        if assistant_model is not None:
            speculative_kwargs = {
                "draft_next": draft_next,
                "draft_cache": draft_cache,
                "verify_next": verify_next,
            }
        token_ids = self.sampler(
            next=next,
            prompt=token_ids,
            cache=cache,
            index=index,
            mask=padding_mask,
            stop_token_ids=stop_token_ids,
            hidden_states=hidden_states,
            model=self,
            **speculative_kwargs,
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

    def _validate_assistant(self, assistant_model):
        """Raise a `ValueError` for an assistant this model cannot use."""
        block_size = assistant_model.block_size
        if block_size < 2:
            raise ValueError(
                "The assistant `block_size` must be at least 2. Received "
                f"`block_size={block_size}`."
            )
        mask_token_id = assistant_model.mask_token_id
        vocabulary_size = self.backbone.vocabulary_size
        if not isinstance(mask_token_id, int) or not (
            0 <= mask_token_id < vocabulary_size
        ):
            raise ValueError(
                "The assistant `mask_token_id` must be an integer in "
                f"`[0, {vocabulary_size})`. Received "
                f"`mask_token_id={mask_token_id}`."
            )
        layer_ids = assistant_model.backbone.context_projection_layer_ids
        if layer_ids is None:
            raise ValueError(
                "The assistant backbone must set "
                "`context_projection_layer_ids`. Received "
                "`context_projection_layer_ids=None`."
            )
        num_layers = self.backbone.num_layers
        if any(not 0 <= layer_id < num_layers for layer_id in layer_ids):
            raise ValueError(
                "Each assistant `context_projection_layer_ids` entry must be "
                f"in `[0, {num_layers})`. Received "
                f"`context_projection_layer_ids={list(layer_ids)}`."
            )
        assistant_hidden_dim = assistant_model.backbone.hidden_dim
        if assistant_hidden_dim != self.backbone.hidden_dim:
            raise ValueError(
                "The assistant `hidden_dim` must equal the `hidden_dim` of "
                f"the target model ({self.backbone.hidden_dim}). Received "
                f"`hidden_dim={assistant_hidden_dim}`."
            )

    def get_quantization_layer_structure(self, mode):
        if mode not in ["gptq", "awq"]:
            return None

        # The first block reads the normed embedding, not the raw embedding.
        backbone = self.backbone
        token_ids = keras.Input(shape=(None,), dtype="int32")
        x = backbone.embed_norm(backbone.token_embedding(token_ids))
        pre_processor = keras.Model(inputs=token_ids, outputs=x)
        return {
            "pre_block_layers": [pre_processor],
            "sequential_blocks": backbone.transformer_layers,
        }

    def _post_quantize(self, mode, **kwargs):
        super()._post_quantize(mode, **kwargs)
        # The cached assisted function holds the old target variables.
        self._assisted_generate_cache = None

    def _normalize_generate_inputs(self, inputs):
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
            inputs = dict(inputs)
            inputs["prompts"], input_is_scalar = normalize(inputs["prompts"])
            if "responses" in inputs:
                inputs["responses"], _ = normalize(inputs["responses"])
        else:
            inputs, input_is_scalar = normalize(inputs)

        return [inputs], input_is_scalar

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
                `num_speculative_tokens`. A greedy compiled `sampler`
                gives greedy acceptance. Any other compiled `sampler`
                becomes the `base_sampler` for rejection sampling. The
                model tracks the assistant only during the call. It keeps
                a weak reference to the assistant and the compiled
                function for the next call. Defaults to `None` (plain
                autoregressive decoding with the compiled `sampler`).
        """
        if assistant_model is None:
            return super().generate(
                inputs,
                max_length=max_length,
                stop_token_ids=stop_token_ids,
                strip_prompt=strip_prompt,
            )

        self._validate_assistant(assistant_model)
        original_sampler = self.sampler
        original_generate_function = self.generate_function

        # A greedy target keeps greedy acceptance. Any other sampler is the
        # `base_sampler` of the speculative sampler. The key holds every
        # input that the compiled function depends on. The cached sampler
        # keeps `base_sampler` alive, so its id stays unique.
        base_sampler = original_sampler
        if isinstance(base_sampler, GreedySampler):
            base_sampler = None
        elif not isinstance(
            base_sampler, (RandomSampler, TopKSampler, TopPSampler)
        ):
            raise ValueError(
                "`assistant_model` supports the greedy, random, top-k and "
                f"top-p samplers. Received: sampler={original_sampler}."
            )
        assistant_variables = (
            assistant_model.trainable_variables
            + assistant_model.non_trainable_variables
        )
        key = (
            assistant_model.block_size,
            id(base_sampler),
            self.run_eagerly,
            getattr(self, "jit_compile", None),
            tuple(id(variable) for variable in assistant_variables),
        )
        cached = getattr(self, "_assisted_generate_cache", None)
        assistant_was_tracked = self._tracker.is_in_store(
            "layers", assistant_model
        )
        try:
            if (
                cached is not None
                and cached[0] == key
                and (cached[1]() is assistant_model)
            ):
                self.sampler = cached[2]
                self.generate_function = cached[3]
            else:
                self.sampler = SpeculativeSampler(
                    num_speculative_tokens=assistant_model.block_size - 1,
                    base_sampler=base_sampler,
                    temperature=getattr(original_sampler, "temperature", 1.0),
                )
                self.generate_function = None  # Compile a new function.

            # Tracking the assistant puts its variables into the compiled
            # function state, like the target variables. `Model.quantize`
            # leaves the tracker locked. Unlock it for the assignment, so a
            # quantized target can track the assistant.
            tracker_was_locked = self._tracker.locked
            self._tracker.unlock()
            try:
                self._assistant_model = assistant_model
            finally:
                if tracker_was_locked:
                    self._tracker.lock()
            outputs = super().generate(
                inputs,
                max_length=max_length,
                stop_token_ids=stop_token_ids,
                strip_prompt=strip_prompt,
            )
            # A weak reference does not keep the assistant alive.
            self._assisted_generate_cache = (
                key,
                weakref.ref(assistant_model),
                self.sampler,
                self.generate_function,
            )
            return outputs
        finally:
            # Delete the attribute. Assigning `None` would keep the
            # dependency in the TF checkpoint of the target.
            if hasattr(self, "_assistant_model"):
                del self._assistant_model
            if not assistant_was_tracked and self._tracker.is_in_store(
                "layers", assistant_model
            ):
                self._tracker.untrack(assistant_model)
            self.sampler = original_sampler
            self.generate_function = original_generate_function
