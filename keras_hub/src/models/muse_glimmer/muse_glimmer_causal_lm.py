import itertools
from functools import partial
from types import SimpleNamespace

import keras
from keras import ops
from keras import tree

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_causal_lm_preprocessor import (  # noqa: E501
    MuseGlimmerCausalLMPreprocessor,
)
from keras_hub.src.samplers.greedy_sampler import GreedySampler
from keras_hub.src.samplers.speculative_sampler import SpeculativeSampler
from keras_hub.src.utils.tensor_utils import any_equal


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

    def generate_step(self, inputs, stop_token_ids=None, assistant_model=None):
        """Run one compiled generation pass.

        Args:
            inputs: dict. The preprocessed generation inputs.
            stop_token_ids: optional tuple of int. Token ids that end
                generation.
            assistant_model: optional
                `keras_hub.models.MuseGlimmerAssistantCausalLM`. When set,
                the pass uses DFlash speculative decoding. `generate()`
                binds this argument. The model never stores the
                assistant on `self`.
        """
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
        sampler_model = self

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

            # The sampler loop reads variables through `model`. Expose the
            # assistant variables without tracking the assistant on `self`.
            sampler_model = SimpleNamespace(
                trainable_variables=(
                    self.trainable_variables
                    + assistant_model.trainable_variables
                ),
                non_trainable_variables=(
                    self.non_trainable_variables
                    + assistant_model.non_trainable_variables
                ),
            )

        token_ids = self.sampler(
            next=next,
            prompt=token_ids,
            cache=cache,
            index=index,
            mask=padding_mask,
            stop_token_ids=stop_token_ids,
            hidden_states=hidden_states,
            model=sampler_model,
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

    def _make_assisted_generate_function(self, assistant_model, sampler):
        """Build the generate function for one assistant.

        The function passes the assistant to `generate_step()` as an
        argument. On JAX, the assistant variables go into the compiled
        function as explicit state, like the target variables.
        """
        generate_step = partial(
            self.generate_step, assistant_model=assistant_model
        )
        backend = keras.config.backend()
        if backend == "torch":
            import torch

            def torch_generate_function(inputs, stop_token_ids=None):
                with torch.no_grad():
                    return generate_step(inputs, stop_token_ids)

            return torch_generate_function
        if backend == "tensorflow" and not self.run_eagerly:
            import tensorflow as tf

            jit_compile = getattr(self, "jit_compile", True)
            return tf.function(generate_step, jit_compile=jit_compile)
        if backend == "jax" and not self.run_eagerly:
            import jax

            def get_model_variables():
                return (
                    self.trainable_variables
                    + self.non_trainable_variables
                    + assistant_model.trainable_variables
                    + assistant_model.non_trainable_variables
                )

            @partial(jax.jit, static_argnames=["stop_token_ids"])
            def compiled_generate_function(inputs, stop_token_ids, state):
                sampler_values, model_values = state
                mapping = itertools.chain(
                    zip(sampler.variables, sampler_values),
                    zip(get_model_variables(), model_values),
                )
                with keras.StatelessScope(state_mapping=mapping) as scope:
                    outputs = generate_step(inputs, stop_token_ids)
                sampler_values = []
                for v in sampler.variables:
                    new_v = scope.get_current_value(v)
                    sampler_values.append(new_v if new_v is not None else v)
                return outputs, sampler_values

            def jax_generate_function(inputs, stop_token_ids=None):
                if isinstance(stop_token_ids, list):
                    stop_token_ids = tuple(stop_token_ids)
                state = (
                    [v.value for v in sampler.variables],
                    [v.value for v in get_model_variables()],
                )
                inputs = tree.map_structure(ops.convert_to_tensor, inputs)
                outputs, sampler_values = compiled_generate_function(
                    inputs, stop_token_ids, state
                )
                for ref_v, v in zip(sampler.variables, sampler_values):
                    ref_v.assign(v)
                return outputs

            return jax_generate_function
        return generate_step

    def _get_assisted_generate_function(self, assistant_model):
        """Return a cached `(sampler, generate_function)` pair.

        A greedy target keeps greedy acceptance. Any other compiled
        sampler becomes the `base_sampler`, so acceptance and the bonus
        token use stochastic rejection sampling. The cache key holds the
        assistant identity, the block size, and the base sampler. The
        cached function keeps both objects alive, so their ids stay
        unique.
        """
        base_sampler = self.sampler
        if isinstance(base_sampler, GreedySampler):
            base_sampler = None
        key = (
            id(assistant_model),
            assistant_model.block_size,
            id(base_sampler),
        )
        cached = getattr(self, "_assisted_generate_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1], cached[2]
        sampler = SpeculativeSampler(
            num_speculative_tokens=assistant_model.block_size - 1,
            base_sampler=base_sampler,
            temperature=getattr(self.sampler, "temperature", 1.0),
        )
        generate_function = self._make_assisted_generate_function(
            assistant_model, sampler
        )
        self._assisted_generate_cache = (key, sampler, generate_function)
        return sampler, generate_function

    def _post_quantize(self, mode, **kwargs):
        super()._post_quantize(mode, **kwargs)
        # The cached assisted function holds the old target variables.
        self._assisted_generate_cache = None

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
                model caches the compiled function per assistant and does
                not track the assistant weights. Defaults to `None`
                (plain autoregressive decoding with the compiled
                `sampler`).
        """
        if assistant_model is None:
            return super().generate(
                inputs,
                max_length=max_length,
                stop_token_ids=stop_token_ids,
                strip_prompt=strip_prompt,
            )

        speculative_sampler, generate_function = (
            self._get_assisted_generate_function(assistant_model)
        )
        original_sampler = self.sampler
        original_generate_function = self.generate_function
        self.sampler = speculative_sampler
        self.generate_function = generate_function
        try:
            return super().generate(
                inputs,
                max_length=max_length,
                stop_token_ids=stop_token_ids,
                strip_prompt=strip_prompt,
            )
        finally:
            self.sampler = original_sampler
            self.generate_function = original_generate_function
