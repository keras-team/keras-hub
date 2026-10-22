import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.qwen3_vl.qwen3_vl_backbone import Qwen3VLBackbone
from keras_hub.src.models.qwen3_vl.qwen3_vl_causal_lm_preprocessor import (
    Qwen3VLCausalLMPreprocessor,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_layers import _flatten_vision_inputs
from keras_hub.src.models.qwen3_vl.qwen3_vl_layers import flat_vision_indices
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.Qwen3VLCausalLM")
class Qwen3VLCausalLM(CausalLM):
    """An end-to-end Qwen3-VL model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. When the backbone has a `vision_encoder`, the model also takes
    images and videos, which the preprocessor turns into `pixel_values`,
    `image_grid_thw`, `vision_indices` and `position_ids`.

    This model has a `generate()` method, which generates text based on a
    prompt. The vision encoder runs once on the prompt. Decoding only uses
    the KV cache.

    Args:
        backbone: A `keras_hub.models.Qwen3VLBackbone` instance.
        preprocessor: A `keras_hub.models.Qwen3VLCausalLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.
    """

    backbone_cls = Qwen3VLBackbone
    preprocessor_cls = Qwen3VLCausalLMPreprocessor

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

    def score(
        self,
        token_ids,
        padding_mask=None,
        scoring_mode="logits",
        layer_intercept_fn=None,
        target_ids=None,
    ):
        """Score a generation represented by the provided token ids.

        Args:
            token_ids: A `<int>[batch_size, num_tokens]` tensor containing
                tokens to score, or a dict of preprocessor outputs with
                `"token_ids"` and optionally `"padding_mask"`,
                `"position_ids"`, `"pixel_values"`, `"image_grid_thw"` and
                `"vision_indices"`, so images and videos are scored too.
            padding_mask: A `<bool>[batch_size, num_tokens]` tensor
                indicating the tokens that should be preserved during
                generation. A `"padding_mask"` in `token_ids` takes
                precedence.
            scoring_mode: The type of scores to return, either `"logits"` or
                `"loss"`, both will be per input token.
            layer_intercept_fn: An optional function for augmenting
                activations with additional computation, for example, as
                part of interpretability research. Called with the
                embeddings and index `-1`, then with the output of each
                decoder layer and its index.
            target_ids: An `<int>[batch_size, num_tokens]` tensor containing
                the predicted tokens against which the loss should be
                computed.

        Returns:
            The per-token scores as a tensor of size
            `<float>[batch_size, num_tokens, vocab_size]` in `"logits"` mode,
            or `<float>[batch_size, num_tokens]` in `"loss"` mode.
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
        if layer_intercept_fn is None:

            def layer_intercept_fn(x, unused_i):
                return x

        inputs = token_ids
        if not isinstance(inputs, dict):
            inputs = {"token_ids": token_ids}
        inputs = dict(inputs)
        if "padding_mask" not in inputs:
            if padding_mask is None:
                padding_mask = ops.ones(ops.shape(inputs["token_ids"])[:2])
            inputs["padding_mask"] = padding_mask
        backbone = self.backbone
        inputs = backbone._fill_inputs(inputs)

        # The same steps as the backbone's functional model.
        x = backbone.token_embedding(inputs["token_ids"])
        deepstack_features = []
        if backbone.vision_encoder is not None:
            image_embeddings, *deepstack_features = backbone.vision_encoder(
                {
                    "pixel_values": inputs["pixel_values"],
                    "image_grid_thw": inputs["image_grid_thw"],
                }
            )
            x = backbone.interleave_embeddings(
                image_embeddings=image_embeddings,
                text_embeddings=x,
                vision_indices=inputs["vision_indices"],
            )
        x = layer_intercept_fn(x, -1)
        position_ids = ops.transpose(inputs["position_ids"], (0, 2, 1))
        rotary_embeddings = backbone.rotary_embedding.compute_embeddings(
            position_ids
        )
        for i, transformer_layer in enumerate(backbone.transformer_layers):
            x = transformer_layer(
                x,
                decoder_padding_mask=inputs["padding_mask"],
                position_ids=position_ids,
                rotary_embeddings=rotary_embeddings,
            )
            if i < len(deepstack_features):
                x = backbone.deepstack_inject(
                    x, deepstack_features[i], inputs["vision_indices"]
                )
            x = layer_intercept_fn(x, i)
        x = backbone.layer_norm(x)
        logits = backbone.token_embedding(x, reverse=True)

        if scoring_mode == "logits":
            return logits
        per_token_loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        return per_token_loss_fn(target_ids, logits)

    def _normalize_generate_inputs(self, inputs):
        # The base class wraps every dict value on its own, which splits a
        # single prompt from its media list. A single string prompt owns all
        # the other entries, so wrap each of them as one per-sample entry.
        if (
            self.preprocessor is not None
            and isinstance(inputs, dict)
            and isinstance(inputs.get("prompts"), str)
        ):
            return [{key: [value] for key, value in inputs.items()}], True
        return super()._normalize_generate_inputs(inputs)

    def _embed_prompt(
        self, token_ids, img_embeddings, deepstack_features, vision_indices
    ):
        """Embed the prompt and densify its DeepStack contributions.

        Returns:
            An `(embeddings, deepstack_updates)` tuple. `embeddings` has shape
            `(batch, seq_len, hidden_dim)` with the vision features scattered
            in. `deepstack_updates` holds one tensor of that shape per
            DeepStack layer, zero outside the vision tokens.
        """
        x = self.backbone.token_embedding(token_ids)
        if img_embeddings is None:
            return x, []
        embeddings = self.backbone.interleave_embeddings(
            image_embeddings=img_embeddings,
            text_embeddings=x,
            vision_indices=vision_indices,
        )
        # Scatter each feature straight into a dense update instead of
        # adding it to a zero tensor.
        shape = ops.shape(embeddings)
        hidden_dim = self.backbone.hidden_dim
        updates = []
        for feature in deepstack_features:
            rows, indices = _flatten_vision_inputs(
                feature, vision_indices, hidden_dim, shape[1]
            )
            update = ops.scatter(
                indices,
                ops.cast(rows, embeddings.dtype),
                (shape[0] * shape[1], hidden_dim),
            )
            updates.append(ops.reshape(update, shape))
        return embeddings, updates

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
        padding_mask=None,
        position_ids=None,
        embeddings=None,
        deepstack_updates=None,
    ):
        """Forward pass of `Qwen3VLCausalLM` with cache.

        Args:
            token_ids: int Tensor `(batch, seq_len)`.
            cache: float Tensor, the key and value cache of every layer.
            cache_update_index: int or int Tensor. The index of the first
                token of `token_ids` in the cache.
            padding_mask: Optional padding mask for the prefill call.
            position_ids: Optional int Tensor `(batch, 3, seq_len)` with the
                T, H and W positions. `None` means plain 1D positions.
            embeddings: Optional float Tensor `(batch, seq_len, hidden_dim)`
                used instead of embedding `token_ids`.
            deepstack_updates: Optional list of float Tensors, added to the
                hidden states after the first decoder layers.

        Returns:
            A `(logits, hidden_states, cache)` tuple.
        """
        hidden_states, cache = self._decode_with_cache(
            token_ids,
            cache,
            cache_update_index,
            padding_mask=padding_mask,
            position_ids=position_ids,
            embeddings=embeddings,
            deepstack_updates=deepstack_updates,
        )
        logits = self.backbone.token_embedding(hidden_states, reverse=True)
        return logits, hidden_states, cache

    def _decode_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
        padding_mask=None,
        position_ids=None,
        embeddings=None,
        deepstack_updates=None,
    ):
        """`call_with_cache` without the vocabulary projection.

        Returns:
            A `(hidden_states, cache)` tuple.
        """
        x = embeddings
        if x is None:
            x = self.backbone.token_embedding(token_ids)
        deepstack_updates = deepstack_updates or []
        updated_cache = []
        layers = self.backbone.transformer_layers
        rotary_embeddings = self.backbone.rotary_embedding.compute_embeddings(
            position_ids,
            sequence_length=ops.shape(x)[1],
            start_index=cache_update_index,
        )
        for i, transformer_layer in enumerate(layers):
            x, next_cache = transformer_layer(
                x,
                decoder_padding_mask=padding_mask,
                position_ids=position_ids,
                rotary_embeddings=rotary_embeddings,
                cache=cache[:, i, ...],
                cache_update_index=cache_update_index,
            )
            updated_cache.append(next_cache)
            if i < len(deepstack_updates):
                x = x + ops.cast(deepstack_updates[i], x.dtype)
        cache = ops.stack(updated_cache, axis=1)
        return self.backbone.layer_norm(x), cache

    def _build_cache(
        self,
        token_ids,
        padding_mask,
        position_ids,
        embeddings=None,
        deepstack_updates=None,
    ):
        """Build an empty cache and fill it with the prompt."""
        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]
        shape = [
            batch_size,
            self.backbone.num_layers,
            2,
            max_length,
            self.backbone.num_key_value_heads,
            self.backbone.head_dim,
        ]
        cache = ops.zeros(shape, dtype=self.compute_dtype)
        # The prompt logits are never used, so skip the vocabulary
        # projection over every prompt position.
        hidden_states, cache = self._decode_with_cache(
            token_ids,
            cache,
            0,
            padding_mask=padding_mask,
            position_ids=position_ids,
            embeddings=embeddings,
            deepstack_updates=deepstack_updates,
        )
        return hidden_states, cache

    def generate_step(self, inputs, stop_token_ids=None):
        """A compilable generation function for a batch of prompts.

        `inputs` is the dict the preprocessor returns. Vision keys are only
        used by the prefill pass.
        """
        token_ids = inputs["token_ids"]
        padding_mask = inputs["padding_mask"]
        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]

        position_ids = inputs.get("position_ids", None)
        if position_ids is None:
            position_ids = ops.broadcast_to(
                ops.arange(max_length, dtype="int32")[None, :, None],
                (batch_size, max_length, 3),
            )
        position_ids = ops.transpose(ops.cast(position_ids, "int32"), (0, 2, 1))

        img_embeddings = deepstack_features = None
        vision_indices = inputs.get("vision_indices", None)
        pixel_values = inputs.get("pixel_values", None)
        # Text-only prompts carry zero patches. The shape is static while
        # tracing, so the vision tower stays out of the graph for them.
        if (
            self.backbone.vision_encoder is not None
            and pixel_values is not None
            and pixel_values.shape[1] != 0
        ):
            img_embeddings, *deepstack_features = self.backbone.vision_encoder(
                {
                    "pixel_values": pixel_values,
                    "image_grid_thw": inputs["image_grid_thw"],
                }
            )

        prompt_embeddings, prompt_updates = self._embed_prompt(
            token_ids, img_embeddings, deepstack_features, vision_indices
        )
        hidden_states, cache = self._build_cache(
            token_ids,
            padding_mask,
            position_ids,
            embeddings=prompt_embeddings,
            deepstack_updates=prompt_updates,
        )

        # HF's rope_deltas: a text token after the prompt sits at its slot
        # index plus (max prompt position + 1 - prompt length), because
        # images advance the position by less than their token count.
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        prompt_max = ops.max(
            ops.where(
                ops.cast(padding_mask, "bool")[:, None, :], position_ids, -1
            ),
            axis=(1, 2),
        )
        delta = prompt_max + 1 - row_lengths
        decode_positions = ops.arange(max_length, dtype="int32")[None, :]
        decode_positions = decode_positions + delta[:, None]
        decode_positions = ops.broadcast_to(
            decode_positions[:, None, :], (batch_size, 3, max_length)
        )
        # Slots inside a longer prompt of the batch keep their prompt
        # positions when the cache is overwritten.
        in_prompt = ops.cast(padding_mask, "bool")[:, None, :]
        position_table = ops.where(in_prompt, position_ids, decode_positions)

        index = ops.min(row_lengths)
        # `BeamSampler` repeats each prompt row `num_beams` times, so the
        # per-prompt tensors captured below need the same repetition.
        num_beams = getattr(self.sampler, "num_beams", 1)

        def per_beam(tensor):
            if num_beams == 1:
                return tensor
            return ops.repeat(tensor, num_beams, axis=0)

        # The steps only read these, so they are repeated for the beams once.
        beam_positions = per_beam(position_table)
        if img_embeddings is not None:
            beam_prompt_mask = per_beam(ops.cast(padding_mask, "bool"))
            # Index into the flattened batch, so the steps match against
            # every vision token at once.
            flat_indices = flat_vision_indices(vision_indices, max_length)[
                None, :
            ]
            image_rows = ops.reshape(
                img_embeddings, (-1, ops.shape(img_embeddings)[-1])
            )
            deepstack_rows = [
                ops.reshape(f, (-1, ops.shape(f)[-1]))
                for f in deepstack_features
            ]

        def next(prompt, cache, index):
            cache_update_index = index - 1
            rows = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, cache_update_index], [rows, 1])
            step_positions = ops.slice(
                beam_positions, [0, 0, cache_update_index], [rows, 3, 1]
            )
            # The sampler re-feeds the slots of the longer prompts in a
            # batch. Those keep their vision features and DeepStack updates
            # instead of the plain token embedding. Only the vision tokens
            # carry features, so look the slot up among them rather than
            # keeping dense `(batch, seq_len, hidden_dim)` tensors around.
            embeddings = updates = None
            if img_embeddings is not None:
                slot_in_prompt = ops.slice(
                    beam_prompt_mask, [0, cache_update_index], [rows, 1]
                )
                row_ids = ops.arange(rows, dtype="int32") // num_beams
                flat_slot = row_ids * max_length + cache_update_index
                match = ops.equal(flat_indices, flat_slot[:, None])
                is_vision = ops.logical_and(
                    ops.any(match, axis=1, keepdims=True), slot_in_prompt
                )[:, :, None]
                # A slot in the prompt matches at most one vision token, so
                # gather that row. Padded indices only point at padding
                # slots, which `slot_in_prompt` masks out.
                row = ops.argmax(match, axis=1)

                def vision_feature(feature_rows):
                    feature = ops.take(feature_rows, row, axis=0)[:, None, :]
                    # The vision encoder may run in a different dtype than
                    # the decoder, as when a preset is loaded in bfloat16.
                    return ops.cast(feature, self.compute_dtype)

                embeddings = ops.where(
                    is_vision,
                    vision_feature(image_rows),
                    self.backbone.token_embedding(prompt),
                )
                updates = [
                    ops.where(is_vision, vision_feature(feature), 0.0)
                    for feature in deepstack_rows
                ]
            logits, hidden_states, cache = self.call_with_cache(
                prompt,
                cache,
                cache_update_index,
                position_ids=step_positions,
                embeddings=embeddings,
                deepstack_updates=updates,
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
