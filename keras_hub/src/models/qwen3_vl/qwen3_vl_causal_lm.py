from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.qwen3_vl.qwen3_vl_attention import (
    compute_mrope_embeddings,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_backbone import Qwen3_VLBackbone
from keras_hub.src.models.qwen3_vl.qwen3_vl_causal_lm_preprocessor import (
    Qwen3_VLCausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.Qwen3_VLCausalLM")
class Qwen3_VLCausalLM(CausalLM):
    """An end-to-end Qwen3-VL model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. When the backbone has a `vision_encoder`, the model also takes
    images and videos, which the preprocessor turns into `pixel_values`,
    `image_grid_thw`, `vision_indices` and `position_ids`.

    This model has a `generate()` method, which generates text based on a
    prompt. The vision encoder runs once on the prompt. Decoding only uses
    the KV cache.

    Args:
        backbone: A `keras_hub.models.Qwen3_VLBackbone` instance.
        preprocessor: A `keras_hub.models.Qwen3_VLCausalLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.
    """

    backbone_cls = Qwen3_VLBackbone
    preprocessor_cls = Qwen3_VLCausalLMPreprocessor

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
        updates = [
            self.backbone.deepstack_inject(
                ops.zeros_like(embeddings), feature, vision_indices
            )
            for feature in deepstack_features
        ]
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
        """Forward pass of `Qwen3_VLCausalLM` with cache.

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
        x = embeddings
        if x is None:
            x = self.backbone.token_embedding(token_ids)
        deepstack_updates = deepstack_updates or []
        updated_cache = []
        layers = self.backbone.transformer_layers
        rotary_embeddings = compute_mrope_embeddings(
            self.backbone.head_dim,
            self.backbone.rope_max_wavelength,
            self.backbone.mrope_section,
            ops.shape(x)[1],
            cache_update_index,
            position_ids,
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
        hidden_states = x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, hidden_states, cache

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
        _, hidden_states, cache = self.call_with_cache(
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
        if self.backbone.vision_encoder is not None and (
            pixel_values is not None
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

        def next(prompt, cache, index):
            cache_update_index = index - 1
            rows = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, cache_update_index], [rows, 1])
            step_positions = ops.slice(
                per_beam(position_table),
                [0, 0, cache_update_index],
                [rows, 3, 1],
            )
            # The sampler re-feeds the slots of the longer prompts in a
            # batch. Those keep their vision features and DeepStack updates
            # instead of the plain token embedding. Only the vision tokens
            # carry features, so look the slot up among them rather than
            # keeping dense `(batch, seq_len, hidden_dim)` tensors around.
            embeddings = updates = None
            if img_embeddings is not None:
                slot_in_prompt = ops.slice(
                    per_beam(ops.cast(padding_mask, "bool")),
                    [0, cache_update_index],
                    [rows, 1],
                )
                row_ids = ops.arange(rows, dtype="int32") // num_beams
                flat_slot = row_ids * max_length + cache_update_index
                match = per_beam(ops.cast(vision_indices, "int32"))
                match = ops.equal(match, flat_slot[:, None])
                is_vision = ops.logical_and(
                    ops.any(match, axis=1, keepdims=True), slot_in_prompt
                )[:, :, None]
                weights = ops.cast(match, img_embeddings.dtype)[:, :, None]

                def vision_feature(features):
                    features = per_beam(features)
                    feature = ops.sum(
                        features * ops.cast(weights, features.dtype),
                        axis=1,
                        keepdims=True,
                    )
                    # The vision encoder may run in a different dtype than
                    # the decoder, as when a preset is loaded in bfloat16.
                    return ops.cast(feature, self.compute_dtype)

                embeddings = ops.where(
                    is_vision,
                    vision_feature(img_embeddings),
                    self.backbone.token_embedding(prompt),
                )
                updates = [
                    ops.where(is_vision, vision_feature(feature), 0.0)
                    for feature in deepstack_features
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
