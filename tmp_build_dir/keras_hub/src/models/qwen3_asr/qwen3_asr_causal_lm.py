import numpy as np
from keras import ops

try:
    import tensorflow as tf
except ImportError:
    tf = None

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone
from keras_hub.src.models.qwen3_asr.qwen3_asr_preprocessor import (
    Qwen3ASRPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.Qwen3ASRCausalLM")
class Qwen3ASRCausalLM(CausalLM):
    """An end-to-end Qwen3 ASR model for causal language modeling.

    Args:
        backbone: A `keras_hub.models.Qwen3ASRBackbone` instance.
        preprocessor: A `keras_hub.models.Qwen3ASRPreprocessor` or `None`.
            If `None`, this model will not apply preprocessing, and inputs
            should be preprocessed before calling the model.
    """

    backbone_cls = Qwen3ASRBackbone
    preprocessor_cls = Qwen3ASRPreprocessor

    def __init__(
        self,
        backbone,
        preprocessor=None,
        **kwargs,
    ):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
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
            inputs = inputs.copy()
            inputs["prompts"], input_is_scalar = normalize(inputs["prompts"])

            if input_is_scalar and "audio" in inputs:
                x = inputs["audio"]
                if isinstance(x, np.ndarray) and len(x.shape) == 1:
                    inputs["audio"] = [x]
                elif tf and isinstance(x, tf.Tensor) and x.shape.rank == 1:
                    inputs["audio"] = x[tf.newaxis]
                elif isinstance(x, list):
                    if len(x) > 0 and isinstance(x[0], (int, float)):
                        inputs["audio"] = [x]
        else:
            inputs, input_is_scalar = normalize(inputs)

        return [inputs], input_is_scalar

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
        audio_embeds=None,
        padding_mask=None,
    ):
        inputs_embeds = self.backbone.token_embedding(token_ids)

        if audio_embeds is not None:
            inputs_embeds = self.backbone.scatter_audio(
                inputs_embeds, audio_embeds, token_ids
            )

        x = inputs_embeds
        caches = []
        for i, transformer_layer in enumerate(self.backbone.transformer_layers):
            current_cache = cache[:, i, ...]
            x, next_cache = transformer_layer(
                x,
                self_attention_cache=current_cache,
                self_attention_cache_update_index=cache_update_index,
                decoder_padding_mask=padding_mask,
            )
            caches.append(next_cache)
        cache = ops.stack(caches, axis=1)
        hidden_states = x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, hidden_states, cache

    def _build_cache(self, token_ids, audio_embeds, padding_mask):
        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]

        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_key_value_heads
        head_dim = self.backbone.head_dim
        shape = [batch_size, num_layers, 2, max_length, num_heads, head_dim]
        cache = ops.zeros(shape, dtype=self.compute_dtype)

        # Seed the cache.
        logits, hidden_states, cache = self.call_with_cache(
            token_ids=token_ids,
            audio_embeds=audio_embeds,
            cache=cache,
            cache_update_index=0,
            padding_mask=padding_mask,
        )
        return hidden_states, cache

    def generate_step(self, inputs, stop_token_ids=None):
        token_ids, padding_mask = (
            inputs["token_ids"],
            inputs["padding_mask"],
        )
        audio_mel, audio_mask = (
            inputs["audio_mel"],
            inputs["audio_mask"],
        )

        # Run audio encoder once
        audio_embeds = self.backbone.audio_encoder(audio_mel, audio_mask)
        audio_embeds = self.backbone.projector(audio_embeds)

        # Create and seed cache with a single forward pass.
        hidden_states, cache = self._build_cache(
            token_ids, audio_embeds, padding_mask
        )

        # Compute the lengths of all user inputted tokens ids.
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        # Start at the first index that has no user inputted id.
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            cache_update_index = index - 1
            batch_size = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, index - 1], [batch_size, 1])
            logits, hidden_states, cache = self.call_with_cache(
                token_ids=prompt,
                cache=cache,
                cache_update_index=cache_update_index,
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

        # Compute an output padding mask with the token ids we updated.
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

        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "audio_mel": audio_mel,
            "audio_mask": audio_mask,
        }
