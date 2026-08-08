import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone
from keras_hub.src.models.qwen3_asr.qwen3_asr_causal_lm_preprocessor import (
    Qwen3ASRCausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.Qwen3ASRCausalLM")
class Qwen3ASRCausalLM(CausalLM):
    """An end-to-end Qwen3-ASR model for causal language modeling.

    Args:
        backbone: A `keras_hub.models.Qwen3ASRBackbone` instance.
        preprocessor: A `keras_hub.models.Qwen3ASRCausalLMPreprocessor` or
            `None`.
    """

    backbone_cls = Qwen3ASRBackbone
    preprocessor_cls = Qwen3ASRCausalLMPreprocessor

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

    def _interleave(self, x, audio_embeddings, audio_indices):
        if audio_embeddings is not None and audio_indices is not None:
            if hasattr(self.backbone, "interleave_layer"):
                return self.backbone.interleave_layer(
                    audio_embeddings=audio_embeddings,
                    text_embeddings=x,
                    audio_indices=audio_indices,
                )
        return x

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
        audio_embeddings=None,
        audio_indices=None,
        padding_mask=None,
    ):
        """Forward pass of `Qwen3ASRCausalLM` with cache."""
        x = self.backbone.token_embedding(token_ids)
        x = self._interleave(x, audio_embeddings, audio_indices)

        updated_cache = []
        for i in range(self.backbone.num_layers):
            current_cache = cache[:, i, ...]
            x, next_cache = self.backbone.transformer_layers[i](
                x,
                decoder_padding_mask=padding_mask,
                self_attention_cache=current_cache,
                self_attention_cache_update_index=cache_update_index,
            )
            updated_cache.append(next_cache)
        cache = ops.stack(updated_cache, axis=1)
        hidden_states = x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, hidden_states, cache

    def _build_cache(
        self,
        token_ids,
        audio_embeddings=None,
        audio_indices=None,
        padding_mask=None,
    ):
        """Build an empty cache for use with `call_with_cache()`."""
        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]
        num_layers = self.backbone.num_layers
        num_key_value_heads = self.backbone.num_key_value_heads
        head_dim = self.backbone.head_dim
        shape = [
            batch_size,
            num_layers,
            2,
            max_length,
            num_key_value_heads,
            head_dim,
        ]
        cache = ops.zeros(shape, dtype=self.compute_dtype)
        # Seed the cache.
        _, hidden_states, cache = self.call_with_cache(
            token_ids=token_ids,
            cache=cache,
            cache_update_index=0,
            audio_embeddings=audio_embeddings,
            audio_indices=audio_indices,
            padding_mask=padding_mask,
        )
        return hidden_states, cache

    def generate_step(
        self,
        inputs,
        stop_token_ids=None,
    ):
        """A compilable generation function for a single batch of inputs."""
        token_ids, padding_mask = inputs["token_ids"], inputs["padding_mask"]
        audio_mel = inputs.get("audio_mel", None)
        audio_mel_mask = inputs.get("audio_mel_mask", None)
        audio_indices = inputs.get("audio_indices", None)

        if self.backbone.audio_encoder is not None and audio_mel is not None:
            audio_embeddings = self.backbone.audio_encoder(
                audio_mel, audio_mel_mask=audio_mel_mask
            )
        else:
            audio_embeddings = None
            audio_indices = None

        # Create and seed cache with a single forward pass.
        hidden_states, cache = self._build_cache(
            token_ids,
            audio_embeddings=audio_embeddings,
            audio_indices=audio_indices,
            padding_mask=padding_mask,
        )

        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            cache_update_index = index - 1
            batch_size = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, cache_update_index], [batch_size, 1])

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
        }

    def score(
        self,
        token_ids,
        padding_mask=None,
        audio_embeddings=None,
        audio_indices=None,
        scoring_mode="logits",
        layer_intercept_fn=None,
        target_ids=None,
    ):
        """Score a generation represented by the provided token ids."""
        if scoring_mode not in ("logits", "loss"):
            raise ValueError(
                "Unsupported scoring_mode. Must be one of 'logits' or 'loss'."
            )

        if scoring_mode == "loss" and target_ids is None:
            raise ValueError(
                "Cannot compute loss without targets. Please provide target "
                "token ids via the target_ids parameter."
            )

        batch_shape = ops.shape(token_ids)[:2]

        if padding_mask is None:
            padding_mask = ops.ones(shape=batch_shape, dtype="bool")

        if layer_intercept_fn is None:

            def default_layer_intercept_fn(x, unused_i):
                return x

            layer_intercept_fn = default_layer_intercept_fn

        x = self.backbone.token_embedding(token_ids)
        x = layer_intercept_fn(x, -1)

        x = self._interleave(x, audio_embeddings, audio_indices)

        for i, transformer_layer in enumerate(self.backbone.transformer_layers):
            x = transformer_layer(x, decoder_padding_mask=padding_mask)
            x = layer_intercept_fn(x, i)
        x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)

        if scoring_mode == "logits":
            return logits

        per_token_loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        per_token_loss = per_token_loss_fn(target_ids, logits)
        return per_token_loss
