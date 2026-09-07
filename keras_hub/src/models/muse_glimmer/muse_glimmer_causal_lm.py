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
    ):
        """Forward pass with a KV cache, for autoregressive decoding."""
        x = self.backbone.token_embedding(token_ids)
        x = self.backbone.embed_norm(x)

        if img_embeddings is not None and vision_indices is not None:
            x = self.backbone.interleave_embeddings(
                image_embeddings=img_embeddings,
                text_embeddings=x,
                vision_indices=vision_indices,
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
        next_cache = ops.stack(next_cache, axis=1)

        hidden_states = x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        logits = self._apply_logit_softcap(logits)
        return logits, hidden_states, next_cache

    def _build_cache(
        self, token_ids, padding_mask, img_embeddings=None, vision_indices=None
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
        hidden_states, cache = self.call_with_cache(
            token_ids,
            cache,
            0,
            padding_mask=padding_mask,
            img_embeddings=img_embeddings,
            vision_indices=vision_indices,
        )[1:]
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

        hidden_states, cache = self._build_cache(
            token_ids,
            padding_mask,
            img_embeddings=img_embeddings,
            vision_indices=vision_indices,
        )
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
        return {"token_ids": token_ids, "padding_mask": padding_mask}
