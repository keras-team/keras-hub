"""BLIP-2 Causal LM model."""

import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.blip2.blip2_backbone import BLIP2Backbone
from keras_hub.src.models.blip2.blip2_causal_lm_preprocessor import (
    BLIP2CausalLMPreprocessor,
)
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.BLIP2CausalLM")
class BLIP2CausalLM(CausalLM):
    """An end-to-end multimodal BLIP-2 model for causal language modeling.

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
        preprocessor: A `keras_hub.models.BLIP2CausalLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.
        backbone: A `keras_hub.models.BLIP2Backbone` instance.
    """

    backbone_cls = BLIP2Backbone
    preprocessor_cls = BLIP2CausalLMPreprocessor

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
        inputs = backbone.input
        hidden_states = backbone(inputs)
        outputs = backbone.token_embedding(hidden_states, reverse=True)
        # Sliced outputs to match text-only labels.
        outputs = outputs[:, backbone.num_query_tokens :, :]

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
        """Normalizes raw inputs into a batched list for generation."""
        if self.preprocessor is None:
            return [inputs], False

        def normalize(x):
            if isinstance(x, str):
                return [x], True
            return x, False

        if isinstance(inputs, dict):
            if "text" in inputs:
                inputs["text"], input_is_scalar = normalize(inputs["text"])
            else:
                input_is_scalar = False

            if input_is_scalar and "images" in inputs:
                x = inputs["images"]
                if isinstance(x, (np.ndarray, list)) and len(np.shape(x)) == 3:
                    inputs["images"] = [x]
        else:
            inputs, input_is_scalar = normalize(inputs)

        return [inputs], input_is_scalar

    def _encode_images(self, images):
        """Run vision encoder → Q-Former → language projection."""
        if ops.ndim(images) == 3:
            images = ops.expand_dims(images, axis=0)
        vision_features = self.backbone.vision_encoder(images)
        qformer_features = self.backbone.qformer(vision_features)
        return self.backbone.language_model.language_projection(qformer_features)

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
        projected_features=None,
        padding_mask=None,
    ):
        """Language-model forward pass with a KV cache."""
        lm = self.backbone.language_model
        num_visual = self.backbone.num_query_tokens

        if projected_features is not None:
            token_embeds = lm.embeddings_layer(token_ids)

            visual_pos_ids = ops.expand_dims(
                ops.arange(2, 2 + num_visual, dtype="int32"), axis=0
            )
            visual_pos_embeds = lm.embeddings_layer.position_embedding(visual_pos_ids)
            projected_features = projected_features + ops.cast(
                visual_pos_embeds, projected_features.dtype
            )

            x = ops.concatenate([projected_features, token_embeds], axis=1)

            batch_size = ops.shape(token_ids)[0]
            visual_mask = ops.ones((batch_size, num_visual), dtype="bool")
            if padding_mask is not None:
                full_padding_mask = ops.concatenate(
                    [visual_mask, ops.cast(padding_mask, "bool")], axis=1
                )
            else:
                text_len = ops.shape(token_ids)[1]
                text_mask = ops.ones((batch_size, text_len), dtype="bool")
                full_padding_mask = ops.concatenate([visual_mask, text_mask], axis=1)

        else:
            position_ids = ops.reshape(
                ops.cast(cache_update_index + 2, "int32"), (1, 1)
            )
            token_embeds = lm.embeddings_layer(token_ids, position_ids=position_ids)
            x = token_embeds
            full_padding_mask = None

        hidden_states, new_cache = lm.call_with_cache(
            x, full_padding_mask, cache, cache_update_index
        )

        logits = lm.embeddings_layer.token_embedding(hidden_states, reverse=True)

        if projected_features is not None:
            logits = logits[:, num_visual:, :]

        return logits, hidden_states, new_cache

    def _build_cache(self, token_ids, projected_features, padding_mask):
        """Build an empty cache for use with `call_with_cache()`."""
        lm = self.backbone.language_model
        num_visual = self.backbone.num_query_tokens
        batch_size = ops.shape(token_ids)[0]
        text_length = ops.shape(token_ids)[1]
        max_length = text_length + num_visual

        cache_shape = [
            batch_size,
            lm.num_layers,
            2,
            max_length,
            lm.num_heads,
            lm.hidden_dim // lm.num_heads,
        ]
        cache = ops.zeros(cache_shape, dtype=self.compute_dtype)

        logits, hidden_states, cache = self.call_with_cache(
            token_ids=token_ids,
            cache=cache,
            cache_update_index=0,
            projected_features=projected_features,
            padding_mask=padding_mask,
        )
        return hidden_states, cache

    def generate_step(self, inputs, stop_token_ids=None):
        """A compilable generation function for a single batch of inputs."""
        token_ids = inputs["token_ids"]
        padding_mask = inputs["padding_mask"]
        images = inputs.get("images")
        num_visual = self.backbone.num_query_tokens

        if images is not None:
            projected_features = self._encode_images(images)
        else:
            projected_features = None

        hidden_states, cache = self._build_cache(
            token_ids, projected_features, padding_mask
        )

        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            cache_update_index = index - 1 + num_visual
            batch_size = ops.shape(prompt)[0]
            prompt_slice = ops.slice(prompt, [0, index - 1], [batch_size, 1])
            logits, hidden_states, cache = self.call_with_cache(
                token_ids=prompt_slice,
                cache=cache,
                cache_update_index=cache_update_index,
                projected_features=None,
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

        outputs = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
        if images is not None:
            outputs["images"] = images
        return outputs

    def generate(
        self,
        inputs,
        max_length=None,
        stop_token_ids="auto",
        strip_prompt=False,
    ):
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
            stop_token_ids = [self.preprocessor.tokenizer.end_token_id]

        if not isinstance(inputs, dict):
            inputs = {"text": inputs}

        return super().generate(
            inputs,
            max_length=max_length,
            stop_token_ids=stop_token_ids,
            strip_prompt=strip_prompt,
        )
