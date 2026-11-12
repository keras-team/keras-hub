from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_backbone import (
    UnlimitedOCRBackbone,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_causal_lm_preprocessor import (  # noqa: E501
    UnlimitedOCRCausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.UnlimitedOCRCausalLM")
class UnlimitedOCRCausalLM(CausalLM):
    """An end-to-end Unlimited-OCR model for document parsing.

    A causal language model predicts the next token from the previous tokens.
    This task reads the text and the images of a prompt. It generates the
    parsed text of the page. You can use this task for fine-tuning with
    `fit()`.

    This model has a `generate()` method. The `sampler` argument of
    `compile()` controls the generation strategy. By default, the model uses
    `"top_k"` sampling. For OCR, use `sampler="greedy"`.

    During generation, the model uses the reference sliding window attention
    (R-SWA). Each new token attends to all prompt tokens, including the image
    tokens, and to the last `sliding_window` positions. Training and `score`
    use full causal attention.

    Args:
        backbone: A `keras_hub.models.UnlimitedOCRBackbone` instance.
        preprocessor: A `keras_hub.models.UnlimitedOCRCausalLMPreprocessor`
            or `None`. If `None`, the model does not preprocess its inputs.

    Example:
    ```python
    causal_lm = keras_hub.models.UnlimitedOCRCausalLM.from_preset(
        "unlimited_ocr"
    )
    causal_lm.compile(sampler="greedy")
    causal_lm.generate(
        {
            "prompts": "<image>document parsing.",
            "images": [page_image],
        },
        max_length=4096,
    )
    ```
    """

    backbone_cls = UnlimitedOCRBackbone
    preprocessor_cls = UnlimitedOCRCausalLMPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        # This must be "backbone.input" i.e. the full input structure,
        # rather than "backbone.inputs" which is the flattened list of inputs.
        inputs = backbone.input
        hidden_states = backbone(inputs)
        outputs = backbone.token_embedding(hidden_states, reverse=True)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
        image_features=None,
        patches_grid=None,
        prefill_length=None,
    ):
        """Forward pass of `UnlimitedOCRCausalLM` with cache.

        `call_with_cache` runs the model for autoregressive inference. It
        caches the key and value tensors of the attention layers. The cache
        avoids a new computation for the tokens that the model has seen.

        Args:
            token_ids: int tensor of shape `(batch_size, length)`.
            cache: float tensor. The cache of keys and values.
            cache_update_index: int or int tensor. The position of the first
                token of `token_ids` in the whole sequence.
            image_features: A `(global_features, local_features)` tuple of
                projected image features, or `None`. The features replace
                the embeddings of the image tokens before the decoder layers
                run.
            patches_grid: int tensor of shape `(batch_size, max_images, 2)`.
                Required when `image_features` is set.
            prefill_length: int tensor of shape `(batch_size,)`, or `None`.
                The number of prompt tokens of each row. When set, the
                attention layers use the R-SWA mask. The prefill pass of
                the cache uses `None`.

        Returns:
            A `(logits, hidden_states, cache)` tuple.
        """
        x = self.backbone.token_embedding(token_ids)
        if image_features is not None:
            global_features, local_features = image_features
            x = self.backbone.image_text_embedding_merger(
                token_ids, x, global_features, local_features, patches_grid
            )
        # Each decoder layer has a cache. The layers update them separately.
        updated_cache = []
        for i in range(self.backbone.num_layers):
            current_cache = cache[:, i, ...]
            x, next_cache = self.backbone.transformer_layers[i](
                x,
                self_attention_cache=current_cache,
                self_attention_cache_update_index=cache_update_index,
                prefill_length=prefill_length,
            )
            updated_cache.append(next_cache)
        cache = ops.stack(updated_cache, axis=1)
        hidden_states = x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, hidden_states, cache

    def _build_cache(self, token_ids, image_features=None, patches_grid=None):
        """Build an empty cache for use with `call_with_cache()`."""
        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]
        shape = [
            batch_size,
            self.backbone.num_layers,
            2,
            max_length,
            self.backbone.num_query_heads,
            self.backbone.head_dim,
        ]
        cache = ops.zeros(shape, dtype=self.compute_dtype)
        # Seed the cache. The prompt is the prefill, so every token attends
        # causally.
        _, hidden_states, cache = self.call_with_cache(
            token_ids,
            cache,
            0,
            image_features=image_features,
            patches_grid=patches_grid,
        )
        return hidden_states, cache

    def generate_step(self, inputs, stop_token_ids=None):
        """A compilable generation function for a single batch of inputs.

        Args:
            inputs: A dictionary with the keys `"token_ids"` and
                `"padding_mask"`. For images, it also has the keys
                `"pixel_values"`, `"pixel_values_local"` and
                `"patches_grid"`.
            stop_token_ids: List of ids of end tokens to stop on. If all
                sequences have produced a new stop token, generation stops.
        """
        token_ids, padding_mask = inputs["token_ids"], inputs["padding_mask"]

        # Compute the image features once, at prefill.
        image_features = None
        patches_grid = None
        if "pixel_values" in inputs:
            image_features = self.backbone.image_feature_extractor(
                inputs["pixel_values"], inputs["pixel_values_local"]
            )
            patches_grid = inputs["patches_grid"]

        hidden_states, cache = self._build_cache(
            token_ids,
            image_features=image_features,
            patches_grid=patches_grid,
        )
        # The lengths of the user prompts. The prompt tokens are the prefill
        # tokens of the R-SWA.
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        # Start at the first index that has no user token.
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            # The cache index is the index of our previous token.
            cache_update_index = index - 1
            batch_size = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, cache_update_index], [batch_size, 1])
            logits, hidden_states, cache = self.call_with_cache(
                prompt,
                cache,
                cache_update_index,
                prefill_length=row_lengths,
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
            # Build a mask of stop tokens that are not in the original prompt.
            end_locations = any_equal(
                token_ids, stop_token_ids, ops.logical_not(padding_mask)
            )
            end_locations = ops.cast(end_locations, "int32")
            # Use cumsum to get ones in all locations after the end locations.
            cumsum = ops.cast(ops.cumsum(end_locations, axis=-1), "int32")
            overflow = cumsum - end_locations
            # Our padding mask is the inverse of these overflow locations.
            padding_mask = ops.logical_not(ops.cast(overflow, "bool"))
        else:
            # Without early stopping, all locations have been updated.
            padding_mask = ops.ones_like(token_ids, dtype="bool")
        return {"token_ids": token_ids, "padding_mask": padding_mask}
