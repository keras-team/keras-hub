from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.mistral.mistral_causal_lm import MistralCausalLM
from keras_hub.src.models.mistral3.mistral3_backbone import Mistral3Backbone
from keras_hub.src.models.mistral3.mistral3_causal_lm_preprocessor import (
    Mistral3CausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.Mistral3CausalLM")
class Mistral3CausalLM(MistralCausalLM):
    """An end-to-end Mistral3 model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on
    plain text input, or to autoregressively generate text (optionally
    grounded in one or more input images) similar to the data used for
    training. This task can be used for pre-training or fine-tuning a
    Mistral3 model, simply by calling `fit()`.

    This model has a `generate()` method, which generates text based on a
    prompt. The generation strategy used is controlled by an additional
    `sampler` argument on `compile()`. You can recompile the model with
    different `keras_hub.samplers` objects to control the generation. By
    default, `"top_k"` sampling will be used.

    Args:
        backbone: A `keras_hub.models.Mistral3Backbone` instance.
        preprocessor: A `keras_hub.models.Mistral3CausalLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.
    """

    backbone_cls = Mistral3Backbone
    preprocessor_cls = Mistral3CausalLMPreprocessor

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
        img_embeddings=None,
        placeholder_indices=None,
    ):
        """Forward pass of `Mistral3CausalLM` with cache.

        `call_with_cache` adds an additional forward pass for the model for
        autoregressive inference. Unlike calling the model directly, this method
        allows caching previous key/value Tensors in multi-head attention layer,
        and avoids recomputing the outputs of seen tokens.

        Args:
            token_ids: a dense int Tensor with shape `(batch_size, max_length)`.
            cache: a dense float Tensor, the cache of key and value.
            cache_update_index: int, or int Tensor. The index of current inputs
            in the whole sequence.
            img_embeddings: a dense float Tensor of projected image features,
                or `None` for a text-only forward pass. Scattered into
                `token_ids`' embeddings at `placeholder_indices` before the
                decoder layers run.
            placeholder_indices: flat positions of image placeholder tokens
                in `token_ids`. Required when `img_embeddings` is not
                `None`.

        Returns:
            A (logits, hidden_states, cache) tuple. Where `logits` is the
            language model logits for the input token_ids, `hidden_states` is
            the final hidden representation of the input tokens, and `cache` is
            the decoding cache.
        """
        x = self.backbone.token_embedding(token_ids)
        if img_embeddings is not None:
            x = self.backbone.image_text_embedding_merger(
                x, img_embeddings, placeholder_indices
            )
        # Each decoder layer has a cache; we update them separately.
        updated_cache = []
        for i in range(self.backbone.num_layers):
            current_cache = cache[:, i, ...]
            x, next_cache = self.backbone.transformer_layers[i](
                x,
                self_attention_cache=current_cache,
                self_attention_cache_update_index=cache_update_index,
            )
            updated_cache.append(next_cache)
        cache = ops.stack(updated_cache, axis=1)
        hidden_states = x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, hidden_states, cache

    def _build_cache(
        self, token_ids, img_embeddings=None, placeholder_indices=None
    ):
        """Build an empty cache for use with `call_with_cache()`."""
        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]
        num_layers = self.backbone.num_layers
        num_key_value_heads = self.backbone.num_key_value_heads
        head_dim = self.backbone.head_dim or (
            self.backbone.hidden_dim // self.backbone.num_query_heads
        )
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
            token_ids,
            cache,
            0,
            img_embeddings=img_embeddings,
            placeholder_indices=placeholder_indices,
        )
        return hidden_states, cache

    def generate_step(
        self,
        inputs,
        stop_token_ids=None,
    ):
        """A compilable generation function for a single batch of inputs.

        This function represents the inner, XLA-compilable, generation function
        for a single batch of inputs.

        Args:
            inputs: A dictionary with keys `"token_ids"` and
                `"padding_mask"`, and batched tensor values. When images are
                present, also includes `"pixel_values"`, `"image_sizes"`,
                and `"placeholder_indices"`.
            stop_token_ids: List of id's of end token's to stop on. If all
                sequences have produced a new stop token, generation
                will stop.
        """
        token_ids, padding_mask = inputs["token_ids"], inputs["padding_mask"]
        pixel_values = inputs.get("pixel_values", None)
        image_sizes = inputs.get("image_sizes", None)
        placeholder_indices = inputs.get("placeholder_indices", None)

        # Compute image features once, at prefill, from a static (Python
        # int, not tensor) shape check on the number of images. An unknown
        # static shape (`None`) is treated as "no images".
        img_embeddings = None
        if pixel_values is not None and pixel_values.shape[0]:
            img_embeddings = self.backbone.image_feature_extractor(
                pixel_values, image_sizes
            )
        else:
            placeholder_indices = None

        # Create and seed cache with a single forward pass.
        hidden_states, cache = self._build_cache(
            token_ids,
            img_embeddings=img_embeddings,
            placeholder_indices=placeholder_indices,
        )
        # Compute the lengths of all user inputted tokens ids.
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        # Start at the first index that has no user inputted id.
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
            # Build a mask of stop_tokens locations not in the original
            # prompt (not in locations where `padding_mask` is True).
            end_locations = any_equal(
                token_ids, stop_token_ids, ops.logical_not(padding_mask)
            )

            end_locations = ops.cast(end_locations, "int32")
            # Use cumsum to get ones in all locations after end_locations.
            cumsum = ops.cast(ops.cumsum(end_locations, axis=-1), "int32")
            overflow = cumsum - end_locations
            # Our padding mask is the inverse of these overflow locations.
            padding_mask = ops.logical_not(ops.cast(overflow, "bool"))
        else:
            # Without early stopping, all locations will have been updated.
            padding_mask = ops.ones_like(token_ids, dtype="bool")
        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
