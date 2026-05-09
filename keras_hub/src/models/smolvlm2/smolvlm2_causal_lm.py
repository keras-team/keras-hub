import numpy as np
import tensorflow as tf
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.smolvlm2.smolvlm2_backbone import SmolVLM2Backbone
from keras_hub.src.models.smolvlm2.smolvlm2_causal_lm_preprocessor import (
    SmolVLM2CausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.SmolVLM2CausalLM")
class SmolVLM2CausalLM(CausalLM):
    """SmolVLM2 causal language model for multimodal text generation.

    A causal language model (LM) that predicts the next token based on
    previous tokens and optional image inputs. This model can be used
    for image captioning, visual question answering, video
    understanding, and general text generation.

    This model has a `generate()` method, which generates text based on
    a prompt (with optional images). The generation strategy is
    controlled by an additional `sampler` argument on `compile()`. You
    can recompile the model with different `keras_hub.samplers` objects
    to control the generation. By default, `"greedy"` sampling is used.

    This model can optionally be configured with a `preprocessor`
    layer, in which case it will automatically apply preprocessing to
    string inputs during `fit()`, `predict()`, `evaluate()` and
    `generate()`.

    Args:
        backbone: A `keras_hub.models.SmolVLM2Backbone` instance.
        preprocessor: A `keras_hub.models.SmolVLM2CausalLMPreprocessor`
            or `None`.

    Examples:

    Use `generate()` for text generation.
    ```python
    smolvlm2_lm = keras_hub.models.SmolVLM2CausalLM.from_preset(
        "smolvlm2_2.2b_instruct"
    )
    smolvlm2_lm.generate("Hello, world!")
    ```
    """

    backbone_cls = SmolVLM2Backbone
    preprocessor_cls = SmolVLM2CausalLMPreprocessor

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

    def __call__(self, inputs, *args, **kwargs):
        """Override to inject default empty vision inputs for text-only calls.

        When the CausalLM receives text-only inputs (no `pixel_values`
        or `vision_indices`), this injects zero-sized dummy tensors so
        the functional graph receives all required keys. This follows
        the Qwen3.5/Gemma4 pattern.
        """
        if isinstance(inputs, dict):
            inputs = dict(inputs)  # shallow copy
            if "pixel_values" not in inputs:
                batch_size = ops.shape(inputs["token_ids"])[0]
                inputs["pixel_values"] = ops.zeros(
                    (
                        batch_size,
                        self.backbone.image_size,
                        self.backbone.image_size,
                        3,
                    ),
                )
            if "vision_indices" not in inputs:
                batch_size = ops.shape(inputs["token_ids"])[0]
                inputs["vision_indices"] = ops.zeros(
                    (batch_size, 0), dtype="int32"
                )
        return super().__call__(inputs, *args, **kwargs)

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
        """Handle unbatched image inputs for generation."""
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
            inputs["prompts"], input_is_scalar = normalize(inputs["prompts"])

            if input_is_scalar and "images" in inputs:
                x = inputs["images"]
                if isinstance(x, np.ndarray) and len(x.shape) == 3:
                    inputs["images"] = [x]
                elif tf and isinstance(x, tf.Tensor) and x.shape.rank == 3:
                    inputs["images"] = x[tf.newaxis]
                elif isinstance(x, list):
                    inputs["images"] = [x]

            if "responses" in inputs:
                inputs["responses"], _ = normalize(inputs["responses"])
        else:
            inputs, input_is_scalar = normalize(inputs)

        return [inputs], input_is_scalar

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
        padding_mask=None,
        img_embeddings=None,
        vision_indices=None,
    ):
        """Forward pass of `SmolVLM2CausalLM` with cache.

        `call_with_cache` adds an additional forward pass for the model
        for autoregressive inference. Unlike calling the model directly,
        this method allows caching previous key/value Tensors in
        multi-head attention layer, and avoids recomputing the outputs
        of seen tokens.

        Args:
            token_ids: A dense int Tensor of shape
                `(batch_size, seq_len)`.
            cache: A dense float Tensor, the cache of key and value.
            cache_update_index: int, or int Tensor. The index of
                current inputs in the whole sequence.
            padding_mask: Optional. A dense int Tensor of shape
                `(batch_size, seq_len)`.
            img_embeddings: Optional. A dense float Tensor of shape
                `(batch_size, image_sequence_length, hidden_dim)`.
                Pre-computed vision embeddings from vision encoder +
                connector. Only used on the first (prefill) call.
            vision_indices: Optional. int32 Tensor of flat indices
                for scattering vision tokens. Only used on prefill.

        Returns:
            A (logits, hidden_states, cache) tuple.
        """
        text_embeddings = self.backbone.token_embedding(token_ids)

        # Interleave vision embeddings on the prefill step.
        if img_embeddings is not None and vision_indices is not None:
            x = self.backbone.interleave_embeddings(
                image_embeddings=img_embeddings,
                text_embeddings=text_embeddings,
                vision_indices=vision_indices,
            )
        else:
            x = text_embeddings

        # Run through decoder layers with cache.
        caches = []
        for i, transformer_layer in enumerate(self.backbone.transformer_layers):
            current_cache = cache[:, i, ...]
            x, next_cache = transformer_layer(
                x,
                cache=current_cache,
                cache_update_index=cache_update_index,
                decoder_padding_mask=padding_mask,
            )
            caches.append(next_cache)
        cache = ops.stack(caches, axis=1)
        hidden_states = x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, hidden_states, cache

    def _build_cache(
        self,
        token_ids,
        padding_mask,
        img_embeddings=None,
        vision_indices=None,
    ):
        """Build an empty cache for use with `call_with_cache()`."""
        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_key_value_heads
        head_dim = self.backbone.hidden_dim // self.backbone.num_query_heads
        shape = [
            batch_size,
            num_layers,
            2,
            max_length,
            num_heads,
            head_dim,
        ]
        cache = ops.zeros(shape, dtype=self.compute_dtype)
        # Seed the cache.
        logits, hidden_states, cache = self.call_with_cache(
            token_ids=token_ids,
            img_embeddings=img_embeddings,
            vision_indices=vision_indices,
            cache=cache,
            cache_update_index=0,
            padding_mask=padding_mask,
        )
        return hidden_states, cache

    def generate_step(self, inputs, stop_token_ids=None):
        """A compilable generation function for a single batch of inputs.

        This function represents the inner, XLA-compilable, generation
        function for a single batch of inputs. For multimodal inputs,
        the preprocessor populates extra keys like `pixel_values` and
        `vision_indices`. These are consumed on the first forward pass
        (cache prefill) and then dropped for subsequent autoregressive
        steps.

        Args:
            inputs: A dictionary with keys `"token_ids"` and
                `"padding_mask"`, with batched tensor values.
            stop_token_ids: Tuple of id's of end tokens to stop on.
        """
        token_ids = inputs["token_ids"]
        padding_mask = inputs["padding_mask"]

        # Check for multimodal inputs.
        pixel_values = inputs.get("pixel_values", None)
        vision_indices = inputs.get("vision_indices", None)

        # Run vision encoder + connector if we have pixel data.
        img_embeddings = None
        if pixel_values is not None:
            # Handle unbatched images.
            if len(ops.shape(pixel_values)) == 3:
                pixel_values = ops.expand_dims(pixel_values, axis=0)

            # Encode images through the vision encoder + connector.
            img_embeddings = self.backbone.vision_encoder(
                {"pixel_values": pixel_values}
            )
            img_embeddings = self.backbone.connector(img_embeddings)

        # Create and seed cache with a single forward pass.
        hidden_states, cache = self._build_cache(
            token_ids,
            padding_mask,
            img_embeddings=img_embeddings,
            vision_indices=vision_indices,
        )
        # Compute the lengths of all user-inputted token ids.
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        # Start at the first index that has no user-inputted id.
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            # The cache index is the index of our previous token.
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
        }

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
            token_ids: A `<int>[batch_size, num_tokens]` tensor
                containing tokens to score.
            padding_mask: A `<bool>[batch_size, num_tokens]` tensor
                indicating the tokens that should be preserved during
                generation.
            scoring_mode: The type of scores to return, either
                `"logits"` or `"loss"`, both will be per input token.
            layer_intercept_fn: An optional function for augmenting
                activations with additional computation.
            target_ids: An `<int>[batch_size, num_tokens]` tensor
                containing the predicted tokens against which the loss
                should be computed.

        Returns:
            The per-token scores as a tensor.
        """
        import keras

        if scoring_mode not in ("logits", "loss"):
            raise ValueError(
                "Unsupported scoring_mode. Must be 'logits' or 'loss'."
            )

        if scoring_mode == "loss" and target_ids is None:
            raise ValueError(
                "Cannot compute loss without targets. Please provide "
                "target token ids via the target_ids parameter."
            )

        batch_shape = ops.shape(token_ids)[:2]
        assert len(batch_shape) == 2

        if padding_mask is None:
            padding_mask = ops.ones(shape=batch_shape)

        if layer_intercept_fn is None:

            def default_layer_intercept_fn(x, unused_i):
                return x

            layer_intercept_fn = default_layer_intercept_fn

        token_embeddings = self.backbone.token_embedding(token_ids)
        x = layer_intercept_fn(token_embeddings, -1)

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
