from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.moondream.moondream_backbone import MoondreamBackbone
from keras_hub.src.models.moondream.moondream_causal_lm_preprocessor import (
    MoondreamCausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.MoondreamCausalLM")
class MoondreamCausalLM(CausalLM):
    """An end-to-end Moondream model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. Moondream is a tiny vision-language model that accepts image and
    text inputs and generates text outputs. This task setup can be used to
    train the model unsupervised on image-text pairs, or to autoregressively
    generate text from image prompts.

    This model has a `generate()` method, which generates text based on an
    image and a text prompt. The generation strategy used is controlled by an
    additional `sampler` argument on `compile()`. You can recompile the model
    with different `keras_hub.samplers` objects to control the generation. By
    default, `"greedy"` sampling will be used.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to string inputs
    during `fit()`, `predict()`, `evaluate()` and `generate()`. This is done
    by default when creating the model with `from_preset()`.

    Args:
        backbone: A `keras_hub.models.MoondreamBackbone` instance.
        preprocessor: A `keras_hub.models.MoondreamCausalLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.

    Examples:

    Use `generate()` to do text generation from image prompts.
    ```python
    import numpy as np
    import keras_hub

    moondream_lm = keras_hub.models.MoondreamCausalLM.from_preset("moondream2")
    image = np.random.rand(378, 378, 3).astype("float32")
    moondream_lm.generate(
        {"images": image, "prompts": ["Describe this image."]}
    )

    # Generate with batched prompts.
    moondream_lm.generate(
        {
            "images": [image, image],
            "prompts": ["Describe this image.", "What is in the photo?"],
        }
    )
    ```

    Use `generate()` without preprocessing.
    ```python
    import numpy as np
    import keras_hub

    image = np.random.rand(378, 378, 3).astype("float32")
    inputs = {
        "images": np.stack([image, image]),
        # Token ids for a short prompt.
        "token_ids": np.array([[2, 1, 2, 0, 0, 0, 0]] * 2),
        # Use `"padding_mask"` to indicate values that should not be overridden.
        "padding_mask": np.array([[1, 1, 1, 0, 0, 0, 0]] * 2),
    }
    moondream_lm = keras_hub.models.MoondreamCausalLM.from_preset(
        "moondream2",
        preprocessor=None,
    )
    moondream_lm.generate(inputs)
    ```

    Custom backbone and vocabulary.
    ```python
    import numpy as np
    import keras_hub

    backbone = keras_hub.models.MoondreamBackbone(
        vocabulary_size=51200,
        image_size=378,
        vision_patch_size=14,
        vision_num_layers=2,
        vision_num_heads=2,
        vision_hidden_dim=8,
        vision_intermediate_dim=16,
        projection_dim=8,
        text_num_layers=2,
        text_hidden_dim=8,
        text_intermediate_dim=16,
        text_num_query_heads=2,
        text_num_key_value_heads=1,
    )
    moondream_lm = keras_hub.models.MoondreamCausalLM(backbone=backbone)
    ```
    """

    backbone_cls = MoondreamBackbone
    preprocessor_cls = MoondreamCausalLMPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        # Use "backbone.input" (the full input structure dict), not
        # "backbone.inputs" (the flattened list).
        inputs = backbone.input
        hidden_states = backbone(inputs=inputs)
        # Slice off the image patch positions; only text token positions
        # produce logits for the next-token prediction task.
        text_hidden_states = hidden_states[
            :, backbone.image_sequence_length :, :
        ]
        outputs = backbone.token_embedding(text_hidden_states, reverse=True)
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

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
        image_embeddings=None,
        padding_mask=None,
    ):
        """Forward pass of `MoondreamCausalLM` with cache.

        `call_with_cache` adds an additional forward pass for the model for
        autoregressive inference. Unlike calling the model directly, this method
        allows caching previous key/value tensors in multi-head attention
        layers, and avoids recomputing the outputs of seen tokens.

        Args:
            token_ids: a dense int Tensor with shape `(batch_size, seq_len)`.
            cache: a dense float Tensor, the cache of key and value.
            cache_update_index: int, or int Tensor. The index of the current
                inputs in the whole sequence.
            image_embeddings: optional dense float Tensor with shape
                `(batch_size, image_sequence_length, projection_dim)`. When
                `None`, no image prefix is prepended.
            padding_mask: optional dense int Tensor with shape
                `(batch_size, seq_len)`.

        Returns:
            A `(logits, hidden_states, cache)` tuple. `logits` are the
            language model logits for the input `token_ids`, `hidden_states`
            is the final hidden representation, and `cache` is the updated
            decoding cache.
        """
        text_embeddings = self.backbone.token_embedding(token_ids)

        if image_embeddings is not None:
            x = ops.concatenate([image_embeddings, text_embeddings], axis=1)
        else:
            x = text_embeddings

        # Each decoder layer has its own cache slot.
        updated_cache = []
        for i, text_layer in enumerate(
            self.backbone.text_transformer_layers
        ):
            current_cache = cache[:, i, ...]
            x, next_cache = text_layer(
                x,
                attention_cache=current_cache,
                attention_cache_update_index=cache_update_index,
            )
            updated_cache.append(next_cache)
        cache = ops.stack(updated_cache, axis=1)
        hidden_states = x = self.backbone.text_layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, hidden_states, cache

    def _build_cache(self, token_ids, image_embeddings, padding_mask):
        """Build an empty cache for use with `call_with_cache()`."""
        batch_size = ops.shape(token_ids)[0]
        # Total sequence length = image patches + text tokens.
        max_length = (
            ops.shape(token_ids)[1] + self.backbone.image_sequence_length
        )
        num_layers = self.backbone.text_num_layers
        num_key_value_heads = self.backbone.text_num_key_value_heads
        head_dim = (
            self.backbone.text_hidden_dim
            // self.backbone.text_num_query_heads
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
        # Seed the cache with a full forward pass.
        _, hidden_states, cache = self.call_with_cache(
            token_ids=token_ids,
            cache=cache,
            cache_update_index=0,
            image_embeddings=image_embeddings,
            padding_mask=padding_mask,
        )
        return hidden_states, cache

    def generate_step(self, inputs, stop_token_ids=None):
        """A compilable generation function for a single batch of inputs.

        This function represents the inner, XLA-compilable, generation function
        for a single batch of inputs. Inputs should have the same structure as
        model inputs: a dictionary with keys `"token_ids"`, `"padding_mask"`,
        and optionally `"images"`.

        Args:
            inputs: A dict with keys `"token_ids"`, `"padding_mask"`, and
                optionally `"images"`.
            stop_token_ids: Tuple of token ids to stop generation on. If all
                sequences have produced a stop token, generation will stop.
        """
        token_ids = inputs["token_ids"]
        padding_mask = inputs["padding_mask"]
        images = inputs.get("images", None)

        # Encode images into projected embeddings once.
        if images is not None:
            if len(ops.shape(images)) == 3:
                # Handle an unbatched image.
                images = ops.expand_dims(images, axis=0)
            # Run the vision encoder and projection.
            vision_x = self.backbone.vision_embedding(images)
            for vision_layer in self.backbone.vision_encoder_layers:
                vision_x = vision_layer(vision_x)
            vision_x = self.backbone.vision_layer_norm(vision_x)
            image_embeddings = self.backbone.vision_projection(vision_x)
        else:
            image_embeddings = None

        # Create and seed the KV cache with a single forward pass.
        hidden_states, cache = self._build_cache(
            token_ids, image_embeddings, padding_mask
        )

        # Compute the lengths of all user-inputted token ids.
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        # Start at the first index that has no user-inputted id.
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            # The cache index is the index of our previous token.
            cache_update_index = (
                index - 1 + self.backbone.image_sequence_length
            )
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

        out = {"token_ids": token_ids, "padding_mask": padding_mask}
        if images is not None:
            out["images"] = images
        return out
