from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import Qwen2VLBackbone
from keras_hub.src.models.qwen2_vl.qwen2_vl_causal_lm_preprocessor import (
    Qwen2VLCausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.Qwen2VLCausalLM")
class Qwen2VLCausalLM(CausalLM):
    """End-to-end Qwen2-VL model for causal vision-language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on
    plain text input, or to autoregressively generate plain text similar to
    the data used for training. This task can be used for pre-training or
    fine-tuning a Qwen2-VL model, simply by calling `fit()`.

    This model has a `generate()` method, which generates text based on a
    prompt and optional image(s). The generation strategy used is controlled
    by an additional `sampler` argument on `compile()`. You can recompile the
    model with different `keras_hub.samplers` objects to control the
    generation. By default, `"greedy"` sampling will be used.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to string inputs
    during `fit()`, `predict()`, `evaluate()`, and `generate()`. This is done
    by default when creating the model with `from_preset()`.

    Args:
        backbone: A `keras_hub.models.Qwen2VLBackbone` instance.
        preprocessor: A `keras_hub.models.Qwen2VLCausalLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.

    Examples:

    Use `generate()` to do vision-language generation.
    ```python
    qwen2_vl_lm = keras_hub.models.Qwen2VLCausalLM.from_preset(
        "qwen2_vl_2b_instruct"
    )
    qwen2_vl_lm.generate(
        {"prompts": "Describe this image", "images": image},
        max_length=128
    )
    ```

    Use `generate()` with batched prompts and images.
    ```python
    qwen2_vl_lm.generate(
        {
            "prompts": ["What is in this image?", "Describe the scene"],
            "images": [image1, image2]
        },
        max_length=128
    )
    ```

    Compile the `generate()` function with a custom sampler.
    ```python
    qwen2_vl_lm = keras_hub.models.Qwen2VLCausalLM.from_preset(
        "qwen2_vl_2b_instruct"
    )
    qwen2_vl_lm.compile(sampler="top_k")
    qwen2_vl_lm.generate(
        {"prompts": "What do you see?", "images": image},
        max_length=128
    )

    qwen2_vl_lm.compile(sampler=keras_hub.samplers.BeamSampler(num_beams=2))
    qwen2_vl_lm.generate(
        {"prompts": "Describe this", "images": image},
        max_length=128
    )
    ```

    Use `generate()` without preprocessing.
    ```python
    prompt = {
        "token_ids": np.array([[151644, 872, 4320]] * 2),
        "padding_mask": np.array([[1, 1, 1]] * 2),
        "patch_values": np.random.rand(256, 1176),
        "image_grid_thw": np.array([[2, 8, 8]]),
    }

    qwen2_vl_lm = keras_hub.models.Qwen2VLCausalLM.from_preset(
        "qwen2_vl_2b_instruct",
        preprocessor=None,
    )
    qwen2_vl_lm.generate(prompt)
    ```

    Call `fit()` on a single batch.
    ```python
    features = {
        "prompts": ["Describe this image", "What is this?"],
        "images": [image1, image2],
    }
    qwen2_vl_lm = keras_hub.models.Qwen2VLCausalLM.from_preset(
        "qwen2_vl_2b_instruct"
    )
    qwen2_vl_lm.fit(x=features, batch_size=2)
    ```

    Call `fit()` without preprocessing.
    ```python
    x = {
        "token_ids": np.array([[151644, 872, 4320, 151645]] * 2),
        "padding_mask": np.array([[1, 1, 1, 1]] * 2),
        "patch_values": np.random.rand(256, 1176),
        "image_grid_thw": np.array([[2, 8, 8], [2, 8, 8]]),
    }
    y = np.array([[872, 4320, 151645, 0]] * 2)
    sw = np.array([[1, 1, 1, 0]] * 2)

    qwen2_vl_lm = keras_hub.models.Qwen2VLCausalLM.from_preset(
        "qwen2_vl_2b_instruct",
        preprocessor=None,
    )
    qwen2_vl_lm.fit(x=x, y=y, sample_weight=sw, batch_size=2)
    ```

    Custom backbone and vocabulary.
    ```python
    tokenizer = keras_hub.models.Qwen2VLTokenizer(
        vocabulary="./vocab.json",
        merges="./merges.txt",
    )
    preprocessor = keras_hub.models.Qwen2VLCausalLMPreprocessor(
        tokenizer=tokenizer,
        sequence_length=128,
    )
    backbone = keras_hub.models.Qwen2VLBackbone(
        vocabulary_size=50000,
        num_layers=12,
        num_query_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
    )
    qwen2_vl_lm = keras_hub.models.Qwen2VLCausalLM(
        backbone=backbone,
        preprocessor=preprocessor,
    )
    qwen2_vl_lm.fit(x=features, batch_size=2)
    ```
    """

    backbone_cls = Qwen2VLBackbone
    preprocessor_cls = Qwen2VLCausalLMPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional model ===
        inputs = backbone.input
        hidden_states = backbone(inputs=inputs)
        outputs = backbone.token_embedding(hidden_states, reverse=True)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
        img_embeddings=None,
    ):
        """Forward pass of `Qwen2VLCausalLM` with cache.

        `call_with_cache` adds an additional forward pass for the model for
        autoregressive inference. Unlike calling the model directly, this
        method allows caching previous key/value Tensors in multi-head
        attention layer, and avoids recomputing the outputs of seen tokens.

        Args:
            token_ids: a dense int Tensor with shape
                `(batch_size, max_length)`.
            cache: a dense float Tensor, the cache of key and value.
            cache_update_index: int, or int Tensor. The index of current
                inputs in the whole sequence.
            img_embeddings: optional float Tensor of shape
                `(total_merged_tokens, hidden_dim)`. Pre-computed vision
                features from the vision encoder. When provided, they are
                scattered into the text embeddings at positions matching
                ``image_token_id``.

        Returns:
            A (logits, hidden_states, cache) tuple.
        """
        x = self.backbone.token_embedding(token_ids)

        # Scatter vision features into image placeholder positions.
        if img_embeddings is not None:
            image_mask = ops.equal(
                token_ids,
                ops.cast(self.backbone.image_token_id, token_ids.dtype),
            )
            batch_size = ops.shape(x)[0]
            seq_len = ops.shape(x)[1]
            x_flat = ops.reshape(x, (-1, self.backbone.hidden_dim))
            mask_flat = ops.reshape(image_mask, (-1,))
            vision_indices = ops.where(mask_flat)
            if isinstance(vision_indices, (list, tuple)):
                vision_indices = vision_indices[0]
            vision_indices = ops.reshape(vision_indices, (-1, 1))
            vision_indices = ops.cast(vision_indices, "int32")
            x_flat = ops.scatter_update(x_flat, vision_indices, img_embeddings)
            x = ops.reshape(
                x_flat, (batch_size, seq_len, self.backbone.hidden_dim)
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

    def _build_cache(self, token_ids, img_embeddings=None):
        """Build an empty cache for use with `call_with_cache()`.

        Args:
            token_ids: int Tensor of shape `(batch_size, max_length)`.
            img_embeddings: optional float Tensor of pre-computed vision
                features to scatter into image placeholder positions
                during the initial seeding pass.
        """
        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]
        num_layers = self.backbone.num_layers
        num_key_value_heads = self.backbone.num_key_value_heads
        head_dim = self.backbone.hidden_dim // self.backbone.num_query_heads
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
        )
        return hidden_states, cache

    def generate_step(
        self,
        inputs,
        stop_token_ids=None,
    ):
        """A compilable generation function for a single batch of inputs.

        This function represents the inner, XLA-compilable, generation function
        for a single batch of inputs. Inputs should have the same structure as
        model inputs, a dictionary with keys `"token_ids"`, `"padding_mask"`,
        and optionally `"patch_values"` and `"image_grid_thw"`.

        Args:
            inputs: A dictionary with keys `"token_ids"`, `"padding_mask"`,
                and optionally `"patch_values"` and `"image_grid_thw"`.
            stop_token_ids: Tuple of id's of the end token to stop on. If all
                sequences have produced a new stop token, generation
                will stop.
        """
        token_ids, padding_mask = inputs["token_ids"], inputs["padding_mask"]
        patch_values = inputs.get("patch_values", None)
        image_grid_thw = inputs.get("image_grid_thw", None)

        # Run vision encoder if images are present.
        img_embeddings = None
        if patch_values is not None and image_grid_thw is not None:
            img_embeddings = self.backbone.vision_encoder(
                patch_values, image_grid_thw
            )

        # Create and seed cache with a single forward pass.
        hidden_states, cache = self._build_cache(
            token_ids,
            img_embeddings=img_embeddings,
        )
        # Compute the lengths of all user inputted tokens ids.
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        # Start at the first index that has no user inputted id.
        index = ops.min(row_lengths)

        def next_token(prompt, cache, index):
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
            next=next_token,
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
            # Build a mask of stop token locations not in the original
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
