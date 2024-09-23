# Copyright 2024 The KerasHub Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writingf, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.pali_gemma.pali_gemma_backbone import (
    PaliGemmaBackbone,
)
from keras_hub.src.models.pali_gemma.pali_gemma_causal_lm_preprocessor import (
    PaliGemmaCausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.PaliGemmaCausalLM")
class PaliGemmaCausalLM(CausalLM):
    """An end-to-end multi modal PaliGemma model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on
    image and plain text input, or to autoregressively generate plain text
    similar to the data used for training.

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
        backbone: A `keras_hub.models.PaliGemmaBackbone` instance.
        preprocessor: A `keras_hub.models.PaliGemmaCausalLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.

    Examples:

    Use `generate()` to do text generation.
    ```python
    image = np.random.rand(224, 224, 3)
    pali_gemma_lm = keras_hub.models.PaliGemmaCausalLM.from_preset(
        "pali_gemma_3b_mix_224"
    )
    pali_gemma_lm.generate(
      {
        "images": image,
        "text": ["answer en where is the cow standing?\\n"]
      }
    )

    # Generate with batched prompts.
    pali_gemma_lm.generate(
      {
        "images": [image, image],
        "text": ["answer en where is the cow standing?\\n", "caption en\\n"]
      }
    )
    ```

    Use `generate()` without preprocessing.
    ```python
    image = np.random.rand(224, 224, 3)
    inputs = {
        "images": [image, image],
        # Token ids for "<bos> Keras is".
        "token_ids": np.array([[2, 214064, 603, 0, 0, 0, 0]] * 2),
        # Use `"padding_mask"` to indicate values that should not be overridden.
        "padding_mask": np.array([[1, 1, 1, 0, 0, 0, 0]] * 2),
    }

    pali_gemma_lm = keras_hub.models.PaliGemmaCausalLM.from_preset(
        "pali_gemma_3b_mix_224",
        preprocessor=None,
    )
    pali_gemma_lm.generate(inputs)
    ```

    Custom backbone and vocabulary.
    ```python
    tokenizer = keras_hub.models.PaliGemmaTokenizer(
        proto="proto.spm",
    )
    preprocessor = keras_hub.models.PaliGemmaCausalLMPreprocessor(
        tokenizer=tokenizer,
        sequence_length=128,
    )
    backbone = keras_hub.models.PaliGemmaBackbone()
    pali_gemma_lm = keras_hub.models.PaliGemmaCausalLM(
        backbone=backbone,
        preprocessor=preprocessor,
    )
    ```
    """

    backbone_cls = PaliGemmaBackbone
    preprocessor_cls = PaliGemmaCausalLMPreprocessor

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
        inputs = backbone.inputs
        hidden_state = backbone(inputs=inputs)
        outputs = backbone.token_embedding(hidden_state, reverse=True)
        outputs = outputs[:, backbone.image_sequence_length :, :]
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
        img_embeddings=None,
        padding_mask=None,
    ):
        """Forward pass of `PaliGemmaCausalLM` with cache.

        `call_with_cache` adds an additional forward pass for the model for
        autoregressive inference. Unlike calling the model directly, this method
        allows caching previous key/value Tensors in multi-head attention layer,
        and avoids recomputing the outputs of seen tokens.

        Args:
            token_ids: a dense int Tensor with shape `(batch_size, max_length)`.
            cache: a dense float Tensor, the cache of key and value.
            cache_update_index: int, or int Tensor. The index of current inputs
                in the whole sequence.
            img_embeddings: a dense float Tensor with shape
                `(batch_size, image_sequence_length, hidden_dim)`.
            padding_mask: a dense int Tensor with shape
                `(batch_size, max_length)`.

        Returns:
            A (logits, hidden_states, cache) tuple. Where `logits` is the
            language model logits for the input token_ids, `hidden_states` is
            the final hidden representation of the input tokens, and `cache` is
            the decoding cache.
        """
        text_embeddings = self.backbone.token_embedding(token_ids)
        text_embeddings = text_embeddings * ops.cast(
            ops.sqrt(self.backbone.hidden_dim), text_embeddings.dtype
        )

        if img_embeddings is not None:
            x = ops.concatenate((img_embeddings, text_embeddings), axis=1)
        else:
            x = text_embeddings

        # Each decoder layer has a cache; we update them separately.
        caches = []
        for i, transformer_layer in enumerate(self.backbone.transformer_layers):
            current_cache = cache[:, i, ...]
            x, next_cache = transformer_layer(
                x,
                cache=current_cache,
                cache_update_index=cache_update_index,
                padding_mask=padding_mask,
            )
            caches.append(next_cache)
        cache = ops.stack(caches, axis=1)
        hidden_states = x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, hidden_states, cache

    def _build_cache(self, token_ids, img_embeddings, padding_mask):
        """Build an empty cache for use with `call_with_cache()`."""
        batch_size = ops.shape(token_ids)[0]
        max_length = (
            ops.shape(token_ids)[1] + self.backbone.image_sequence_length
        )
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_key_value_heads
        head_dim = self.backbone.head_dim
        shape = [batch_size, num_layers, 2, max_length, num_heads, head_dim]
        cache = ops.zeros(shape, dtype=self.compute_dtype)
        # Seed the cache.
        logits, hidden_states, cache = self.call_with_cache(
            token_ids=token_ids,
            img_embeddings=img_embeddings,
            cache=cache,
            cache_update_index=0,
            padding_mask=padding_mask,
        )
        return hidden_states, cache

    def generate_step(self, inputs, stop_token_ids=None):
        """A compilable generation function for a single batch of inputs.

        This function represents the inner, XLA-compilable, generation function
        for a single batch of inputs. Inputs should have the same structure as
        model inputs, a dictionary with keys `"token_ids"` and `"padding_mask"`.

        Args:
            inputs: A dictionary with two keys `"token_ids"` and
                `"padding_mask"` and batched tensor values.
            stop_token_ids: Tuple of id's of end token's to stop on. If all
                sequences have produced a new stop token, generation
                will stop.
        """
        token_ids, padding_mask, images = (
            inputs["token_ids"],
            inputs["padding_mask"],
            inputs["images"],
        )
        if len(ops.shape(images)) == 3:
            # Handle an unbatched image. Unlike `token_ids` and `padding_mask`
            # this will not automatically be upranked.
            images = ops.expand_dims(images, axis=0)
        img_embeddings = self.backbone.vit_encoder(images)

        # Create and seed cache with a single forward pass.
        hidden_states, cache = self._build_cache(
            token_ids, img_embeddings, padding_mask
        )
        # Compute the lengths of all user inputted tokens ids.
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        # Start at the first index that has no user inputted id.
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            # The cache index is the index of our previous token.
            cache_update_index = index - 1 + self.backbone.image_sequence_length
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
            # Build a mask of `stop_token_ids` locations not in the original
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
            "images": images,
        }
