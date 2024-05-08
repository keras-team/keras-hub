# Copyright 2024 The KerasNLP Authors
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
from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.backend import keras
from keras_nlp.src.backend import ops
from keras_nlp.src.models.causal_lm import CausalLM
from keras_nlp.src.models.pali_gemma.pali_gemma_backbone import (
    PaliGemmaBackbone,
)
from keras_nlp.src.models.pali_gemma.pali_gemma_causal_lm_preprocesor import (
    PaliGemmaCausalLMPreprocessor,
)
from keras_nlp.src.utils.tensor_utils import any_equal


@keras_nlp_export("keras_nlp.models.PaliGemmaCausalLM")
class PaliGemmaCausalLM(CausalLM):
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

    def call_with_cache(
        self,
        token_ids,
        img_embeddings,
        cache,
        cache_update_index,
    ):
        """Forward pass of `PaliGemmaCausalLM` with cache.

        `call_with_cache` adds an additional forward pass for the model for
        autoregressive inference. Unlike calling the model directly, this method
        allows caching previous key/value Tensors in multi-head attention layer,
        and avoids recomputing the outputs of seen tokens.

        Args:
            token_ids: a dense int Tensor with shape `(batch_size, max_length)`.
            img_embeddings: a dense int Tensor with shape `(batch_size, sequence_length, hidden_dim)`.
            cache: a dense float Tensor, the cache of key and value.
            cache_update_index: int, or int Tensor. The index of current inputs in the
                whole sequence.

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
            )
            caches.append(next_cache)
        cache = ops.stack(caches, axis=1)
        hidden_states = x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, hidden_states, cache

    def _build_cache(self, token_ids, img_embeddings):
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
            token_ids, img_embeddings, cache, 0
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
        img_embeddings = self.backbone.vit_encoder(images)

        # Create and seed cache with a single forward pass.
        hidden_states, cache = self._build_cache(token_ids, img_embeddings)
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
                prompt,
                None,
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

    def score(
        self,
        token_ids,
        img_embeddings,
        padding_mask=None,
        scoring_mode="logits",
        layer_intercept_fn=None,
        target_ids=None,
    ):
        """Score a generation represented by the provided token ids.

        Args:
            token_ids: A <int>[batch_size, num_tokens] tensor containing tokens
                to score. Typically, this tensor captures the output from a call
                to `PaliGemmaCausalLM.generate()`, i.e., tokens for both the input
                text and the model-generated text.
            img_embeddings: A <float32>[batch_size, resolution, resolution, 3] tensor containing
                image data for scoring.
            padding_mask: A <bool>[batch_size, num_tokens] tensor indicating the
                tokens that should be preserved during generation. This is an
                artifact required by the GemmaBackbone and isn't influential on
                the computation of this function. If omitted, this function uses
                `keras.ops.ones()` to create a tensor of the appropriate shape.
            scoring_mode: The type of scores to return, either "logits" or
                "loss", both will be per input token.
            layer_intercept_fn: An optional function for augmenting activations
                with additional computation, for example, as part of
                interpretability research. This function will be passed the
                activations as its first parameter and a numeric index
                associated with that backbone layer. _This index _is not_ an
                index into `self.backbone.layers`_. The index -1 accompanies the
                embeddings returned by calling `self.backbone.token_embedding()`
                on `token_ids` in the forward direction. All subsequent indexes
                will be 0-based indices for the activations returned by each of
                the Transformers layers in the backbone. This function must
                return a <float>[batch_size, num_tokens, hidden_dims] tensor
                that can be passed as an input to the next layer in the model.
            target_ids: An <bool>[batch_size, num_tokens] tensor containing the
                predicted tokens against which the loss should be computed. If a
                span of tokens is provided (sequential truthy values along
                axis=1 in the tensor), the loss will be computed as the
                aggregate across those tokens.

        Raises:
            ValueError: If an unsupported scoring_mode is provided, or if the
                target_ids are not provided when using ScoringMode.LOSS.

        Returns:
            The per-token scores as a tensor of size
            <float>[batch_size, num_tokens, vocab_size] in "logits" mode, or
            <float>[batch_size, num_tokens] in "loss" mode.

        Example:

        Compute gradients between embeddings and loss scores with TensorFlow:
        ```python
        pali_gemma_lm = keras_nlp.models.PaliGemmaCausalLM.from_preset(
            "gemma_2b_en"
        )
        generations = pali_gemma_lm.generate(
            ["This is a", "Where are you"],
            max_length=30
        )
        preprocessed = pali_gemma_lm.preprocessor.generate_preprocess(generations)
        generation_ids = preprocessed["token_ids"]
        padding_mask = preprocessed["padding_mask"]
        target_ids = keras.ops.roll(generation_ids, shift=-1, axis=1)

        embeddings = None
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            def layer_intercept_fn(x, i):
                if i == -1:
                    nonlocal embeddings, tape
                    embeddings = x
                    tape.watch(embeddings)
                return x

            losses = pali_gemma_lm.score(
                token_ids=generation_ids,
                padding_mask=padding_mask,
                scoring_mode="loss",
                layer_intercept_fn=layer_intercept_fn,
                target_ids=target_ids,
            )

        grads = tape.gradient(losses, embeddings)
        ```
        """
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
        assert len(batch_shape) == 2

        if padding_mask is None:
            padding_mask = ops.ones(shape=batch_shape)

        if layer_intercept_fn is None:

            def default_layer_intercept_fn(x, unused_i):
                return x

            layer_intercept_fn = default_layer_intercept_fn

        text_embeddings = self.backbone.token_embedding(token_ids)
        text_embeddings = layer_intercept_fn(text_embeddings, -1)

        text_embeddings = text_embeddings * ops.cast(
            ops.sqrt(self.backbone.hidden_dim), dtype=self.compute_dtype
        )

        img_embeddings = self.backbone.vit_encoder(img_embeddings)

        complete_sequence = keras.ops.concatenate(
            (img_embeddings, text_embeddings), axis=1
        )

        for i, transformer_layer in enumerate(self.backbone.transformer_layers):
            complete_sequence = transformer_layer(
                complete_sequence, padding_mask=padding_mask
            )
            complete_sequence = layer_intercept_fn(complete_sequence, i)
        complete_sequence = self.backbone.layer_norm(complete_sequence)
        logits = self.backbone.token_embedding(complete_sequence, reverse=True)

        if scoring_mode == "logits":
            return logits

        per_token_loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        per_token_loss = per_token_loss_fn(target_ids, logits)
        return per_token_loss
