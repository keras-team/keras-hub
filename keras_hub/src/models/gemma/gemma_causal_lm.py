import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.gemma.gemma_backbone import GemmaBackbone
from keras_hub.src.models.gemma.gemma_causal_lm_preprocessor import (
    GemmaCausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.GemmaCausalLM")
class GemmaCausalLM(CausalLM):
    """An end-to-end Gemma model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on
    plain text input, or to autoregressively generate plain text similar to
    the data used for training. This task can be used for pre-training or
    fine-tuning a Gemma model, simply by calling `fit()`.

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
        backbone: A `keras_hub.models.GemmaBackbone` instance.
        preprocessor: A `keras_hub.models.GemmaCausalLMPreprocessor` or `None`.
            If `None`, this model will not apply preprocessing, and inputs
            should be preprocessed before calling the model.

    Examples:

    Use `generate()` to do text generation.
    ```python
    gemma_lm = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b_en")
    gemma_lm.generate("I want to say", max_length=30)

    # Generate with batched prompts.
    gemma_lm.generate(["This is a", "Where are you"], max_length=30)
    ```

    Compile the `generate()` function with a custom sampler.
    ```python
    gemma_lm = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b_en")
    gemma_lm.compile(sampler="top_k")
    gemma_lm.generate("I want to say", max_length=30)

    gemma_lm.compile(sampler=keras_hub.samplers.BeamSampler(num_beams=2))
    gemma_lm.generate("I want to say", max_length=30)
    ```

    Use `generate()` without preprocessing.
    ```python
    prompt = {
        # Token ids for "<bos> Keras is".
        "token_ids": np.array([[2, 214064, 603, 0, 0, 0, 0]] * 2),
        # Use `"padding_mask"` to indicate values that should not be overridden.
        "padding_mask": np.array([[1, 1, 1, 0, 0, 0, 0]] * 2),
    }

    gemma_lm = keras_hub.models.GemmaCausalLM.from_preset(
        "gemma_2b_en",
        preprocessor=None,
    )
    gemma_lm.generate(prompt)
    ```

    Call `fit()` on a single batch.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    gemma_lm = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b_en")
    gemma_lm.fit(x=features, batch_size=2)
    ```

    Call `fit()` with LoRA fine-tuning enabled.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    gemma_lm = keras_hub.models.GemmaCausalLM.from_preset("gemma_2b_en")
    gemma.backbone.enable_lora(rank=4)
    gemma_lm.fit(x=features, batch_size=2)
    ```

    Call `fit()` without preprocessing.
    ```python
    x = {
        # Token ids for "<bos> Keras is deep learning library<eos>"
        "token_ids": np.array([[2, 214064, 603, 5271, 6044, 9581, 1, 0]] * 2),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 0]] * 2),
    }
    y = np.array([[214064, 603, 5271, 6044, 9581, 3, 0, 0]] * 2)
    sw = np.array([[1, 1, 1, 1, 1, 1, 0, 0]] * 2)

    gemma_lm = keras_hub.models.GemmaCausalLM.from_preset(
        "gemma_2b_en",
        preprocessor=None,
    )
    gemma_lm.fit(x=x, y=y, sample_weight=sw, batch_size=2)
    ```

    Custom backbone and vocabulary.
    ```python
    tokenizer = keras_hub.models.GemmaTokenizer(
        proto="proto.spm",
    )
    preprocessor = keras_hub.models.GemmaCausalLMPreprocessor(
        tokenizer=tokenizer,
        sequence_length=128,
    )
    backbone = keras_hub.models.GemmaBackbone(
        vocabulary_size=30552,
        num_layers=4,
        num_heads=4,
        hidden_dim=256,
        intermediate_dim=512,
        max_sequence_length=128,
    )
    gemma_lm = keras_hub.models.GemmaCausalLM(
        backbone=backbone,
        preprocessor=preprocessor,
    )
    gemma_lm.fit(x=features, batch_size=2)
    ```
    """

    backbone_cls = GemmaBackbone
    preprocessor_cls = GemmaCausalLMPreprocessor

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
        hidden_states = backbone(inputs)
        outputs = backbone.token_embedding(hidden_states, reverse=True)
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
    ):
        """Forward pass of `GemmaCausalLM` with cache.

        `call_with_cache` adds an additional forward pass for the model for
        autoregressive inference. Unlike calling the model directly, this method
        allows caching previous key/value Tensors in multi-head attention layer,
        and avoids recomputing the outputs of seen tokens.

        Args:
            token_ids: a dense int Tensor with shape `(batch_size, max_length)`.
            cache: a dense float Tensor, the cache of key and value.
            cache_update_index: int, or int Tensor. The index of current inputs
                in the whole sequence.

        Returns:
            A (logits, hidden_states, cache) tuple. Where `logits` is the
            language model logits for the input token_ids, `hidden_states` is
            the final hidden representation of the input tokens, and `cache` is
            the decoding cache.
        """
        x = self.backbone.token_embedding(token_ids)
        x = x * ops.cast(ops.sqrt(self.backbone.hidden_dim), x.dtype)
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

    def _build_cache(self, token_ids):
        """Build an empty cache for use with `call_with_cache()`."""
        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_key_value_heads
        head_dim = self.backbone.head_dim
        shape = [batch_size, num_layers, 2, max_length, num_heads, head_dim]
        cache = ops.zeros(shape, dtype=self.compute_dtype)
        # Seed the cache.
        _, hidden_states, cache = self.call_with_cache(token_ids, cache, 0)
        return hidden_states, cache

    def generate_step(
        self,
        inputs,
        stop_token_ids=None,
    ):
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
        token_ids, padding_mask = inputs["token_ids"], inputs["padding_mask"]
        # Create and seed cache with a single forward pass.
        hidden_states, cache = self._build_cache(token_ids)
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
            token_ids: A <int>[batch_size, num_tokens] tensor containing tokens
                to score. Typically, this tensor captures the output from a call
                to `GemmaCausalLM.generate()`, i.e., tokens for both the input
                text and the model-generated text.
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
        gemma_lm = keras_hub.models.GemmaCausalLM.from_preset(
            "gemma_2b_en"
        )
        generations = gemma_lm.generate(
            ["This is a", "Where are you"],
            max_length=30
        )
        preprocessed = gemma_lm.preprocessor.generate_preprocess(generations)
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

            losses = gemma_lm.score(
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

        token_embeddings = self.backbone.token_embedding(token_ids)
        x = layer_intercept_fn(token_embeddings, -1)

        x = token_embeddings * ops.cast(
            ops.sqrt(self.backbone.hidden_dim), dtype=self.compute_dtype
        )
        for i, transformer_layer in enumerate(self.backbone.transformer_layers):
            x = transformer_layer(x, padding_mask=padding_mask)
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
