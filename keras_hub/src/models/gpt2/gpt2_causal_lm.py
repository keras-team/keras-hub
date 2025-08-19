import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_hub.src.models.gpt2.gpt2_causal_lm_preprocessor import (
    GPT2CausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.GPT2CausalLM")
class GPT2CausalLM(CausalLM):
    """An end-to-end GPT2 model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on
    plain text input, or to autoregressively generate plain text similar to
    the data used for training. This task can be used for pre-training or
    fine-tuning a GPT-2 model, simply by calling `fit()`.

    This model has a `generate()` method, which generates text based on a
    prompt. The generation strategy used is controlled by an additional
    `sampler` argument on `compile()`. You can recompile the model with
    different `keras_hub.samplers` objects to control the generation. By
    default, `"top_k"` sampling will be used.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to string inputs during
    `fit()`, `predict()`, `evaluate()` and `generate()`. This is done by default
    when creating the model with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/openai/gpt-2).

    Args:
        backbone: A `keras_hub.models.GPT2Backbone` instance.
        preprocessor: A `keras_hub.models.GPT2CausalLMPreprocessor` or `None`.
            If `None`, this model will not apply preprocessing, and inputs
            should be preprocessed before calling the model.

    Examples:

    Use `generate()` to do text generation.
    ```python
    gpt2_lm = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")
    gpt2_lm.generate("I want to say", max_length=30)

    # Generate with batched prompts.
    gpt2_lm.generate(["This is a", "Where are you"], max_length=30)
    ```

    Compile the `generate()` function with a custom sampler.
    ```python
    gpt2_lm = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")
    gpt2_lm.compile(sampler="greedy")
    gpt2_lm.generate("I want to say", max_length=30)

    gpt2_lm.compile(sampler=keras_hub.samplers.BeamSampler(num_beams=2))
    gpt2_lm.generate("I want to say", max_length=30)
    ```

    Use `generate()` without preprocessing.
    ```python
    # Prompt the model with `5338, 318` (the token ids for `"Who is"`).
    # Use `"padding_mask"` to indicate values that should not be overridden.
    prompt = {
        "token_ids": np.array([[5338, 318, 0, 0, 0]] * 2),
        "padding_mask": np.array([[1, 1, 0, 0, 0]] * 2),
    }

    gpt2_lm = keras_hub.models.GPT2CausalLM.from_preset(
        "gpt2_base_en",
        preprocessor=None,
    )
    gpt2_lm.generate(prompt)
    ```

    Call `fit()` on a single batch.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    gpt2_lm = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")
    gpt2_lm.fit(x=features, batch_size=2)
    ```

    Call `fit()` without preprocessing.
    ```python
    x = {
        "token_ids": np.array([[50256, 1, 2, 3, 4]] * 2),
        "padding_mask": np.array([[1, 1, 1, 1, 1]] * 2),
    }
    y = np.array([[1, 2, 3, 4, 50256]] * 2)
    sw = np.array([[1, 1, 1, 1, 1]] * 2)

    gpt2_lm = keras_hub.models.GPT2CausalLM.from_preset(
        "gpt2_base_en",
        preprocessor=None,
    )
    gpt2_lm.fit(x=x, y=y, sample_weight=sw, batch_size=2)
    ```

    Custom backbone and vocabulary.
    ```python
    features = ["a quick fox.", "a fox quick."]
    vocab = {"<|endoftext|>": 0, "a": 4, "Ġquick": 5, "Ġfox": 6}
    merges = ["Ġ q", "u i", "c k", "ui ck", "Ġq uick"]
    merges += ["Ġ f", "o x", "Ġf ox"]

    tokenizer = keras_hub.models.GPT2Tokenizer(
        vocabulary=vocab,
        merges=merges,
    )
    preprocessor = keras_hub.models.GPT2CausalLMPreprocessor(
        tokenizer=tokenizer,
        sequence_length=128,
    )
    backbone = keras_hub.models.GPT2Backbone(
        vocabulary_size=30552,
        num_layers=4,
        num_heads=4,
        hidden_dim=256,
        intermediate_dim=512,
        max_sequence_length=128,
    )
    gpt2_lm = keras_hub.models.GPT2CausalLM(
        backbone=backbone,
        preprocessor=preprocessor,
    )
    gpt2_lm.fit(x=features, batch_size=2)
    ```
    """

    backbone_cls = GPT2Backbone
    preprocessor_cls = GPT2CausalLMPreprocessor

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

    def call_with_cache(
        self,
        token_ids,
        cache,
        cache_update_index,
    ):
        """Forward pass of `GPT2CausalLM` with cache.

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
        tokens = self.backbone.token_embedding(token_ids)
        positions = self.backbone.position_embedding(
            tokens, start_index=cache_update_index
        )
        x = self.backbone.embeddings_add((tokens, positions))
        x = self.backbone.embeddings_dropout(x)
        # Each decoder layer has a cache; we update them separately.
        caches = []
        for i, transformer_layer in enumerate(self.backbone.transformer_layers):
            current_cache = cache[:, i, ...]
            x, next_cache = transformer_layer(
                x,
                self_attention_cache=current_cache,
                self_attention_cache_update_index=cache_update_index,
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
        num_heads = self.backbone.num_heads
        head_dim = self.backbone.hidden_dim // self.backbone.num_heads
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
            stop_token_ids: List of id's of end token's to stop on. If all
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
            # Build a mask of stop tokens locations not in the original
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
                to `GPT2CausalLM.generate()`, i.e., tokens for both the input
                text and the model-generated text.
            padding_mask: A <bool>[batch_size, num_tokens] tensor indicating the
                tokens that should be preserved during generation. This is an
                artifact required by the `GPT2Backbone` and isn't influential on
                the computation of this function. If omitted, this function uses
                `keras.ops.ones()` to create a tensor of the appropriate shape.
            scoring_mode: The type of scores to return, either "logits" or
                "loss", both will be per input token.
            layer_intercept_fn: An optional function for augmenting activations
                with additional computation, for example, as part of
                interpretability research. This function will be passed the
                activations as its first parameter and a numeric index
                associated with that backbone layer. This index is not an index
                into `self.backbone.layers`. The index -1 accompanies the
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
        gpt2_lm = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")
        generations = gpt2_lm.generate(
            ["This is a", "Where are you"],
            max_length=30
        )
        preprocessed = gpt2_lm.preprocessor.generate_preprocess(generations)
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

            losses = gpt2_lm.score(
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
        position_embeddings = self.backbone.position_embedding(token_embeddings)
        summed_embeddings = self.backbone.embeddings_add(
            (token_embeddings, position_embeddings)
        )
        x = layer_intercept_fn(summed_embeddings, -1)
        x = self.backbone.embeddings_dropout(x)

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
