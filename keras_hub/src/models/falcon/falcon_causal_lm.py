from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.falcon.falcon_backbone import FalconBackbone
from keras_hub.src.models.falcon.falcon_causal_lm_preprocessor import (
    FalconCausalLMPreprocessor,
)


@keras_hub_export("keras_hub.models.FalconCausalLM")
class FalconCausalLM(CausalLM):
    """An end-to-end Falcon model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on
    plain text input, or to autoregressively generate plain text similar to
    the data used for training. This task can be used for pre-training or
    fine-tuning a Falcon model, simply by calling `fit()`.

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
        backbone: A `keras_hub.models.FalconBackbone` instance.
        preprocessor: A `keras_hub.models.FalconCausalLMPreprocessor` or `None`.
            If `None`, this model will not apply preprocessing, and inputs
            should be preprocessed before calling the model.

    Examples:

    Use `generate()` to do text generation.
    ```python
    falcon_lm = keras_hub.models.FalconCausalLM.from_preset(
        "falcon_refinedweb_1b_en"
    )
    falcon_lm.generate("I want to say", max_length=30)

    # Generate with batched prompts.
    falcon_lm.generate(["This is a", "Where are you"], max_length=30)
    ```

    Compile the `generate()` function with a custom sampler.
    ```python
    falcon_lm = keras_hub.models.FalconCausalLM.from_preset(
        "falcon_refinedweb_1b_en"
    )
    falcon_lm.compile(sampler="top_k")
    falcon_lm.generate("I want to say", max_length=30)

    falcon_lm.compile(sampler=keras_hub.samplers.BeamSampler(num_beams=2))
    falcon_lm.generate("I want to say", max_length=30)
    ```

    Use `generate()` without preprocessing.
    ```python
    prompt = {
        # Token ids for
        # "<|endoftext|> Keras is".
        "token_ids": np.array([[50256, 17337,   292,   318]] * 2),
        # Use `"padding_mask"` to indicate values that should not be overridden.
        "padding_mask": np.array([[1, 1, 1, 1]] * 2),
    }

    falcon_lm = keras_hub.models.FalconCausalLM.from_preset(
        "falcon_refinedweb_1b_en",
        preprocessor=None,
    )
    falcon_lm.generate(prompt)
    ```

    Call `fit()` on a single batch.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    falcon_lm = keras_hub.models.FalconCausalLM.from_preset(
        "falcon_refinedweb_1b_en"
    )
    falcon_lm.fit(x=features, batch_size=2)
    ```

    Call `fit()` without preprocessing.
    ```python
    x = {
        # Token ids for
        # "<|endoftext|> Keras is deep learning library<|endoftext|>"
        "token_ids": np.array(
            [[50256, 17337, 292, 318, 2769,4673,5888, 50256, 0]] * 2
        ),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0]] * 2),
    }
    y = np.array([[17337,   292,   318,  2769,  4673,  5888, 50256, 0, 0]] * 2)
    sw = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2)

    falcon_lm = keras_hub.models.FalconCausalLM.from_preset(
        "falcon_refinedweb_1b_en",
        preprocessor=None,
    )
    falcon_lm.fit(x=x, y=y, sample_weight=sw, batch_size=2)
    ```

    Custom backbone and vocabulary.
    ```python
    vocab = {"<|endoftext|>": 0, "a": 4, "Ġquick": 5, "Ġfox": 6}
    merges = ["Ġ q", "u i", "c k", "ui ck", "Ġq uick"]
    merges += ["Ġ f", "o x", "Ġf ox"]
    tokenizer = keras_hub.models.FalconTokenizer(
        vocabulary=vocab,
        merges=merges,
    )
    preprocessor = keras_hub.models.FalconCausalLMPreprocessor(
        tokenizer=tokenizer,
        sequence_length=128,
    )
    backbone = keras_hub.models.FalconBackbone(
        vocabulary_size=50304,
        num_layers=24,
        num_attention_heads=64,
        hidden_dim=2048,
        intermediate_dim=4*2048,
    )
    falcon_lm = keras_hub.models.FalconCausalLM(
        backbone=backbone,
        preprocessor=preprocessor,
    )
    falcon_lm.fit(x=features, batch_size=2)
    ```
    """

    backbone_cls = FalconBackbone
    preprocessor_cls = FalconCausalLMPreprocessor

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
        """Forward pass of `FalconCausalLM` with cache.

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
        # Each decoder layer has a cache; we update them separately.
        caches = []
        for i, transformer_layer in enumerate(self.backbone.transformer_layers):
            current_cache = cache[:, i, ...]
            x, next_cache = transformer_layer(
                x,
                attention_cache=current_cache,
                attention_cache_update_index=cache_update_index,
            )
            caches.append(next_cache)
        cache = ops.stack(caches, axis=1)
        hidden_states = x = self.backbone.final_layernorm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, hidden_states, cache

    def _build_cache(self, token_ids):
        """Build an empty cache for use with `call_with_cache()`."""
        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_attention_heads
        head_dim = self.backbone.hidden_dim // self.backbone.num_attention_heads
        shape = [batch_size, num_layers, 2, max_length, num_heads, head_dim]
        cache = ops.zeros(shape, dtype=self.compute_dtype)
        # Seed the cache.
        _, hidden_states, cache = self.call_with_cache(token_ids, cache, 0)
        return hidden_states, cache
