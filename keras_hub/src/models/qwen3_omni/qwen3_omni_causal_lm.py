import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.qwen3_omni.qwen3_omni_backbone import Qwen3OmniBackbone
from keras_hub.src.models.qwen3_omni.qwen3_omni_causal_lm_preprocessor import (
    Qwen3OmniCausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export(
    "keras_hub.models.Qwen3OmniCausalLM",
)
class Qwen3OmniCausalLM(CausalLM):
    """An end-to-end Qwen3-Omni model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on plain
    text input, or to autoregressively generate plain text similar to the data
    used for training. This task can be used for pre-training or fine-tuning a
    Qwen3-Omni model, simply by calling `fit()`.

    This model has a `generate()` method, which generates text based on a
    prompt. The generation strategy used is controlled by an additional
    `sampler` argument on `compile()`. You can recompile the model with
    different `keras_hub.samplers` objects to control the generation.
    By default, `"greedy"` sampling will be used.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to string inputs during
    `fit()`, `predict()`, `evaluate()`, and `generate()`. This is done by
    default when creating the model with `from_preset()`.

    Qwen3-Omni is a multimodal extension of Qwen3 that combines Mixture of
    Experts (MoE) architecture with multimodal capabilities (audio, vision, text).
    It uses Multimodal Rotary Position Embedding (M-RoPE) to handle different
    modalities efficiently.

    Args:
        backbone: A `keras_hub.models.Qwen3OmniBackbone` instance.
        preprocessor: A `keras_hub.models.Qwen3OmniCausalLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.

    Examples:

    Use `generate()` to do text generation.
    ```python
    qwen3_omni_lm = keras_hub.models.Qwen3OmniCausalLM.from_preset(
        "qwen3_omni_0.5b_en"
    )
    qwen3_omni_lm.generate("I want to say", max_length=30)

    # Generate with batched prompts.
    qwen3_omni_lm.generate(["This is a", "Where are you"], max_length=30)
    ```

    Compile the `generate()` function with a custom sampler.
    ```python
    qwen3_omni_lm = keras_hub.models.Qwen3OmniCausalLM.from_preset(
        "qwen3_omni_0.5b_en"
    )
    qwen3_omni_lm.compile(sampler="top_k")
    qwen3_omni_lm.generate("I want to say", max_length=30)

    qwen3_omni_lm.compile(sampler=keras_hub.samplers.BeamSampler(num_beams=2))
    qwen3_omni_lm.generate("I want to say", max_length=30)
    ```

    Use `generate()` without preprocessing.
    ```python
    prompt = {
        # Token ids for "<bos> Qwen3-Omni is".
        "token_ids": np.array([[2, 12345, 678, 0, 0, 0, 0]] * 2),
        # Use `"padding_mask"` to indicate values that should not be overridden.
        "padding_mask": np.array([[1, 1, 1, 0, 0, 0, 0]] * 2),
    }

    qwen3_omni_lm = keras_hub.models.Qwen3OmniCausalLM.from_preset(
        "qwen3_omni_0.5b_en",
        preprocessor=None,
    )
    qwen3_omni_lm.generate(prompt)
    ```

    Call `fit()` on a single batch.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    qwen3_omni_lm = keras_hub.models.Qwen3OmniCausalLM.from_preset(
        "qwen3_omni_0.5b_en"
    )
    qwen3_omni_lm.fit(x=features, batch_size=2)
    ```

    Call `fit()` with LoRA fine-tuning enabled.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    qwen3_omni_lm = keras_hub.models.Qwen3OmniCausalLM.from_preset(
        "qwen3_omni_0.5b_en"
    )
    qwen3_omni_lm.backbone.enable_lora(rank=4)
    qwen3_omni_lm.fit(x=features, batch_size=2)
    ```

    Call `fit()` without preprocessing.
    ```python
    x = {
        # Token ids for "<bos> Qwen3-Omni is a language model<eos>"
        "token_ids": np.array([[2, 12345, 678, 543, 9876, 1, 0, 0]] * 2),
        "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 0, 0]] * 2),
    }
    y = np.array([[12345, 678, 543, 9876, 1, 0, 0, 0]] * 2)
    sw = np.array([[1, 1, 1, 1, 1, 0, 0, 0]] * 2)

    qwen3_omni_lm = keras_hub.models.Qwen3OmniCausalLM.from_preset(
        "qwen3_omni_0.5b_en",
        preprocessor=None,
    )
    qwen3_omni_lm.fit(x=x, y=y, sample_weight=sw, batch_size=2)
    ```

    Custom backbone and vocabulary.
    ```python
    tokenizer = keras_hub.tokenizers.Qwen3OmniTokenizer(
        vocabulary="qwen3_omni_vocab.json",
        merges="qwen3_omni_merges.txt",
    )
    preprocessor = keras_hub.models.Qwen3OmniCausalLMPreprocessor(
        tokenizer=tokenizer,
        sequence_length=128,
    )
    backbone = keras_hub.models.Qwen3OmniBackbone(
        vocabulary_size=151936,
        num_layers=28,
        num_query_heads=16,
        num_key_value_heads=2,
        hidden_dim=1536,
        intermediate_dim=8960,
        moe_intermediate_dim=2560,
        num_experts=64,
        top_k=8,
        max_sequence_length=8192,
    )
    qwen3_omni_lm = keras_hub.models.Qwen3OmniCausalLM(
        backbone=backbone,
        preprocessor=preprocessor,
    )
    qwen3_omni_lm.fit(x=features, batch_size=2)
    ```
    """

    backbone_cls = Qwen3OmniBackbone
    preprocessor_cls = Qwen3OmniCausalLMPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        self.backbone = backbone
        self.preprocessor = preprocessor

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
        """Forward pass of `Qwen3OmniCausalLM` with cache.

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
        updated_cache = []
        for i in range(self.backbone.num_layers):
            current_cache = cache[:, i, ...]
            x, next_cache = self.backbone.transformer_layers[i](
                x,
                cache=current_cache,
                cache_update_index=cache_update_index,
            )
            updated_cache.append(next_cache)
        cache = ops.stack(updated_cache, axis=1)
        hidden_states = x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, hidden_states, cache

    def _build_cache(self, token_ids):
        """Build an empty cache for use with `call_with_cache()`."""
        batch_size = ops.shape(token_ids)[0]
        max_length = ops.shape(token_ids)[1]
        num_layers = self.backbone.num_layers
        num_key_value_heads = self.backbone.num_key_value_heads
        head_dim = self.backbone.head_dim
        shape = [
            batch_size,
            num_layers,
            2,
            max_length,
            num_key_value_heads,
            head_dim,
        ]
        cache = ops.zeros(shape, dtype=self.compute_dtype)
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
            stop_token_ids: Tuple of id's of the end token to stop on. If all
                sequences have produced a new stop token, generation
                will stop.
        """
        token_ids, padding_mask = inputs["token_ids"], inputs["padding_mask"]
        hidden_states, cache = self._build_cache(token_ids)
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
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
        
        # Validate token_ids has at least 2 dimensions
        if len(ops.shape(token_ids).shape) < 2:
            raise ValueError(
                f"token_ids must have at least 2 dimensions (batch, sequence). "
                f"Received shape: {ops.shape(token_ids)}"
            )

        if padding_mask is None:
            padding_mask = ops.ones(shape=batch_shape, dtype="bool")

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
