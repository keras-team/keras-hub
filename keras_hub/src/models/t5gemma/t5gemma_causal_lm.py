import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm import CausalLM
from keras_hub.src.models.t5gemma.t5gemma_backbone import T5GemmaBackbone
from keras_hub.src.models.t5gemma.t5gemma_causal_lm_preprocessor import (
    T5GemmaCausalLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.T5GemmaCausalLM")
class T5GemmaCausalLM(CausalLM):
    """An end-to-end T5Gemma model for causal language modeling.

    A causal language model (LM) predicts the next token based on previous
    tokens. This task setup can be used to train the model unsupervised on
    plain text input, or to autoregressively generate plain text similar to
    the data used for training. This task can be used for pre-training or
    fine-tuning a T5Gemma model, simply by calling `fit()`.

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
        backbone: A `keras_hub.models.T5GemmaBackbone` instance.
        preprocessor: A `keras_hub.models.T5GemmaCausalLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model.

    Examples:

    Use `generate()` to do text generation.
    ```python
    t5gemma_lm = keras_hub.models.T5GemmaCausalLM.from_preset(
        "t5gemma_b_b_prefixlm_it"
    )
    t5gemma_lm.generate("I want to say", max_length=30)

    # Generate with batched prompts.
    t5gemma_lm.generate(["This is a", "Where are you"], max_length=30)
    ```

    Compile the `generate()` function with a custom sampler.
    ```python
    t5gemma_lm = keras_hub.models.T5GemmaCausalLM.from_preset(
        "t5gemma_b_b_prefixlm_it"
    )
    t5gemma_lm.compile(sampler="top_k")
    t5gemma_lm.generate("I want to say", max_length=30)

    t5gemma_lm.compile(sampler=keras_hub.samplers.BeamSampler(num_beams=2))
    t5gemma_lm.generate("I want to say", max_length=30)
    ```

    Use `generate()` without preprocessing.
    ```python
    # The preprocessor is responsible for creating a dictionary of tensors.
    # If you are not using a preprocessor, you must format your inputs
    # yourself.
    prompt = {
        # Token ids for "<bos> Keras is".
        "token_ids": np.array([[2, 214064, 603, 0, 0, 0, 0]] * 2),
        # Use `"padding_mask"` to indicate values that should not be overridden.
        "padding_mask": np.array([[1, 1, 1, 0, 0, 0, 0]] * 2),
    }

    t5gemma_lm = keras_hub.models.T5GemmaCausalLM.from_preset(
        "t5gemma_b_b_prefixlm_it",
        preprocessor=None,
    )
    t5gemma_lm.generate(prompt)
    ```

    Call `fit()` on a single batch.
    ```python
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    t5gemma_lm = keras_hub.models.T5GemmaCausalLM.from_preset(
        "t5gemma_b_b_prefixlm_it"
    )
    t5gemma_lm.fit(x=features, batch_size=2)
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

    t5gemma_lm = keras_hub.models.T5GemmaCausalLM.from_preset(
        "t5gemma_b_b_prefixlm_it",
        preprocessor=None,
    )
    t5gemma_lm.fit(x=x, y=y, sample_weight=sw, batch_size=2)
    ```

    Custom backbone and vocabulary.
    ```python
    tokenizer = keras_hub.models.T5GemmaTokenizer(
        proto="proto.spm",
    )
    preprocessor = keras_hub.models.T5GemmaCausalLMPreprocessor(
        tokenizer=tokenizer,
        sequence_length=128,
    )
    backbone = keras_hub.models.T5GemmaBackbone(
        vocabulary_size=32000,
        num_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_dim=256,
        intermediate_dim=512,
        dropout_rate=0.1,
        rms_norm_eps=1e-6,
        query_pre_attn_scalar=1.0,
        attention_bias=False,
        hidden_activation="gelu_approximate",
        layer_types=["full_attention"] * 4
    )
    t5gemma_lm = keras_hub.models.T5GemmaCausalLM(
        backbone=backbone,
        preprocessor=preprocessor,
    )
    t5gemma_lm.fit(x=features, batch_size=2)
    ```
    """

    backbone_cls = T5GemmaBackbone
    preprocessor_cls = T5GemmaCausalLMPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        # This must be "backbone.input" i.e. the full input structure,
        # rather than "backbone.inputs" which is the flattened list of inputs.
        inputs = backbone.input
        sequence_output = backbone(inputs)
        outputs = backbone.token_embedding(sequence_output, reverse=True)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    def call_with_cache(
        self, token_ids, padding_mask, cache, cache_update_index
    ):
        """Forward pass of `T5GemmaCausalLM` with cache.

        `call_with_cache` adds an additional forward pass for the model for
        autoregressive inference. Unlike calling the model directly, this method
        allows caching previous key/value Tensors in the attention layers,
        and avoids recomputing the outputs of seen tokens.

        Args:
            token_ids: A dense int Tensor with shape `(batch_size, max_length)`.
            padding_mask: A dense int Tensor with shape `(batch_size,
            max_length)`.
            cache: A dense float Tensor, the cache of key and value states.
            cache_update_index: int, or int Tensor. The index of the current
                token being processed in the whole sequence.

        Returns:
            A `(logits, hidden_states, cache)` tuple. Where `logits` is the
            language model logits for the input token_ids, `hidden_states` is
            the final hidden representation of the input tokens, and `cache` is
            the updated decoding cache.
        """
        (
            encoder_output,
            past_key_values,
            encoder_padding_mask,
        ) = cache
        hidden_states = self.backbone.token_embedding(token_ids)
        hidden_states *= keras.ops.cast(
            keras.ops.sqrt(self.backbone.hidden_dim), hidden_states.dtype
        )
        hidden_states = self.backbone.decoder_dropout(hidden_states)
        updated_key_values = []
        for i, layer in enumerate(self.backbone.decoder_layers):
            current_cache = past_key_values[:, i, ...]
            hidden_states, current_cache = layer(
                (hidden_states, encoder_output),
                self_attention_padding_mask=padding_mask,
                cross_attention_padding_mask=encoder_padding_mask,
                self_attention_cache=current_cache,
                cache_update_index=cache_update_index,
            )
            updated_key_values.append(current_cache)
        past_key_values = keras.ops.stack(updated_key_values, axis=1)
        hidden_states = self.backbone.decoder_norm(hidden_states)
        logits = self.backbone.token_embedding(hidden_states, reverse=True)
        cache = (
            encoder_output,
            past_key_values,
            encoder_padding_mask,
        )
        return logits, hidden_states, cache

    def _build_cache(self, token_ids, padding_mask):
        """Build an empty cache for use with `call_with_cache()`."""
        # Encoder.
        encoder_embeddings = self.backbone.token_embedding(token_ids)
        encoder_embeddings *= keras.ops.cast(
            keras.ops.sqrt(self.backbone.hidden_dim), encoder_embeddings.dtype
        )
        encoder_hidden_states = self.backbone.encoder_dropout(
            encoder_embeddings
        )
        for layer in self.backbone.encoder_layers:
            encoder_hidden_states = layer(
                encoder_hidden_states,
                padding_mask=padding_mask,
            )
        encoder_output = self.backbone.encoder_norm(encoder_hidden_states)
        hidden_states = self.backbone.token_embedding(token_ids)
        hidden_states *= keras.ops.cast(
            keras.ops.sqrt(self.backbone.hidden_dim), hidden_states.dtype
        )
        hidden_states = self.backbone.decoder_dropout(hidden_states)
        past_key_values = []
        for layer in self.backbone.decoder_layers:
            hidden_states, kv_cache_for_layer = layer(
                (hidden_states, encoder_output),
                self_attention_padding_mask=padding_mask,
                cross_attention_padding_mask=padding_mask,
            )
            past_key_values.append(kv_cache_for_layer)
        past_key_values = keras.ops.stack(past_key_values, axis=1)
        hidden_states = self.backbone.decoder_norm(hidden_states)
        cache = (
            encoder_output,
            past_key_values,
            padding_mask,
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
        # Create and seed cache with a single forward pass.
        hidden_states, cache = self._build_cache(
            token_ids=inputs["token_ids"],
            padding_mask=inputs["padding_mask"],
        )
        token_ids = inputs["token_ids"]
        padding_mask = inputs["padding_mask"]
        # Compute the lengths of all user inputted tokens ids.
        row_lengths = keras.ops.sum(
            keras.ops.cast(padding_mask, "int32"), axis=-1
        )
        # Start at the first index that has no user inputted id.
        index = keras.ops.min(row_lengths)

        def next(prompt, cache, index):
            # The cache index is the index of our previous token.
            cache_update_index = index - 1
            batch_size = keras.ops.shape(prompt)[0]
            prompt = keras.ops.slice(
                prompt, [0, cache_update_index], [batch_size, 1]
            )
            prompt_padding_mask = keras.ops.ones_like(prompt, dtype="int32")
            logits, _, cache = self.call_with_cache(
                prompt,
                prompt_padding_mask,
                cache,
                cache_update_index,
            )
            return keras.ops.squeeze(logits, axis=1), None, cache

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
                token_ids,
                stop_token_ids,
                keras.ops.logical_not(padding_mask),
            )
            # Use cumsum to get ones in all locations after end_locations.
            end_locations = keras.ops.cast(end_locations, "int32")
            cumsum = keras.ops.cast(
                keras.ops.cumsum(end_locations, axis=-1), "int32"
            )
            overflow = cumsum - end_locations
            # Our padding mask is the inverse of these overflow locations.
            padding_mask = keras.ops.logical_not(
                keras.ops.cast(overflow, "bool")
            )
        else:
            # Without early stopping, all locations will have been updated.
            padding_mask = keras.ops.ones_like(token_ids, dtype="bool")

        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
