import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
from keras_hub.src.models.t5gemma.t5gemma_backbone import T5GemmaBackbone
from keras_hub.src.models.t5gemma.t5gemma_seq_2_seq_lm_preprocessor import (
    T5GemmaSeq2SeqLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.T5GemmaSeq2SeqLM")
class T5GemmaSeq2SeqLM(Seq2SeqLM):
    """An end-to-end T5Gemma model for seq2seq language modeling.

    A seq2seq language model (LM) is an encoder-decoder model which is used for
    conditional text generation. The encoder is given a "context" text (fed to
    the encoder), and the decoder predicts the next token based on both the
    encoder inputs and the previous tokens. You can finetune `T5GemmaSeq2SeqLM`
    to generate text for any seq2seq task (e.g., translation or summarization).

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
        preprocessor: A `keras_hub.models.T5GemmaSeq2SeqLMPreprocessor` or
            `None`. If `None`, this model will not apply preprocessing, and
            inputs should be preprocessed before calling the model. Defaults
            to `None`.

    Examples:

    Use `generate()` to do text generation.
    ```python
    import numpy as np
    t5gemma_lm = keras_hub.models.T5GemmaSeq2SeqLM.from_preset(
        "t5gemma_b_b_prefixlm_it"
    )
    # Generate with encoder-only input.
    t5gemma_lm.generate("The quick brown fox jumped.", max_length=30)

    # Generate with batched encoder-only inputs.
    t5gemma_lm.generate(
        ["The quick brown fox jumped.", "The whale."],
        max_length=30
    )
    # Generate with encoder and decoder inputs.
    t5gemma_lm.generate(
        {
            "encoder_text": "The quick brown fox jumped.",
            "decoder_text": "A fast fox"
        },
        max_length=30
    )
    ```

    Compile the `generate()` function with a custom sampler.
    ```python
    t5gemma_lm = keras_hub.models.T5GemmaSeq2SeqLM.from_preset(
        "t5gemma_b_b_prefixlm_it"
    )
    t5gemma_lm.compile(sampler="top_k")
    t5gemma_lm.generate("I want to say", max_length=30)

    t5gemma_lm.compile(sampler=keras_hub.samplers.BeamSampler(num_beams=2))
    t5gemma_lm.generate("I want to say", max_length=30)
    ```

    Use `generate()` without preprocessing.
    ```python
    # Preprocessed inputs, with encoder inputs corresponding to
    # "The quick brown fox", and the decoder inputs to "A fast fox".
    # Use `"padding_mask"` to indicate values that should not be overridden.
    prompt = {
        "encoder_token_ids": np.array([[2, 10, 133, 2119, 6219, 23602, 1, 0]]),
        "encoder_padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 0]]),
        "decoder_token_ids": np.array([[2, 133, 1769, 1, 0, 0, 0]]),
        "decoder_padding_mask": np.array([[1, 1, 1, 1, 0, 0, 0]])
    }

    t5gemma_lm = keras_hub.models.T5GemmaSeq2SeqLM.from_preset(
        "t5gemma_b_b_prefixlm_it",
        preprocessor=None,
    )
    t5gemma_lm.generate(prompt)
    ```

    Call `fit()` on a single batch.
    ```python
    features = {
        "encoder_text": ["The quick fox jumped.", "I forgot my homework."],
        "decoder_text": ["The fast hazel fox leapt.", "I forgot my assignment."]
    }
    t5gemma_lm = keras_hub.models.T5GemmaSeq2SeqLM.from_preset(
        "t5gemma_b_b_prefixlm_it"
    )
    t5gemma_lm.fit(x=features, batch_size=2)
    ```

    Call `fit()` without preprocessing.
    ```python
    x = {
        "encoder_token_ids": np.array([[2, 133, 2119, 1, 0]] * 2),
        "encoder_padding_mask": np.array([[1, 1, 1, 1, 0]] * 2),
        "decoder_token_ids": np.array([[2, 133, 1769, 1, 0]] * 2),
        "decoder_padding_mask": np.array([[1, 1, 1, 1, 1]] * 2),
    }
    y = np.array([[133, 1769, 1, 0, 0]] * 2)
    sw = np.array([[1, 1, 1, 0, 0]] * 2)

    t5gemma_lm = keras_hub.models.T5GemmaSeq2SeqLM.from_preset(
        "t5gemma_b_b_prefixlm_it",
        preprocessor=None,
    )
    t5gemma_lm.fit(x=x, y=y, sample_weight=sw, batch_size=2)
    ```

    Custom backbone and vocabulary.
    ```python
    features = {
        "encoder_text": ["The quick fox jumped.", "I forgot my homework."],
        "decoder_text": ["The fast hazel fox leapt.", "I forgot my assignment."]
    }
    tokenizer = keras_hub.models.T5GemmaTokenizer(
        proto="proto.spm",
    )
    preprocessor = keras_hub.models.T5GemmaSeq2SeqLMPreprocessor(
        tokenizer=tokenizer,
        encoder_sequence_length=128,
        decoder_sequence_length=128,
    )
    backbone = keras_hub.models.T5GemmaBackbone(
        vocabulary_size=32000,
        # Encoder parameters.
        encoder_hidden_dim=256,
        encoder_intermediate_dim=512,
        encoder_num_layers=4,
        encoder_num_attention_heads=4,
        encoder_num_key_value_heads=2,
        encoder_head_dim=64,
        encoder_layer_types=["full_attention"] * 4,
        # Decoder parameters.
        decoder_hidden_dim=256,
        decoder_intermediate_dim=512,
        decoder_num_layers=4,
        decoder_num_attention_heads=4,
        decoder_num_key_value_heads=2,
        decoder_head_dim=64,
        decoder_layer_types=["full_attention"] * 4,
        # Common parameters.
        dropout_rate=0.1,
        rms_norm_eps=1e-6,
        query_pre_attn_scalar=1.0,
        attention_bias=False,
        hidden_activation="gelu_approximate",
    )
    t5gemma_lm = keras_hub.models.T5GemmaSeq2SeqLM(
        backbone=backbone,
        preprocessor=preprocessor,
    )
    t5gemma_lm.fit(x=features, batch_size=2)
    ```
    """

    backbone_cls = T5GemmaBackbone
    preprocessor_cls = T5GemmaSeq2SeqLMPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
        # === Layers ===
        self.backbone = backbone
        self.preprocessor = preprocessor

        # === Functional Model ===
        # This must be "backbone.input" i.e. the full input structure,
        # rather than "backbone.inputs" which is the flattened list of inputs.
        inputs = backbone.input
        sequence_output = backbone(inputs)["decoder_sequence_output"]
        logits = backbone.decoder_token_embedding(sequence_output, reverse=True)
        if self.backbone.final_logit_softcapping is not None:
            logits = logits / self.backbone.final_logit_softcapping
            logits = keras.ops.tanh(logits)
            logits = logits * self.backbone.final_logit_softcapping
        super().__init__(
            inputs=inputs,
            outputs=logits,
            **kwargs,
        )

    def call_encoder(self, token_ids, padding_mask):
        """Process inputs through the encoder stack."""
        encoder_embeddings = self.backbone.token_embedding(token_ids)
        encoder_embeddings *= keras.ops.cast(
            keras.ops.sqrt(self.backbone.encoder_hidden_dim),
            encoder_embeddings.dtype,
        )
        encoder_hidden_states = self.backbone.encoder_dropout(
            encoder_embeddings, training=False
        )
        for layer in self.backbone.encoder_layers:
            encoder_hidden_states = layer(
                encoder_hidden_states, padding_mask=padding_mask, training=False
            )
        encoder_output = self.backbone.encoder_norm(encoder_hidden_states)
        encoder_output = self.backbone.encoder_dropout(
            encoder_output, training=False
        )
        return encoder_output, padding_mask

    def call_decoder_with_cache(
        self,
        decoder_token_ids,
        decoder_padding_mask,
        cache,
        cache_update_index,
        encoder_output,
        encoder_padding_mask,
    ):
        """Forward pass of `T5GemmaSeq2SeqLM`'s decoder with cache.

        `call_decoder_with_cache` adds an additional forward pass for the model
        for autoregressive inference. Unlike calling the model directly, this
        method allows caching previous key/value Tensors in the attention
        layers, and avoids recomputing the outputs of seen tokens.

        Args:
            decoder_token_ids: A dense int Tensor with shape
                `(batch_size, max_length)`. The token ids for the decoder.
            decoder_padding_mask: A dense int Tensor with shape `(batch_size,
                max_length)`. The padding mask for the decoder.
            cache: A dense float Tensor, the cache of key and value states.
            cache_update_index: int, or int Tensor. The index of the current
                token being processed in the whole sequence.
            encoder_output: A dense float Tensor. The output of the encoder.
            encoder_padding_mask: A dense int Tensor. The padding mask for
                the encoder output.

        Returns:
            A `(logits, hidden_states, cache)` tuple. Where `logits` is the
            language model logits for the input token_ids, `hidden_states` is
            the final hidden representation of the input tokens, and `cache` is
            the updated decoding cache.
        """
        self_attention_cache, cross_attention_cache = cache
        hidden_states = self.backbone.decoder_token_embedding(decoder_token_ids)
        hidden_states *= keras.ops.cast(
            keras.ops.sqrt(self.backbone.decoder_hidden_dim),
            hidden_states.dtype,
        )
        hidden_states = self.backbone.decoder_dropout(
            hidden_states, training=False
        )
        # Every decoder layer has a separate cache for the self-attention layer
        # and the cross-attention layer. We update all of them separately.
        updated_self_attention_caches = []
        updated_cross_attention_caches = []
        for i, layer in enumerate(self.backbone.decoder_layers):
            layer_self_cache = (
                self_attention_cache[:, i, ...]
                if self_attention_cache is not None
                else None
            )
            layer_cross_cache = (
                cross_attention_cache[:, i, ...]
                if cross_attention_cache is not None
                else None
            )
            layer_cache = (layer_self_cache, layer_cross_cache)
            hidden_states, updated_layer_cache = layer(
                (hidden_states, encoder_output),
                self_attention_padding_mask=decoder_padding_mask,
                cross_attention_padding_mask=encoder_padding_mask,
                cache=layer_cache,
                cache_update_index=cache_update_index,
                training=False,
            )
            new_self_cache, new_cross_cache = updated_layer_cache
            updated_self_attention_caches.append(new_self_cache)
            updated_cross_attention_caches.append(new_cross_cache)
        self_attention_cache = keras.ops.stack(
            updated_self_attention_caches, axis=1
        )
        cross_attention_cache = keras.ops.stack(
            updated_cross_attention_caches, axis=1
        )
        hidden_states = self.backbone.decoder_norm(hidden_states)
        logits = self.backbone.decoder_token_embedding(
            hidden_states, reverse=True
        )
        if self.backbone.final_logit_softcapping is not None:
            logits = logits / self.backbone.final_logit_softcapping
            logits = keras.ops.tanh(logits)
            logits = logits * self.backbone.final_logit_softcapping
        return (
            logits,
            hidden_states,
            (self_attention_cache, cross_attention_cache),
        )

    def _build_cache(
        self,
        encoder_token_ids,
        encoder_padding_mask,
        decoder_token_ids,
        decoder_padding_mask,
    ):
        """Build an empty cache for use with `call_with_cache()`."""
        encoder_output, encoder_padding_mask = self.call_encoder(
            encoder_token_ids, encoder_padding_mask
        )
        batch_size = keras.ops.shape(decoder_token_ids)[0]
        num_layers = self.backbone.decoder_num_layers
        num_kv_heads = self.backbone.decoder_num_key_value_heads
        head_dim = self.backbone.decoder_head_dim
        self_cache_shape = (
            batch_size,
            num_layers,
            2,
            keras.ops.shape(decoder_token_ids)[1],
            num_kv_heads,
            head_dim,
        )
        self_attention_cache = keras.ops.zeros(
            self_cache_shape, dtype=self.compute_dtype
        )
        cross_attention_cache = None
        _, hidden_states, cache = self.call_decoder_with_cache(
            decoder_token_ids=decoder_token_ids,
            decoder_padding_mask=decoder_padding_mask,
            cache=(self_attention_cache, cross_attention_cache),
            cache_update_index=0,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask,
        )
        extra_cache_info = (encoder_output, encoder_padding_mask)
        return hidden_states, cache, extra_cache_info

    def generate_step(self, inputs, stop_token_ids=None):
        """A compilable generation function for a single batch of inputs.

        This function represents the inner, XLA-compilable, generation function
        for a single batch of inputs. Inputs should have the same structure as
        model inputs, a dictionary with keys `"token_ids"` and `"padding_mask"`.
        `"encoder_token_ids"`, `"encoder_padding_mask"`, `"decoder_token_ids"`
        and `"decoder_padding_mask"`.

        Args:
            inputs: A dictionary with four keys - `"encoder_token_ids"`,
                `"encoder_padding_mask"`, `"decoder_token_ids"` and
                `"decoder_padding_mask"`, with batched tensor values.
            stop_token_ids: Tuple of id's of end token's to stop on. If all
                sequences have produced a new stop token, generation
                will stop.
        """
        encoder_token_ids = inputs["encoder_token_ids"]
        encoder_padding_mask = inputs["encoder_padding_mask"]
        decoder_token_ids = inputs["decoder_token_ids"]
        decoder_padding_mask = inputs["decoder_padding_mask"]
        # Create and seed cache with a single forward pass.
        hidden_states, cache, extra_cache_info = self._build_cache(
            encoder_token_ids=encoder_token_ids,
            encoder_padding_mask=encoder_padding_mask,
            decoder_token_ids=decoder_token_ids,
            decoder_padding_mask=decoder_padding_mask,
        )
        encoder_output, encoder_padding_mask = extra_cache_info
        # Compute the lengths of all user inputted tokens ids.
        row_lengths = keras.ops.sum(
            keras.ops.cast(decoder_padding_mask, "int32"), axis=-1
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
            (
                logits,
                _,
                updated_cache,
            ) = self.call_decoder_with_cache(
                decoder_token_ids=prompt,
                decoder_padding_mask=None,
                cache_update_index=cache_update_index,
                cache=cache,
                encoder_output=encoder_output,
                encoder_padding_mask=encoder_padding_mask,
            )
            return keras.ops.squeeze(logits, axis=1), None, updated_cache

        decoder_token_ids = self.sampler(
            next=next,
            prompt=decoder_token_ids,
            cache=cache,
            index=index,
            mask=decoder_padding_mask,
            stop_token_ids=stop_token_ids,
            hidden_states=hidden_states,
            model=self,
        )

        # Compute an output padding mask with the token ids we updated.
        if stop_token_ids is not None:
            # Build a mask of `stop_token_ids` locations not in the original
            # prompt (not in locations where `decoder_padding_mask` is True).
            end_locations = any_equal(
                decoder_token_ids,
                stop_token_ids,
                keras.ops.logical_not(decoder_padding_mask),
            )
            # Use cumsum to get ones in all locations after end_locations.
            end_locations = keras.ops.cast(end_locations, "int32")
            cumsum = keras.ops.cast(
                keras.ops.cumsum(end_locations, axis=-1), "int32"
            )
            overflow = cumsum - end_locations
            # Our padding mask is the inverse of these overflow locations.
            decoder_padding_mask = keras.ops.logical_not(
                keras.ops.cast(overflow, "bool")
            )
        else:
            # Without early stopping, all locations will have been updated.
            decoder_padding_mask = keras.ops.ones_like(
                decoder_token_ids, dtype="bool"
            )

        return {
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }
