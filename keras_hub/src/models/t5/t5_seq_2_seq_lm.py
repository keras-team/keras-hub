from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
from keras_hub.src.models.t5.t5_backbone import T5Backbone
from keras_hub.src.models.t5.t5_seq_2_seq_lm_preprocessor import (
    T5Seq2SeqLMPreprocessor,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.T5Seq2SeqLM")
class T5Seq2SeqLM(Seq2SeqLM):
    """An end-to-end T5 model for seq2seq language modeling.

    A seq2seq language model (LM) is an encoder-decoder model which is used for
    conditional text generation. The encoder is given a "context" text (fed to
    the encoder), and the decoder predicts the next token based on both the
    encoder inputs and the previous tokens. You can finetune `T5Seq2SeqLM` to
    generate text for any seq2seq task (e.g., translation or summarization).

    T5 is trained with a task prefix on the encoder input, e.g.
    `"translate English to German: ..."` or `"summarize: ..."`, and the
    pretrained checkpoints will follow whichever prefixes they were trained on.

    This model has a `generate()` method, which generates text based on
    encoder inputs and an optional prompt for the decoder. The generation
    strategy used is controlled by an additional `sampler` argument passed to
    `compile()`. You can recompile the model with different `keras_hub.samplers`
    objects to control the generation. By default, `"top_k"` sampling will be
    used.

    This model can optionally be configured with a `preprocessor` layer, in
    which case it will automatically apply preprocessing to string inputs during
    `fit()`, `predict()`, `evaluate()` and `generate()`. This is done by default
    when creating the model with `from_preset()`.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind.

    Args:
        backbone: A `keras_hub.models.T5Backbone` instance.
        preprocessor: A `keras_hub.models.T5Seq2SeqLMPreprocessor` or `None`.
            If `None`, this model will not apply preprocessing, and inputs
            should be preprocessed before calling the model.

    Examples:

    Use `generate()` to do text generation, given an input context.
    ```python
    t5_lm = keras_hub.models.T5Seq2SeqLM.from_preset("t5_small_multi")
    t5_lm.generate("translate English to German: The quick brown fox")

    # Generate with batched inputs.
    t5_lm.generate(
        [
            "translate English to German: The quick brown fox",
            "summarize: The quick brown fox jumped over the lazy dog.",
        ]
    )
    ```

    Compile the `generate()` function with a custom sampler.
    ```python
    t5_lm = keras_hub.models.T5Seq2SeqLM.from_preset("t5_small_multi")
    t5_lm.compile(sampler="greedy")
    t5_lm.generate("summarize: The quick brown fox")
    ```

    Use `generate()` with encoder inputs and an incomplete decoder input
    (prompt).
    ```python
    t5_lm = keras_hub.models.T5Seq2SeqLM.from_preset("t5_small_multi")
    t5_lm.generate(
        {
            "encoder_text": "translate English to German: The quick brown fox",
            "decoder_text": "Der schnelle",
        }
    )
    ```

    Call `fit()` on a single batch.
    ```python
    features = {
        "encoder_text": ["summarize: The quick fox jumped.", "summarize: Hi."],
        "decoder_text": ["The fast fox leapt.", "Hello."],
    }
    t5_lm = keras_hub.models.T5Seq2SeqLM.from_preset("t5_small_multi")
    t5_lm.fit(x=features, batch_size=2)
    ```

    Call `fit()` without preprocessing.
    ```python
    x = {
        "encoder_token_ids": np.array([[133, 2119, 1]] * 2),
        "encoder_padding_mask": np.array([[1, 1, 1]] * 2),
        "decoder_token_ids": np.array([[0, 133, 1769]] * 2),
        "decoder_padding_mask": np.array([[1, 1, 1]] * 2),
    }
    y = np.array([[133, 1769, 1]] * 2)
    sw = np.array([[1, 1, 1]] * 2)

    t5_lm = keras_hub.models.T5Seq2SeqLM.from_preset(
        "t5_small_multi",
        preprocessor=None,
    )
    t5_lm.fit(x=x, y=y, sample_weight=sw, batch_size=2)
    ```
    """

    backbone_cls = T5Backbone
    preprocessor_cls = T5Seq2SeqLMPreprocessor

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
        hidden_states = backbone(inputs)["decoder_sequence_output"]
        outputs = self._logits(hidden_states)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    def _logits(self, hidden_states):
        """Project decoder hidden states onto the vocabulary."""
        # T5 rescales the decoder output before the language model head when
        # the head shares weights with the token embedding.
        if self.backbone.tie_embedding_weights:
            hidden_states = ops.multiply(
                hidden_states,
                ops.cast(self.backbone.hidden_dim**-0.5, hidden_states.dtype),
            )
        return self.backbone.token_embedding(hidden_states, reverse=True)

    def call_encoder(self, token_ids, padding_mask):
        """Does a forward pass on the encoder and returns the encoder output."""
        x = self.backbone.token_embedding(token_ids)
        x = self.backbone.encoder_embedding_dropout(x)
        attention_mask = padding_mask[:, None, :]
        position_bias = None
        for transformer_layer in self.backbone.encoder_transformer_layers:
            x, position_bias = transformer_layer(
                x,
                attention_mask=attention_mask,
                position_bias=position_bias,
                use_causal_mask=False,
            )
        x = self.backbone.encoder_layer_norm(x)
        x = self.backbone.encoder_dropout(x)
        return x

    def call_decoder_with_cache(
        self,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_token_ids,
        self_attention_cache,
        self_attention_cache_update_index,
        cross_attention_cache,
        cross_attention_cache_update_index,
    ):
        """Forward pass with key/value caches for generative decoding.

        `call_decoder_with_cache` adds an additional inference-time forward pass
        for the model for seq2seq text generation. Unlike calling the model
        directly, this method does two things to optimize text generation:

        - Allows caching previous key/value tensors in the decoder's
          self-attention layer to avoid recomputing the outputs of seen tokens.
        - Allows caching key/value tensors in the decoder's cross-attention
          layer to avoid recomputing the encoder outputs.

        Args:
            encoder_hidden_states: a dense float Tensor of shape
                `(batch_size, encoder_sequence_length, hidden_dim)`. The
                sequence of hidden states at the output of the encoder's last
                layer.
            encoder_padding_mask: a dense float Tensor of shape
                `(batch_size, encoder_sequence_length)`. The padding mask for
                the encoder input.
            decoder_token_ids: a dense int Tensor of shape
                `(batch_size, max_length)`. Input token ids to be fed to
                the decoder.
            self_attention_cache: a dense float Tensor of shape
                `(batch_size, num_layers, 2, max_length, num_heads, key_value_dim)`.
                The cached key/value tensors of previously seen tokens in the
                decoder's self-attention layer.
            self_attention_cache_update_index: an int or int Tensor, the index
                at which to update the `self_attention_cache`. Usually, this is
                the index of the current token being processed during decoding.
            cross_attention_cache: a dense float Tensor of shape
                `(batch_size, num_layers, 2, encoder_sequence_length, num_heads, key_value_dim)`.
                The cached key/value tensors of the encoder outputs in the
                decoder's cross-attention layer.
            cross_attention_cache_update_index: an int or int Tensor, the index
                at which to update the `cross_attention_cache`. Usually, this is
                either `0` (compute the entire `cross_attention_cache`), or
                `None` (reuse a previously computed `cross_attention_cache`).

        Returns:
            A `(logits, hidden_states, self_attention_cache, cross_attention_cache)`
            tuple, where `logits` is the language model logits for the input
            `decoder_token_ids`, `hidden_states` is the final hidden
            representation of the input tokens, `self_attention_cache` is the
            key/value cache in the decoder's self-attention layer and
            `cross_attention_cache` is the key/value cache in the decoder's
            cross-attention layer.
        """  # noqa: E501
        x = self.backbone.token_embedding(decoder_token_ids)
        x = self.backbone.decoder_embedding_dropout(x)
        encoder_attention_mask = encoder_padding_mask[:, None, :]

        # Every decoder layer has a separate cache for the self-attention layer
        # and the cross-attention layer. We update all of them separately. The
        # relative attention bias is computed once by the first layer, along
        # with the causal mask, and threaded through the rest of the stack.
        self_attention_caches = []
        cross_attention_caches = []
        position_bias = None
        for i, layer in enumerate(self.backbone.decoder_transformer_layers):
            (
                x,
                position_bias,
                next_self_attention_cache,
                next_cross_attention_cache,
            ) = layer(
                x,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_causal_mask=True,
                self_attention_cache=self_attention_cache[:, i, ...],
                self_attention_cache_update_index=self_attention_cache_update_index,
                cross_attention_cache=cross_attention_cache[:, i, ...],
                cross_attention_cache_update_index=cross_attention_cache_update_index,
            )
            if self_attention_cache_update_index is not None:
                self_attention_caches.append(next_self_attention_cache)
            if cross_attention_cache_update_index is not None:
                cross_attention_caches.append(next_cross_attention_cache)

        if self_attention_cache_update_index is not None:
            self_attention_cache = ops.stack(self_attention_caches, axis=1)
        if cross_attention_cache_update_index is not None:
            cross_attention_cache = ops.stack(cross_attention_caches, axis=1)

        x = self.backbone.decoder_layer_norm(x)
        x = self.backbone.decoder_dropout(x)
        hidden_states = x
        logits = self._logits(hidden_states)
        return (
            logits,
            hidden_states,
            self_attention_cache,
            cross_attention_cache,
        )

    def _initialize_cache(self, encoder_token_ids, decoder_token_ids):
        """Initializes empty self-attention cache and cross-attention cache."""
        batch_size = ops.shape(encoder_token_ids)[0]
        encoder_max_length = ops.shape(encoder_token_ids)[1]
        decoder_max_length = ops.shape(decoder_token_ids)[1]

        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_heads
        head_dim = self.backbone.key_value_dim or (
            self.backbone.hidden_dim // num_heads
        )

        shape = [
            batch_size,
            num_layers,
            2,
            decoder_max_length,
            num_heads,
            head_dim,
        ]
        self_attention_cache = ops.zeros(shape, dtype=self.compute_dtype)

        shape[3] = encoder_max_length
        cross_attention_cache = ops.zeros(shape, dtype=self.compute_dtype)

        return (self_attention_cache, cross_attention_cache)

    def _build_cache(
        self, encoder_token_ids, encoder_padding_mask, decoder_token_ids
    ):
        """Builds the self-attention cache and the cross-attention cache."""
        encoder_hidden_states = self.call_encoder(
            token_ids=encoder_token_ids, padding_mask=encoder_padding_mask
        )
        self_attention_cache, cross_attention_cache = self._initialize_cache(
            encoder_token_ids, decoder_token_ids
        )

        # Seed the self-attention cache and the cross-attention cache.
        (
            _,
            hidden_states,
            self_attention_cache,
            cross_attention_cache,
        ) = self.call_decoder_with_cache(
            encoder_hidden_states=encoder_hidden_states,
            encoder_padding_mask=encoder_padding_mask,
            decoder_token_ids=decoder_token_ids,
            self_attention_cache=self_attention_cache,
            self_attention_cache_update_index=0,
            cross_attention_cache=cross_attention_cache,
            cross_attention_cache_update_index=0,
        )
        return (
            hidden_states,
            encoder_hidden_states,
            self_attention_cache,
            cross_attention_cache,
        )

    def generate_step(
        self,
        inputs,
        stop_token_ids=None,
    ):
        """A compilable generation function for a batch of inputs.

        This function represents the inner, XLA-compilable, generation function
        for a single batch of inputs. Inputs should have the same structure as
        model inputs, a dictionary with keys `"encoder_token_ids"`,
        `"encoder_padding_mask"`, `"decoder_token_ids"` and
        `"decoder_padding_mask"`.

        Args:
            inputs: A dictionary with four keys - `"encoder_token_ids"`,
                `"encoder_padding_mask"`, `"decoder_token_ids"` and
                `"decoder_padding_mask"`, with batched tensor values.
            stop_token_ids: Tuple of id's of end token's to stop on. If all
                sequences have produced a new stop token, generation
                will stop.
        """
        (
            encoder_token_ids,
            encoder_padding_mask,
            decoder_token_ids,
            decoder_padding_mask,
        ) = (
            inputs["encoder_token_ids"],
            inputs["encoder_padding_mask"],
            inputs["decoder_token_ids"],
            inputs["decoder_padding_mask"],
        )

        batch_size = ops.shape(encoder_token_ids)[0]

        # Create and seed cache with a single forward pass.
        (
            hidden_states,
            encoder_hidden_states,
            self_attention_cache,
            cross_attention_cache,
        ) = self._build_cache(
            encoder_token_ids, encoder_padding_mask, decoder_token_ids
        )
        # Compute the lengths of all user inputted tokens ids.
        row_lengths = ops.sum(ops.cast(decoder_padding_mask, "int32"), axis=-1)
        # Start at the first index that has no user inputted id.
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            # The cache index is the index of our previous token.
            cache_index = index - 1
            num_samples = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, cache_index], [num_samples, 1])

            def repeat_tensor(x):
                """Repeats along batch axis to match dim for beam search."""
                if ops.shape(x)[0] == num_samples:
                    return x
                return ops.repeat(x, repeats=num_samples // batch_size, axis=0)

            logits, hidden_states, cache, _ = self.call_decoder_with_cache(
                encoder_hidden_states=repeat_tensor(encoder_hidden_states),
                encoder_padding_mask=repeat_tensor(encoder_padding_mask),
                decoder_token_ids=prompt,
                self_attention_cache=cache,
                self_attention_cache_update_index=cache_index,
                cross_attention_cache=repeat_tensor(cross_attention_cache),
                cross_attention_cache_update_index=None,
            )
            return (
                ops.squeeze(logits, axis=1),
                ops.squeeze(hidden_states, axis=1),
                cache,
            )

        decoder_token_ids = self.sampler(
            next=next,
            prompt=decoder_token_ids,
            cache=self_attention_cache,
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
                ops.logical_not(decoder_padding_mask),
            )
            end_locations = ops.cast(end_locations, "int32")
            # Use cumsum to get ones in all locations after `end_locations`.
            cumsum = ops.cast(ops.cumsum(end_locations, axis=-1), "int32")
            overflow = cumsum - end_locations
            # Our padding mask is the inverse of these overflow locations.
            decoder_padding_mask = ops.logical_not(ops.cast(overflow, "bool"))
        else:
            # Without early stopping, all locations will have been updated.
            decoder_padding_mask = ops.ones_like(
                decoder_token_ids, dtype="bool"
            )

        return {
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }
