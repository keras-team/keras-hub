# Copyright 2024 The KerasHub Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.bart.bart_backbone import BartBackbone
from keras_hub.src.models.bart.bart_seq_2_seq_lm_preprocessor import (
    BartSeq2SeqLMPreprocessor,
)
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.BartSeq2SeqLM")
class BartSeq2SeqLM(Seq2SeqLM):
    """An end-to-end BART model for seq2seq language modeling.

    A seq2seq language model (LM) is an encoder-decoder model which is used for
    conditional text generation. The encoder is given a "context" text (fed to
    the encoder), and the decoder predicts the next token based on both the
    encoder inputs and the previous tokens. You can finetune `BartSeq2SeqLM` to
    generate text for any seq2seq task (e.g., translation or summarization).

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
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/facebookresearch/fairseq/).

    Args:
        backbone: A `keras_hub.models.BartBackbone` instance.
        preprocessor: A `keras_hub.models.BartSeq2SeqLMPreprocessor` or `None`.
            If `None`, this model will not apply preprocessing, and inputs
            should be preprocessed before calling the model.

    Examples:

    Use `generate()` to do text generation, given an input context.
    ```python
    bart_lm = keras_hub.models.BartSeq2SeqLM.from_preset("bart_base_en")
    bart_lm.generate("The quick brown fox", max_length=30)

    # Generate with batched inputs.
    bart_lm.generate(["The quick brown fox", "The whale"], max_length=30)
    ```

    Compile the `generate()` function with a custom sampler.
    ```python
    bart_lm = keras_hub.models.BartSeq2SeqLM.from_preset("bart_base_en")
    bart_lm.compile(sampler="greedy")
    bart_lm.generate("The quick brown fox", max_length=30)
    ```

    Use `generate()` with encoder inputs and an incomplete decoder input (prompt).
    ```python
    bart_lm = keras_hub.models.BartSeq2SeqLM.from_preset("bart_base_en")
    bart_lm.generate(
        {
            "encoder_text": "The quick brown fox",
            "decoder_text": "The fast"
        }
    )
    ```

    Use `generate()` without preprocessing.
    ```python
    # Preprocessed inputs, with encoder inputs corresponding to
    # "The quick brown fox", and the decoder inputs to "The fast". Use
    # `"padding_mask"` to indicate values that should not be overridden.
    prompt = {
        "encoder_token_ids": np.array([[0, 133, 2119, 6219, 23602, 2, 1, 1]]),
        "encoder_padding_mask": np.array(
            [[True, True, True, True, True, True, False, False]]
        ),
        "decoder_token_ids": np.array([[2, 0, 133, 1769, 2, 1, 1]]),
        "decoder_padding_mask": np.array([[True, True, True, True, False, False]])
    }

    bart_lm = keras_hub.models.BartSeq2SeqLM.from_preset(
        "bart_base_en",
        preprocessor=None,
    )
    bart_lm.generate(prompt)
    ```

    Call `fit()` on a single batch.
    ```python
    features = {
        "encoder_text": ["The quick brown fox jumped.", "I forgot my homework."],
        "decoder_text": ["The fast hazel fox leapt.", "I forgot my assignment."]
    }
    bart_lm = keras_hub.models.BartSeq2SeqLM.from_preset("bart_base_en")
    bart_lm.fit(x=features, batch_size=2)
    ```

    Call `fit()` without preprocessing.
    ```python
    x = {
        "encoder_token_ids": np.array([[0, 133, 2119, 2, 1]] * 2),
        "encoder_padding_mask": np.array([[1, 1, 1, 1, 0]] * 2),
        "decoder_token_ids": np.array([[2, 0, 133, 1769, 2]] * 2),
        "decoder_padding_mask": np.array([[1, 1, 1, 1, 1]] * 2),
    }
    y = np.array([[0, 133, 1769, 2, 1]] * 2)
    sw = np.array([[1, 1, 1, 1, 0]] * 2)

    bart_lm = keras_hub.models.BartSeq2SeqLM.from_preset(
        "bart_base_en",
        preprocessor=None,
    )
    bart_lm.fit(x=x, y=y, sample_weight=sw, batch_size=2)
    ```

    Custom backbone and vocabulary.
    ```python
    features = {
        "encoder_text": [" afternoon sun"],
        "decoder_text": ["noon sun"],
    }
    vocab = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "Ġafter": 5,
        "noon": 6,
        "Ġsun": 7,
    }
    merges = ["Ġ a", "Ġ s", "Ġ n", "e r", "n o", "o n", "Ġs u", "Ġa f", "no on"]
    merges += ["Ġsu n", "Ġaf t", "Ġaft er"]

    tokenizer = keras_hub.models.BartTokenizer(
        vocabulary=vocab,
        merges=merges,
    )
    preprocessor = keras_hub.models.BartSeq2SeqLMPreprocessor(
        tokenizer=tokenizer,
        encoder_sequence_length=128,
        decoder_sequence_length=128,
    )
    backbone = keras_hub.models.BartBackbone(
        vocabulary_size=50265,
        num_layers=6,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=128,
    )
    bart_lm = keras_hub.models.BartSeq2SeqLM(
        backbone=backbone,
        preprocessor=preprocessor,
    )
    bart_lm.fit(x=features, batch_size=2)
    ```
    """

    backbone_cls = BartBackbone
    preprocessor_cls = BartSeq2SeqLMPreprocessor

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
        outputs = backbone.token_embedding(hidden_states, reverse=True)
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            **kwargs,
        )

    def call_decoder_with_cache(
        self,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_token_ids,
        self_attention_cache=None,
        self_attention_cache_update_index=None,
        cross_attention_cache=None,
        cross_attention_cache_update_index=None,
    ):
        """Forward pass with a key/value caches for generative decoding..

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
                `(batch_size, num_layers, 2, max_length, num_heads, key_dims)`.
                The cached key/value tensors of previously seen tokens in the
                decoder's self-attention layer.
            self_attention_cache_update_index: an int or int Tensor, the index
                at which to update the `self_attention_cache`. Usually, this is
                the index of the current token being processed during decoding.
            cross_attention_cache: a dense float Tensor of shape
                `(batch_size, num_layers, 2, encoder_sequence_length, num_heads, key_dims)`.
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
        """
        # Embedding layers.
        tokens = self.backbone.token_embedding(decoder_token_ids)
        positions = self.backbone.decoder_position_embedding(
            tokens,
            start_index=self_attention_cache_update_index,
        )
        # Sum, normalize and apply dropout to embeddings.
        x = self.backbone.decoder_embeddings_add((tokens, positions))
        x = self.backbone.decoder_embeddings_layer_norm(x)
        x = self.backbone.decoder_embeddings_dropout(x)

        # Every decoder layer has a separate cache for the self-attention layer
        # and the cross-attention layer. We update all of them separately.
        self_attention_caches = []
        cross_attention_caches = []
        for i, layer in enumerate(self.backbone.decoder_transformer_layers):
            current_self_attention_cache = self_attention_cache[:, i, ...]
            current_cross_attention_cache = cross_attention_cache[:, i, ...]
            (
                x,
                next_self_attention_cache,
                next_cross_attention_cache,
            ) = layer(
                decoder_sequence=x,
                encoder_sequence=encoder_hidden_states,
                encoder_padding_mask=encoder_padding_mask,
                self_attention_cache=current_self_attention_cache,
                self_attention_cache_update_index=self_attention_cache_update_index,
                cross_attention_cache=current_cross_attention_cache,
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

        hidden_states = x
        logits = self.backbone.token_embedding(hidden_states, reverse=True)
        return (
            logits,
            hidden_states,
            self_attention_cache,
            cross_attention_cache,
        )

    def call_encoder(self, token_ids, padding_mask):
        """Does a forward pass on the encoder and returns the encoder output."""
        tokens = self.backbone.token_embedding(token_ids)
        positions = self.backbone.encoder_position_embedding(tokens)
        x = self.backbone.decoder_embeddings_add((tokens, positions))
        x = self.backbone.encoder_embeddings_layer_norm(x)
        x = self.backbone.encoder_embeddings_dropout(x)
        for transformer_layer in self.backbone.encoder_transformer_layers:
            x = transformer_layer(x, padding_mask=padding_mask)
        return x

    def _initialize_cache(self, encoder_token_ids, decoder_token_ids):
        """Initializes empty self-attention cache and cross-attention cache."""
        batch_size = ops.shape(encoder_token_ids)[0]
        encoder_max_length = ops.shape(encoder_token_ids)[1]
        decoder_max_length = ops.shape(decoder_token_ids)[1]

        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_heads
        head_dim = self.backbone.hidden_dim // self.backbone.num_heads

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
        """Builds the self-attention cache and the cross-attention cache (key/value pairs)."""
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
                """Repeats tensors along batch axis to match dim for beam search."""
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
