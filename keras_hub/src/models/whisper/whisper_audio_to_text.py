import keras
import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.audio_to_text import AudioToText
from keras_hub.src.models.whisper.whisper_audio_to_text_preprocessor import (
    WhisperAudioToTextPreprocessor,
)
from keras_hub.src.models.whisper.whisper_backbone import WhisperBackbone
from keras_hub.src.utils.tensor_utils import any_equal
from keras_hub.src.utils.tensor_utils import tf


@keras_hub_export("keras_hub.models.WhisperAudioToText")
class WhisperAudioToText(AudioToText):
    """An end-to-end Whisper model for audio-to-text tasks.

    A Seq2Seq LM designed for automatic speech recognition and translation. The
    encoder consumes log-mel spectrogram features produced by
    `keras_hub.layers.WhisperAudioConverter`, and the decoder auto-regressively
    generates text tokens. You can finetune `WhisperAudioToText` for any
    audio-to-text task (e.g., multilingual transcription or translation).

    This model has a `generate()` method that generates text based on an audio
    input and an optional text prompt for the decoder. The generation strategy
    is controlled by the `sampler` argument passed to `compile()`. By default,
    `"greedy"` sampling is used, matching Whisper's deterministic decoding.

    The preprocessor starts every decoder sequence with Whisper's control
    prefix (`<|startoftranscript|>`, then the language and task tokens on
    multilingual checkpoints, then `<|notimestamps|>`). A `"text"` value passed
    to `generate()` is appended after that prefix as a forced transcript
    prefix: decoding continues from it, and it stays in the returned
    transcription (as with `keras_hub.models.MoonshineAudioToText`). Whisper's
    prior context prompt (the tokens after `<|startofprev|>` that Hugging Face
    passes as `prompt_ids`) is not supported yet.

    Args:
        backbone: A `keras_hub.models.WhisperBackbone` instance.
        preprocessor: A `keras_hub.models.WhisperAudioToTextPreprocessor` or
            `None`. If `None`, model inputs must be preprocessed ahead of time.
            Defaults to `None`.
        suppress_tokens: list of ints, or `None`. Token ids whose logits are
            masked out at every generation step. Defaults to `None`.
        begin_suppress_tokens: list of ints, or `None`. Token ids whose logits
            are masked out at the first generated step only (Whisper uses the
            space token and `<|endoftext|>`). Defaults to `None`.

    Examples:

    Use `generate()` to transcribe audio.
    ```python
    whisper_lm = keras_hub.models.WhisperAudioToText.from_preset(
        "whisper_tiny_en"
    )
    audio_tensor = keras.random.normal((1, 16000))
    whisper_lm.generate({"audio": audio_tensor})
    ```

    Compile the `generate()` function with a custom sampler.
    ```python
    whisper_lm.compile(sampler="top_k")
    whisper_lm.generate({"audio": audio_tensor})
    ```

    Use `generate()` with a forced transcript prefix.
    ```python
    whisper_lm.generate({"audio": audio_tensor, "text": " The quick"})
    ```

    Call `fit()` on a single batch.
    ```python
    features = {
        "audio": keras.random.normal((2, 16000)),
        "text": [" The quick brown fox.", " Jumped over the lazy dog."],
    }
    whisper_lm = keras_hub.models.WhisperAudioToText.from_preset(
        "whisper_tiny_en"
    )
    whisper_lm.fit(x=features, batch_size=2)
    ```
    """

    backbone_cls = WhisperBackbone
    preprocessor_cls = WhisperAudioToTextPreprocessor

    def __init__(
        self,
        backbone,
        preprocessor=None,
        suppress_tokens=None,
        begin_suppress_tokens=None,
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

        # === Config ===
        self.suppress_tokens = (
            None
            if suppress_tokens is None
            else [int(t) for t in suppress_tokens]
        )
        self.begin_suppress_tokens = (
            None
            if begin_suppress_tokens is None
            else [int(t) for t in begin_suppress_tokens]
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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "suppress_tokens": self.suppress_tokens,
                "begin_suppress_tokens": self.begin_suppress_tokens,
            }
        )
        return config

    def _normalize_generate_inputs(self, inputs):
        """Batch unbatched audio and decide whether the output is scalar.

        Unlike `CausalLM._normalize_generate_inputs`, this copies the input
        dictionary instead of mutating it, and decides `input_is_scalar` from
        the rank of `"audio"` alone, so the result does not depend on the key
        order or on whether `"text"` is a scalar string.
        """
        if tf and isinstance(inputs, tf.data.Dataset):
            return inputs.as_numpy_iterator(), False

        if self.preprocessor is None:
            return [inputs], False

        if isinstance(inputs, dict):
            inputs = dict(inputs)
        else:
            inputs = {"audio": inputs}

        audio = inputs["audio"]
        input_is_scalar = _rank(audio) == 1
        if input_is_scalar:
            inputs["audio"] = _add_batch_dim(audio)
            if "text" in inputs:
                text = inputs["text"]
                if isinstance(text, (str, bytes)):
                    inputs["text"] = [text]
                elif _rank(text) == 0:
                    inputs["text"] = _add_batch_dim(text)
        return [inputs], input_is_scalar

    def call_encoder(self, encoder_features):
        """Run the audio encoder on log-mel features."""
        x = keras.activations.gelu(
            self.backbone.encoder_conv_layer_1(encoder_features),
            approximate=False,
        )
        x = self.backbone.encoder_padder(x)
        x = keras.activations.gelu(
            self.backbone.encoder_conv_layer_2(x),
            approximate=False,
        )
        positions = self.backbone.encoder_position_embedding(x)
        x = self.backbone.encoder_embeddings_add((x, positions))
        x = self.backbone.encoder_embeddings_dropout(x)
        for layer in self.backbone.encoder_transformer_layers:
            x = layer(x)
        x = self.backbone.encoder_layer_norm(x)
        return x

    def call_decoder_with_cache(
        self,
        encoder_hidden_states,
        decoder_token_ids,
        decoder_padding_mask=None,
        self_attention_cache=None,
        self_attention_cache_update_index=None,
        cross_attention_cache=None,
        cross_attention_cache_update_index=None,
    ):
        """Forward pass of the decoder with cached key/value tensors.

        Args:
            encoder_hidden_states: The encoder output of shape
                `(batch_size, encoder_sequence_length, hidden_dim)`.
            decoder_token_ids: Decoder input token ids.
            decoder_padding_mask: Optional padding mask for
                `decoder_token_ids`, forwarded to every decoder layer. Pass it
                when prefilling the cache with padded inputs. Defaults to
                `None`.
            self_attention_cache: Self-attention cache of shape
                `(batch_size, num_layers, 2, max_length, num_heads, head_dim)`,
                or `None` to run the decoder without caches. Must be set
                together with `cross_attention_cache`.
            self_attention_cache_update_index: Index at which to update the
                self-attention cache.
            cross_attention_cache: Cross-attention cache of shape
                `(batch_size, num_layers, 2, encoder_seq_len, num_heads,
                head_dim)`, or `None` to run the decoder without caches.
            cross_attention_cache_update_index: Index at which to update the
                cross-attention cache. Pass `0` to compute the full cache from
                scratch, or `None` to reuse a previously computed cache.

        Returns:
            A tuple `(logits, hidden_states, self_attention_cache,
            cross_attention_cache)`.
        """
        start_index = (
            self_attention_cache_update_index
            if self_attention_cache_update_index is not None
            else 0
        )
        x = self.backbone.decoder_embeddings(
            decoder_token_ids,
            start_index=start_index,
        )
        x = self.backbone.decoder_embeddings_dropout(x)
        if (self_attention_cache is None) != (cross_attention_cache is None):
            raise ValueError(
                "`self_attention_cache` and `cross_attention_cache` should "
                "both be set or both be `None`. Received: "
                f"self_attention_cache={self_attention_cache}, "
                f"cross_attention_cache={cross_attention_cache}."
            )
        if self_attention_cache is None:
            # Uncached forward pass, e.g. for scoring a full sequence.
            for layer in self.backbone.decoder_transformer_layers:
                x = layer(
                    decoder_sequence=x,
                    encoder_sequence=encoder_hidden_states,
                    decoder_padding_mask=decoder_padding_mask,
                )
        else:
            self_attention_caches = []
            cross_attention_caches = []
            for i, layer in enumerate(self.backbone.decoder_transformer_layers):
                current_self_cache = self_attention_cache[:, i, ...]
                current_cross_cache = cross_attention_cache[:, i, ...]
                (
                    x,
                    next_self_cache,
                    next_cross_cache,
                ) = layer(
                    decoder_sequence=x,
                    encoder_sequence=encoder_hidden_states,
                    decoder_padding_mask=decoder_padding_mask,
                    self_attention_cache=current_self_cache,
                    self_attention_cache_update_index=(
                        self_attention_cache_update_index
                    ),
                    cross_attention_cache=current_cross_cache,
                    cross_attention_cache_update_index=(
                        cross_attention_cache_update_index
                    ),
                )
                if self_attention_cache_update_index is not None:
                    self_attention_caches.append(next_self_cache)
                if cross_attention_cache_update_index is not None:
                    cross_attention_caches.append(next_cross_cache)
            if self_attention_cache_update_index is not None:
                self_attention_cache = ops.stack(self_attention_caches, axis=1)
            if cross_attention_cache_update_index is not None:
                cross_attention_cache = ops.stack(
                    cross_attention_caches, axis=1
                )
        hidden_states = self.backbone.decoder_layer_norm(x)
        logits = self.backbone.token_embedding(hidden_states, reverse=True)
        return (
            logits,
            hidden_states,
            self_attention_cache,
            cross_attention_cache,
        )

    def _initialize_cache(self, encoder_hidden_states, decoder_token_ids):
        """Initialize empty self-attention and cross-attention caches."""
        batch_size = ops.shape(encoder_hidden_states)[0]
        encoder_length = ops.shape(encoder_hidden_states)[1]
        decoder_length = ops.shape(decoder_token_ids)[1]
        num_layers = self.backbone.decoder_num_layers
        num_heads = self.backbone.decoder_num_heads
        head_dim = self.backbone.hidden_dim // num_heads
        self_cache_shape = [
            batch_size,
            num_layers,
            2,
            decoder_length,
            num_heads,
            head_dim,
        ]
        self_attention_cache = ops.zeros(
            self_cache_shape, dtype=self.compute_dtype
        )
        cross_cache_shape = [
            batch_size,
            num_layers,
            2,
            encoder_length,
            num_heads,
            head_dim,
        ]
        cross_attention_cache = ops.zeros(
            cross_cache_shape, dtype=self.compute_dtype
        )
        return self_attention_cache, cross_attention_cache

    def _build_cache(
        self, encoder_features, decoder_token_ids, decoder_padding_mask=None
    ):
        """Seed the self-attention and cross-attention caches."""
        encoder_hidden_states = self.call_encoder(encoder_features)
        self_attention_cache, cross_attention_cache = self._initialize_cache(
            encoder_hidden_states, decoder_token_ids
        )
        (
            _,
            hidden_states,
            self_attention_cache,
            cross_attention_cache,
        ) = self.call_decoder_with_cache(
            encoder_hidden_states=encoder_hidden_states,
            decoder_token_ids=decoder_token_ids,
            decoder_padding_mask=decoder_padding_mask,
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

    def _suppress_mask(self, token_ids):
        """A `(vocabulary_size,)` boolean mask with `token_ids` set."""
        mask = np.zeros((self.backbone.vocabulary_size,), dtype="bool")
        mask[np.asarray(token_ids, dtype="int64")] = True
        return ops.convert_to_tensor(mask)

    def _suppress_logits(self, logits, index, row_lengths):
        """Mask out Whisper's suppressed tokens.

        `suppress_tokens` are masked at every step and `begin_suppress_tokens`
        at the first generated step of each sequence, i.e. where `index` is
        the length of that sequence's prompt.
        """
        neg_inf = ops.cast(-1e9, logits.dtype)
        if self.suppress_tokens:
            mask = self._suppress_mask(self.suppress_tokens)
            logits = ops.where(mask[None, :], neg_inf, logits)
        if self.begin_suppress_tokens:
            mask = self._suppress_mask(self.begin_suppress_tokens)
            is_first_step = ops.equal(index, row_lengths)
            mask = ops.logical_and(is_first_step[:, None], mask[None, :])
            logits = ops.where(mask, neg_inf, logits)
        return logits

    def generate_step(self, inputs, stop_token_ids=None):
        """A compilable generation function for a batch of inputs.

        Args:
            inputs: A dictionary with keys `"encoder_features"`,
                `"decoder_token_ids"`, and `"decoder_padding_mask"`.
            stop_token_ids: Tuple of stop token ids. Generation halts once
                every sequence has produced a stop token.

        Returns:
            A dictionary with keys `"decoder_token_ids"` and
            `"decoder_padding_mask"`.
        """
        encoder_features = inputs["encoder_features"]
        decoder_token_ids = inputs["decoder_token_ids"]
        decoder_padding_mask = ops.cast(
            inputs["decoder_padding_mask"], dtype="bool"
        )
        # Whisper uses `pad_token_id == eos_token_id`, so the buffer's pad
        # region holds a stop token. The sampler's stop predicate inspects
        # unmasked positions for stop tokens, which would short-circuit the
        # loop before any token is sampled. Zero out the unmasked region so
        # only actually generated tokens trigger the stop predicate.
        decoder_token_ids = ops.where(
            decoder_padding_mask,
            decoder_token_ids,
            ops.zeros_like(decoder_token_ids),
        )
        (
            hidden_states,
            encoder_hidden_states,
            self_attention_cache,
            cross_attention_cache,
        ) = self._build_cache(
            encoder_features, decoder_token_ids, decoder_padding_mask
        )
        row_lengths = ops.sum(ops.cast(decoder_padding_mask, "int32"), axis=-1)
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            # Carry encoder tensors (and the prompt lengths used for
            # `begin_suppress_tokens`) through the loop, as in Moonshine, so
            # OpenVINO can trace them and beam search expands every cache leaf.
            (
                self_attention_cache,
                cross_attention_cache,
                encoder_hidden_states,
                row_lengths,
            ) = cache
            cache_index = index - 1
            num_samples = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, cache_index], [num_samples, 1])

            logits, hidden_states, self_attention_cache, _ = (
                self.call_decoder_with_cache(
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_token_ids=prompt,
                    self_attention_cache=self_attention_cache,
                    self_attention_cache_update_index=cache_index,
                    cross_attention_cache=cross_attention_cache,
                    cross_attention_cache_update_index=None,
                )
            )
            logits = ops.squeeze(logits, axis=1)
            logits = self._suppress_logits(logits, index, row_lengths)
            return (
                logits,
                ops.squeeze(hidden_states, axis=1),
                (
                    self_attention_cache,
                    cross_attention_cache,
                    encoder_hidden_states,
                    row_lengths,
                ),
            )

        decoder_token_ids = self.sampler(
            next=next,
            prompt=decoder_token_ids,
            cache=(
                self_attention_cache,
                cross_attention_cache,
                encoder_hidden_states,
                row_lengths,
            ),
            index=index,
            mask=decoder_padding_mask,
            stop_token_ids=stop_token_ids,
            hidden_states=hidden_states,
            model=self,
        )

        if stop_token_ids is not None:
            end_locations = any_equal(
                decoder_token_ids,
                stop_token_ids,
                ops.logical_not(decoder_padding_mask),
            )
            end_locations = ops.cast(end_locations, "int32")
            cumsum = ops.cast(ops.cumsum(end_locations, axis=-1), "int32")
            overflow = cumsum - end_locations
            decoder_padding_mask = ops.logical_not(ops.cast(overflow, "bool"))
        else:
            decoder_padding_mask = ops.ones_like(
                decoder_token_ids, dtype="bool"
            )

        return {
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }


def _rank(x):
    """The rank of a tensor, array, or nested list."""
    if hasattr(x, "shape") and x.shape is not None:
        return len(x.shape)
    if isinstance(x, (str, bytes)):
        return 0
    return np.ndim(x)


def _add_batch_dim(x):
    """Add a leading batch dimension without changing the input's type."""
    if tf is not None and isinstance(x, tf.Tensor):
        return x[tf.newaxis]
    if isinstance(x, np.ndarray):
        return x[np.newaxis]
    if ops.is_tensor(x):
        return ops.expand_dims(x, axis=0)
    return [x]
