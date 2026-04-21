import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.audio_to_text import AudioToText
from keras_hub.src.models.whisper.whisper_audio_to_text_preprocessor import (
    WhisperAudioToTextPreprocessor,
)
from keras_hub.src.models.whisper.whisper_backbone import WhisperBackbone
from keras_hub.src.utils.tensor_utils import any_equal


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
    `"top_k"` sampling is used.

    Args:
        backbone: A `keras_hub.models.WhisperBackbone` instance.
        preprocessor: A `keras_hub.models.WhisperAudioToTextPreprocessor` or
            `None`. If `None`, model inputs must be preprocessed ahead of time.

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
    whisper_lm.compile(sampler="greedy")
    whisper_lm.generate({"audio": audio_tensor})
    ```

    Use `generate()` with a decoder prompt.
    ```python
    whisper_lm.generate({"audio": audio_tensor, "text": "The quick"})
    ```
    """

    backbone_cls = WhisperBackbone
    preprocessor_cls = WhisperAudioToTextPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
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
            self_attention_cache: Self-attention cache of shape
                `(batch_size, num_layers, 2, max_length, num_heads, head_dim)`.
            self_attention_cache_update_index: Index at which to update the
                self-attention cache.
            cross_attention_cache: Cross-attention cache of shape
                `(batch_size, num_layers, 2, encoder_seq_len, num_heads,
                head_dim)`.
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
                self_attention_cache=current_self_cache,
                self_attention_cache_update_index=self_attention_cache_update_index,
                cross_attention_cache=current_cross_cache,
                cross_attention_cache_update_index=cross_attention_cache_update_index,
            )
            if self_attention_cache_update_index is not None:
                self_attention_caches.append(next_self_cache)
            if cross_attention_cache_update_index is not None:
                cross_attention_caches.append(next_cross_cache)
        if self_attention_cache_update_index is not None:
            self_attention_cache = ops.stack(self_attention_caches, axis=1)
        if cross_attention_cache_update_index is not None:
            cross_attention_cache = ops.stack(cross_attention_caches, axis=1)
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
        num_layers = self.backbone.num_layers
        num_heads = self.backbone.num_heads
        head_dim = self.backbone.hidden_dim // self.backbone.num_heads
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

    def _build_cache(self, encoder_features, decoder_token_ids):
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
        batch_size = ops.shape(encoder_features)[0]

        (
            hidden_states,
            encoder_hidden_states,
            self_attention_cache,
            cross_attention_cache,
        ) = self._build_cache(encoder_features, decoder_token_ids)
        row_lengths = ops.sum(ops.cast(decoder_padding_mask, "int32"), axis=-1)
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            cache_index = index - 1
            num_samples = ops.shape(prompt)[0]
            prompt = ops.slice(prompt, [0, cache_index], [num_samples, 1])

            def repeat_tensor(x):
                if ops.shape(x)[0] == num_samples:
                    return x
                return ops.repeat(x, repeats=num_samples // batch_size, axis=0)

            logits, hidden_states, cache, _ = self.call_decoder_with_cache(
                encoder_hidden_states=repeat_tensor(encoder_hidden_states),
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
