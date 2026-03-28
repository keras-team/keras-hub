from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.audio_to_text import AudioToText
from keras_hub.src.models.qwen3_asr.qwen3_asr_audio_to_text_preprocessor import (  # noqa: E501
    Qwen3ASRAudioToTextPreprocessor,
)
from keras_hub.src.models.qwen3_asr.qwen3_asr_backbone import Qwen3ASRBackbone
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.Qwen3ASRAudioToText")
class Qwen3ASRAudioToText(AudioToText):
    """End-to-end Qwen3-ASR model for audio-to-text tasks.

    This model wraps a ``Qwen3ASRBackbone`` for speech recognition. The
    architecture uses replacement embedding: audio encoder outputs are
    scattered into ``<|AUDIO|>`` placeholder positions in the token
    sequence, and the full sequence is processed by the Qwen3 decoder.

    The model provides a ``generate()`` method for autoregressive text
    generation from audio input. The generation strategy is controlled by
    a ``sampler`` argument passed to ``compile()``. By default,
    ``"greedy"`` sampling is used.

    Args:
        backbone: A ``Qwen3ASRBackbone`` instance.
        preprocessor: A ``Qwen3ASRAudioToTextPreprocessor`` or ``None``.
            If ``None``, inputs must be preprocessed before calling the
            model.

    Examples:
    ```python
    asr = keras_hub.models.Qwen3ASRAudioToText.from_preset(
        "qwen3_asr_1.7b"
    )
    audio = np.random.randn(1, 16000).astype("float32")
    asr.generate({"audio": audio})

    # With a text prompt.
    asr.generate({"audio": audio, "text": "hello"})

    # Different sampling strategy.
    asr.compile(sampler="greedy")
    asr.generate({"audio": audio})
    ```
    """

    backbone_cls = Qwen3ASRBackbone
    preprocessor_cls = Qwen3ASRAudioToTextPreprocessor

    def __init__(self, backbone, preprocessor=None, **kwargs):
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
        """Forward pass with KV cache for autoregressive generation.

        During generation, only the new token embedding is processed
        through the decoder layers, reusing cached key/value states from
        previous steps. Audio replacement is not needed here since audio
        positions were already resolved in ``_build_cache``.

        Args:
            token_ids: int tensor of shape ``(batch_size, 1)``.
            cache: float tensor. Cached key/value states.
            cache_update_index: int. Position index for the new token.

        Returns:
            Tuple of ``(logits, hidden_states, cache)``.
        """
        x = self.backbone.token_embedding(token_ids)
        updated_cache = []
        for i in range(self.backbone.num_layers):
            current_cache = cache[:, i, ...]
            x, next_cache = self.backbone.transformer_layers[i](
                x,
                self_attention_cache=current_cache,
                self_attention_cache_update_index=cache_update_index,
            )
            updated_cache.append(next_cache)
        cache = ops.stack(updated_cache, axis=1)
        hidden_states = x = self.backbone.layer_norm(x)
        logits = self.backbone.token_embedding(x, reverse=True)
        return logits, hidden_states, cache

    def _build_cache(self, audio_features, token_ids, padding_mask):
        """Build the KV cache by running a full forward pass.

        The first pass processes the full token sequence with audio
        replacement to seed the cache. Subsequent generation steps use
        ``call_with_cache`` with single tokens (which are always text
        tokens, not audio placeholders).

        Args:
            audio_features: float tensor of mel spectrogram features.
            token_ids: int tensor with audio placeholders and text tokens.
            padding_mask: int tensor.

        Returns:
            Tuple of ``(hidden_states, cache)``.
        """
        # Embed all tokens.
        embeddings = self.backbone.token_embedding(token_ids)

        # Encode audio and replace placeholder embeddings.
        audio_embeddings = self.backbone.audio_encoder(audio_features)
        x = self.backbone.audio_replacer(
            embeddings, audio_embeddings, token_ids
        )

        full_seq_len = ops.shape(x)[1]
        batch_size = ops.shape(x)[0]
        num_layers = self.backbone.num_layers
        num_key_value_heads = self.backbone.num_key_value_heads
        head_dim = self.backbone.head_dim

        cache_shape = [
            batch_size,
            num_layers,
            2,
            full_seq_len,
            num_key_value_heads,
            head_dim,
        ]
        cache = ops.zeros(cache_shape, dtype=self.compute_dtype)

        updated_cache = []
        for i in range(num_layers):
            current_cache = cache[:, i, ...]
            x, next_cache = self.backbone.transformer_layers[i](
                x,
                decoder_padding_mask=padding_mask,
                self_attention_cache=current_cache,
                self_attention_cache_update_index=0,
            )
            updated_cache.append(next_cache)

        cache = ops.stack(updated_cache, axis=1)
        hidden_states = self.backbone.layer_norm(x)
        return hidden_states, cache

    def generate_step(self, inputs, stop_token_ids=None):
        """A compilable generation function for a batch of inputs.

        Args:
            inputs: Dictionary with keys ``"audio_features"``,
                ``"token_ids"``, and ``"padding_mask"``.
            stop_token_ids: Tuple of token IDs that signal generation
                should stop.

        Returns:
            Dictionary with ``"token_ids"`` and ``"padding_mask"``.
        """
        audio_features = inputs["audio_features"]
        token_ids = inputs["token_ids"]
        padding_mask = inputs["padding_mask"]

        # Build cache with audio replacement.
        hidden_states, cache = self._build_cache(
            audio_features, token_ids, padding_mask
        )

        # Compute where generation should start.
        row_lengths = ops.sum(ops.cast(padding_mask, "int32"), axis=-1)
        index = ops.min(row_lengths)

        def next(prompt, cache, index):
            cache_update_index = index - 1
            batch_size = ops.shape(prompt)[0]
            next_token = ops.slice(
                prompt, [0, cache_update_index], [batch_size, 1]
            )
            logits, hidden_states, cache = self.call_with_cache(
                next_token,
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

        # Build output padding mask.
        if stop_token_ids is not None:
            end_locations = any_equal(
                token_ids,
                stop_token_ids,
                ops.logical_not(ops.cast(padding_mask, "bool")),
            )
            end_locations = ops.cast(end_locations, "int32")
            cumsum = ops.cumsum(end_locations, axis=-1)
            overflow = cumsum - end_locations
            padding_mask = ops.logical_not(ops.cast(overflow, "bool"))
        else:
            padding_mask = ops.ones_like(token_ids, dtype="bool")

        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
