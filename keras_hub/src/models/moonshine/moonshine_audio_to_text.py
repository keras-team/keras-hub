import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.audio_to_text import AudioToText
from keras_hub.src.models.moonshine.moonshine_audio_to_text_preprocessor import (  # noqa: E501
    MoonshineAudioToTextPreprocessor,
)
from keras_hub.src.models.moonshine.moonshine_backbone import Arange
from keras_hub.src.models.moonshine.moonshine_backbone import MoonshineBackbone
from keras_hub.src.models.moonshine.moonshine_backbone import (
    compute_output_lengths,
)
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.MoonshineAudioToText")
class MoonshineAudioToText(AudioToText):
    """An end-to-end Moonshine model for audio-to-text tasks.

    A Seq2Seq LM designed for audio-to-text tasks, such as speech recognition.
    The encoder processes audio features, and the decoder generates text
    transcriptions. You can finetune `MoonshineAudioToText` for any
    audio-to-text task (e.g., live transcription or voice commands).

    This model includes a `generate()` method for text generation based on audio
    inputs and an optional text prompt for the decoder. The generation strategy
    is controlled by a `sampler` argument passed to `compile()`. By default,
    `"top_k"` sampling is used.

    Args:
        backbone: A `keras_hub.models.MoonshineBackbone` instance.
        preprocessor: A `keras_hub.models.MoonshineAudioToTextPreprocessor` or
            `None`. If `None`, inputs must be preprocessed before calling the
            model.

    Examples:
    ```python
    # Initialize model from preset.
    moonshine_lm = keras_hub.models.MoonshineAudioToText.from_preset(
        "moonshine_base"
    )

    # Generate with single audio input.
    audio_tensor = keras.random.normal((1, 16000, 1))
    moonshine_lm.generate({"audio": audio_tensor})

    # Generate with text prompt.
    moonshine_lm.generate({"audio": audio_tensor, "text": "quick"})

    # Use different sampling strategy.
    moonshine_lm.compile(sampler="greedy")
    moonshine_lm.generate({"audio": audio_tensor})
    ```
    """

    # References:
    # Defined and formulated based on the Hugging Face implementation of the
    # MoonshineForConditionalGeneration class (https://github.com/huggingface/transformers/blob/dcbdf7e962c4b36140cc9ee76f870016121e69e5/src/transformers/models/moonshine/modeling_moonshine.py#L1509-L1626).

    backbone_cls = MoonshineBackbone
    preprocessor_cls = MoonshineAudioToTextPreprocessor

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

    def call_decoder_with_cache(
        self,
        encoder_hidden_states,
        encoder_padding_mask,
        decoder_token_ids,
        self_attention_cache=None,
        self_attention_cache_update_index=None,
        cross_attention_cache=None,
    ):
        """Process decoder inputs with attention caching for efficient
        generation.

        Args:
            encoder_hidden_states: Tensor. Encoder outputs.
            encoder_padding_mask: Tensor. Padding mask for encoder outputs.
            decoder_token_ids: Tensor. Decoder input token IDs.
            self_attention_cache: Tensor, optional. Cache for self-attention
                layers.
            self_attention_cache_update_index: int, optional. Index for cache
                updates.
            cross_attention_cache: Tensor, optional. Cache for cross-attention
                layers. This cache is computed once and reused.

        Returns:
            Tuple: Tuple of (logits, hidden_states, new_self_attention_cache,
            cross_attention_cache).
        """
        tokens = self.backbone.token_embedding(decoder_token_ids)
        x = tokens

        # Cache management for audio-to-text generation.
        self_attention_caches = []
        position = keras.ops.array(
            [self_attention_cache_update_index], dtype="int32"
        )
        rotary_embedding = self.backbone.decoder_rotary_embedding(position)

        for i, layer in enumerate(self.backbone.decoder_blocks):
            current_self_cache = self_attention_cache[:, i, ...]
            current_cross_cache = cross_attention_cache[:, i, ...]
            x, new_self_cache = layer(
                decoder_sequence=x,
                encoder_sequence=encoder_hidden_states,
                rotary_embedding=rotary_embedding,
                encoder_padding_mask=encoder_padding_mask,
                self_attention_cache=current_self_cache,
                self_attention_cache_update_index=self_attention_cache_update_index,
                cross_attention_cache=current_cross_cache,
                training=False,
            )
            # Update self-attention cache.
            self_attention_caches.append(new_self_cache)

        # [batch_size, num_layers, 2, seq_len, num_heads, head_dim].
        new_self_attention_cache = keras.ops.stack(
            self_attention_caches, axis=1
        )
        hidden_states = self.backbone.decoder_post_norm(x)
        logits = self.backbone.token_embedding(hidden_states, reverse=True)
        return (
            logits,
            hidden_states,
            new_self_attention_cache,
            cross_attention_cache,
        )

    def _build_cache(
        self,
        encoder_input_values,
        encoder_padding_mask,
        decoder_token_ids,
        decoder_padding_mask,
    ):
        """Build initial cache states from inputs."""
        encoder_hidden_states, encoder_attention_mask_for_decoder = (
            self.call_encoder(
                encoder_input_values=encoder_input_values,
                padding_mask=encoder_padding_mask,
            )
        )
        precomputed_cross_caches = []
        for layer in self.backbone.decoder_blocks:
            cross_k = layer.cross_attention._key_dense(encoder_hidden_states)
            cross_v = layer.cross_attention._value_dense(encoder_hidden_states)
            layer_cross_cache = keras.ops.stack([cross_k, cross_v], axis=1)
            precomputed_cross_caches.append(layer_cross_cache)
        precomputed_cross_cache = keras.ops.stack(
            precomputed_cross_caches, axis=1
        )
        batch_size = keras.ops.shape(encoder_input_values)[0]
        num_layers = self.backbone.decoder_num_layers
        num_heads = self.backbone.decoder_num_heads
        head_dim = self.backbone.hidden_dim // self.backbone.decoder_num_heads
        if self.backbone.pad_head_dim_to_multiple_of is not None:
            head_dim = (
                (head_dim + self.backbone.pad_head_dim_to_multiple_of - 1)
                // self.backbone.pad_head_dim_to_multiple_of
            ) * self.backbone.pad_head_dim_to_multiple_of
        # Use the full sequence length for the cache dimension.
        cache_length = keras.ops.shape(decoder_token_ids)[1]
        initial_self_cache_shape = (
            batch_size,
            num_layers,
            2,
            cache_length,
            num_heads,
            head_dim,
        )
        initial_self_cache = keras.ops.zeros(
            initial_self_cache_shape, dtype=self.compute_dtype
        )
        tokens = self.backbone.token_embedding(decoder_token_ids)
        x = tokens
        positions = keras.ops.arange(0, cache_length, dtype="int32")
        rotary_embedding = self.backbone.decoder_rotary_embedding(positions)
        seeded_self_caches = []
        for i, layer in enumerate(self.backbone.decoder_blocks):
            current_initial_self_cache = initial_self_cache[:, i, ...]
            current_precomputed_cross_cache = precomputed_cross_cache[:, i, ...]
            x, seeded_self_cache_layer = layer(
                decoder_sequence=x,
                encoder_sequence=encoder_hidden_states,
                rotary_embedding=rotary_embedding,
                decoder_padding_mask=decoder_padding_mask,
                encoder_padding_mask=encoder_attention_mask_for_decoder,
                self_attention_cache=current_initial_self_cache,
                self_attention_cache_update_index=0,
                cross_attention_cache=current_precomputed_cross_cache,
                training=False,
            )
            seeded_self_caches.append(seeded_self_cache_layer)
        hidden_states = self.backbone.decoder_post_norm(x)
        self_attn_cache = keras.ops.stack(seeded_self_caches, axis=1)
        return (
            hidden_states,
            self_attn_cache,
            precomputed_cross_cache,
            encoder_hidden_states,
            encoder_attention_mask_for_decoder,
        )

    def call_encoder(self, encoder_input_values, padding_mask):
        """Process audio input through the encoder stack."""
        x = self.backbone.conv1(encoder_input_values)
        x = self.backbone.tanh_after_conv1(x)
        x = self.backbone.group_norm(x)
        x = self.backbone.conv2(x)
        x = self.backbone.gelu_after_conv2(x)
        x = self.backbone.conv3(x)
        x = self.backbone.gelu_after_conv3(x)
        original_lengths = keras.ops.sum(
            keras.ops.cast(padding_mask, "int32"), axis=1
        )
        output_lengths = compute_output_lengths(original_lengths)
        padding_mask = self.backbone._compute_mask_layer(x, output_lengths)
        positions = Arange(name="encoder_positions")(x)
        rotary_embedding = self.backbone.encoder_rotary_embedding(positions)
        x = self.backbone.encoder_dropout(x, training=False)
        for transformer_layer in self.backbone.encoder_blocks:
            x = transformer_layer(
                inputs=x,
                rotary_embedding=rotary_embedding,
                attention_mask=padding_mask,
                training=False,
            )
        x = self.backbone.encoder_final_layer_norm(x)
        return x, padding_mask

    # Source: https://github.com/huggingface/transformers/blob/9e94801146ceeb3b215bbdb9492be74d7d7b7210/src/transformers/generation/utils.py#L1970-L2463
    def generate_step(self, inputs, stop_token_ids=None):
        """A compilable generation function for a batch of inputs.

        This function represents the inner, XLA-compilable, generation function
        for a single batch of inputs. Inputs should have the same structure as
        model inputs, a dictionary with keys `"encoder_input_values"`,
        `"encoder_padding_mask"`, `"decoder_token_ids"` and
        `"decoder_padding_mask"`.

        Args:
            inputs: A dictionary with four keys - `"encoder_input_values"`,
                `"encoder_padding_mask"`, `"decoder_token_ids"` and
                `"decoder_padding_mask"`, with batched tensor values.
            stop_token_ids: Tuple of id's of end token's to stop on. If all
                sequences have produced a new stop token, generation
                will stop.

        Returns:
            Dictionary: A dictionary with two keys - `"decoder_token_ids"`
                containing the updated token sequence with newly generated
                tokens, and `"decoder_padding_mask"` containing the updated
                padding mask for the generated sequence.
        """
        encoder_input_values = inputs["encoder_input_values"]
        encoder_padding_mask = inputs["encoder_padding_mask"]
        decoder_token_ids = inputs["decoder_token_ids"]
        decoder_padding_mask = inputs["decoder_padding_mask"]

        if (
            encoder_input_values is None
            or encoder_padding_mask is None
            or decoder_token_ids is None
        ):
            raise ValueError("Input tensors cannot be None")

        (
            hidden_states,
            self_attention_cache,
            cross_attention_cache,
            encoder_hidden_states,
            encoder_attention_mask_for_decoder,
        ) = self._build_cache(
            encoder_input_values,
            encoder_padding_mask,
            decoder_token_ids,
            decoder_padding_mask,
        )
        row_lengths = keras.ops.sum(
            keras.ops.cast(decoder_padding_mask, "int32"),
            axis=-1,
        )
        index = keras.ops.min(row_lengths)

        def next(prompt, cache, index):
            if isinstance(cache, tuple) and len(cache) == 2:
                current_self_attention_cache = cache[0]
                current_cross_attention_cache = cache[1]
            elif cache is not None and not isinstance(cache, tuple):
                current_self_attention_cache = cache
                current_cross_attention_cache = cross_attention_cache
            else:
                cache = None
            cache_index = index - 1
            num_samples = keras.ops.shape(prompt)[0]
            next_token_input = keras.ops.slice(
                prompt, [0, cache_index], [num_samples, 1]
            )

            batch_size = keras.ops.shape(encoder_input_values)[0]

            def repeat_tensor(x):
                if keras.ops.shape(x)[0] == num_samples:
                    return x
                return keras.ops.repeat(
                    x, repeats=num_samples // batch_size, axis=0
                )

            cross_attention_cache_repeated = repeat_tensor(
                current_cross_attention_cache
            )
            logits, hidden_states, new_self_attention_cache, _ = (
                self.call_decoder_with_cache(
                    encoder_hidden_states=repeat_tensor(encoder_hidden_states),
                    encoder_padding_mask=repeat_tensor(
                        encoder_attention_mask_for_decoder
                    ),
                    decoder_token_ids=next_token_input,
                    self_attention_cache=current_self_attention_cache,
                    self_attention_cache_update_index=cache_index,
                    cross_attention_cache=cross_attention_cache_repeated,
                )
            )
            return (
                logits[:, 0, :],
                hidden_states[:, 0, :],
                (new_self_attention_cache, current_cross_attention_cache),
            )

        decoder_token_ids = self.sampler(
            next=next,
            prompt=decoder_token_ids,
            cache=(self_attention_cache, cross_attention_cache),
            index=index,
            mask=keras.ops.cast(
                decoder_token_ids != self.preprocessor.tokenizer.pad_token_id
                if self.preprocessor is not None
                else decoder_padding_mask,
                dtype="bool",
            ),
            stop_token_ids=stop_token_ids,
            hidden_states=hidden_states,
            model=self,
        )

        if stop_token_ids is not None:
            end_locations = any_equal(
                decoder_token_ids,
                stop_token_ids,
                decoder_token_ids == self.preprocessor.tokenizer.pad_token_id
                if self.preprocessor is not None
                else False,
            )
            end_locations = keras.ops.cast(end_locations, "int32")
            cumsum = keras.ops.cumsum(end_locations, axis=-1)
            overflow = cumsum - end_locations
            decoder_padding_mask = keras.ops.logical_not(
                keras.ops.cast(overflow, "bool")
            )
        else:
            decoder_padding_mask = keras.ops.ones_like(
                decoder_token_ids, dtype="bool"
            )

        return {
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }
