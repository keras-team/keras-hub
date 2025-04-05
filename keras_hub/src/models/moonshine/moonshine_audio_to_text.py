import warnings

import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.moonshine.moonshine_backbone import MoonshineBackbone
from keras_hub.src.models.moonshine.moonshine_seq_2_seq_lm_preprocessor import (
    MoonshineSeq2SeqLMPreprocessor,
)
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
from keras_hub.src.utils.tensor_utils import any_equal


@keras_hub_export("keras_hub.models.MoonshineAudioToText")
class MoonshineAudioToText(Seq2SeqLM):
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
        preprocessor: A `keras_hub.models.MoonshineSeq2SeqLMPreprocessor` or
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
    """

    # References:
    # Defined and formulated based on the Hugging Face implementation of the
    # MoonshineForConditionalGeneration class (https://github.com/huggingface/transformers/blob/dcbdf7e962c4b36140cc9ee76f870016121e69e5/src/transformers/models/moonshine/modeling_moonshine.py#L1509-L1626).

    backbone_cls = MoonshineBackbone
    preprocessor_cls = MoonshineSeq2SeqLMPreprocessor

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
        decoder_attention_mask=None,
    ):
        """Process decoder inputs with attention caching for efficient
        generation."""
        tokens = self.backbone.token_embedding(decoder_token_ids)
        x = tokens

        # Cache management for audio-to-text generation.
        self_attention_caches = []
        cross_attention_caches = []

        # Determine if this is initialization or generation.
        if self_attention_cache_update_index is None:
            # Initialization: Process full sequence, compute caches.
            seq_len = keras.ops.shape(decoder_token_ids)[1]
            positions = keras.ops.arange(0, seq_len, dtype="int32")
            rotary_embedding = self.backbone.decoder_rotary_embedding(positions)

            self_attention_caches = []
            cross_attention_caches = []
            for layer in self.backbone.decoder_blocks:
                x, cache_k, cache_v, x_attn_cache_k, x_attn_cache_v = layer(
                    [x, encoder_hidden_states, rotary_embedding],
                    use_cache=False,
                    decoder_attention_mask=None,
                    encoder_attention_mask=encoder_padding_mask,
                )
                # Stack key and value for each layer.
                self_attention_caches.append(
                    keras.ops.stack([cache_k, cache_v], axis=1)
                )
                cross_attention_caches.append(
                    keras.ops.stack([x_attn_cache_k, x_attn_cache_v], axis=1)
                )
            self_attention_cache = keras.ops.stack(
                self_attention_caches, axis=1
            )
            cross_attention_cache = keras.ops.stack(
                cross_attention_caches, axis=1
            )

        else:
            position = keras.ops.array(
                [self_attention_cache_update_index], dtype="int32"
            )
            rotary_embedding = self.backbone.decoder_rotary_embedding(position)

            for i, layer in enumerate(self.backbone.decoder_blocks):
                # [batch_size, 2, seq_len, num_heads, head_dim].
                current_self_cache = self_attention_cache[:, i, :, :, :, :]
                cache_k = current_self_cache[
                    :, 0, :, :, :
                ]  # [batch_size, seq_len, num_heads, head_dim]
                cache_v = current_self_cache[
                    :, 1, :, :, :
                ]  # [batch_size, seq_len, num_heads, head_dim]
                # [batch_size, 2, context_len, num_heads, head_dim].
                current_cross_cache = cross_attention_cache[:, i, :, :, :, :]
                x_attn_cache_k = current_cross_cache[
                    :, 0, :, :, :
                ]  # [batch_size, context_len, num_heads, head_dim]
                x_attn_cache_v = current_cross_cache[
                    :, 1, :, :, :
                ]  # [batch_size, context_len, num_heads, head_dim]

                # Call layer with 7 inputs.
                x, new_cache_k, new_cache_v = layer(
                    [
                        x,
                        encoder_hidden_states,
                        cache_k,
                        cache_v,
                        x_attn_cache_k,
                        x_attn_cache_v,
                        rotary_embedding,
                    ],
                    use_cache=True,
                    decoder_attention_mask=decoder_attention_mask,
                    encoder_attention_mask=encoder_padding_mask,
                )
                # Update self-attention cache.
                new_self_cache = keras.ops.stack(
                    [new_cache_k, new_cache_v], axis=1
                )
                self_attention_caches.append(new_self_cache)

            # [batch_size, num_layers, 2, seq_len, num_heads, head_dim].
            self_attention_cache = keras.ops.stack(
                self_attention_caches, axis=1
            )

        hidden_states = self.backbone.decoder_post_norm(x)
        logits = self.backbone.logits(hidden_states)

        return (
            logits,
            hidden_states,
            self_attention_cache,
            cross_attention_cache,
        )

    def call_encoder(self, encoder_input_values, padding_mask):
        """Process audio input through the encoder stack."""
        x = encoder_input_values
        seq_length = keras.ops.shape(x)[1]
        positions = keras.ops.arange(0, seq_length, dtype="int32")
        rotary_embedding = self.backbone.encoder_rotary_embedding(positions)
        for transformer_layer in self.backbone.encoder_blocks:
            x = transformer_layer(
                x, rotary_embedding=rotary_embedding, padding_mask=padding_mask
            )
        return x

    def _initialize_cache(self, encoder_input_values, max_sequence_length=1024):
        """Create empty self-attention and cross-attention caches for
        efficient generation."""
        batch_size = keras.ops.shape(encoder_input_values)[0]
        audio_max_length = keras.ops.shape(encoder_input_values)[1]

        num_layers = self.backbone.decoder_num_layers
        num_heads = self.backbone.decoder_num_heads
        head_dim = self.backbone.hidden_dim // self.backbone.decoder_num_heads

        self_attention_cache_shape = [
            batch_size,
            num_layers,
            2,
            max_sequence_length,
            num_heads,
            head_dim,
        ]
        cross_attention_cache_shape = [
            batch_size,
            num_layers,
            2,
            audio_max_length,
            num_heads,
            head_dim,
        ]

        self_attention_cache = keras.ops.zeros(
            self_attention_cache_shape, dtype=self.compute_dtype
        )
        cross_attention_cache = keras.ops.zeros(
            cross_attention_cache_shape, dtype=self.compute_dtype
        )

        return self_attention_cache, cross_attention_cache

    def _build_cache(
        self,
        audio_inputs,
        audio_padding_mask,
        decoder_token_ids,
        max_sequence_length=1024,
    ):
        """Initialize and populate attention caches with encoder and decoder
        outputs."""
        seq_len = decoder_token_ids.shape[1]
        if seq_len > self.max_sequence_length:
            warnings.warn(
                f"Prompt sequence length {seq_len} exceeds maximum sequence "
                f"length {self.max_sequence_length}. Truncating to "
                f"{self.max_sequence_length} tokens."
            )
        encoder_hidden_states = self.call_encoder(
            audio_inputs, padding_mask=audio_padding_mask
        )
        self_attention_cache, cross_attention_cache = self._initialize_cache(
            audio_inputs, max_sequence_length=max_sequence_length
        )
        _, hidden_states, self_attention_cache, cross_attention_cache = (
            self.call_decoder_with_cache(
                encoder_hidden_states=encoder_hidden_states,
                encoder_padding_mask=audio_padding_mask,
                decoder_token_ids=decoder_token_ids,
                self_attention_cache=None,
                cross_attention_cache=None,
            )
        )
        return (
            hidden_states,
            encoder_hidden_states,
            self_attention_cache,
            cross_attention_cache,
        )

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

        batch_size = keras.ops.shape(encoder_input_values)[0]
        # max_sequence_length is set to a fixed value (1024) to pre-allocate
        # memory for attention caches.
        # Taken to be equal to the default for the BART backbone, 1024.
        max_sequence_length = 1024

        encoder_hidden_states = self.call_encoder(
            encoder_input_values=encoder_input_values,
            padding_mask=encoder_padding_mask,
        )
        self_attention_cache, cross_attention_cache = self._initialize_cache(
            encoder_input_values, max_sequence_length=max_sequence_length
        )
        (
            _,
            hidden_states,
            init_self_attention_cache,
            init_cross_attention_cache,
        ) = self.call_decoder_with_cache(
            encoder_hidden_states=encoder_hidden_states,
            encoder_padding_mask=encoder_padding_mask,
            decoder_token_ids=decoder_token_ids,
            self_attention_cache=None,
            cross_attention_cache=None,
        )
        # Get the full shape of init_self_attention_cache dynamically.
        cache_shape = keras.ops.shape(init_self_attention_cache)
        seq_len = keras.ops.shape(decoder_token_ids)[1]
        slice_sizes = [
            cache_shape[0],  # batch_size
            cache_shape[1],  # num_layers
            cache_shape[2],  # 2 (key/value)
            seq_len,  # sequence length from decoder_token_ids
            cache_shape[4],  # num_heads
            cache_shape[5],  # head_dim
        ]
        self_attention_cache = keras.ops.slice_update(
            self_attention_cache,
            [0, 0, 0, 0, 0, 0],
            keras.ops.slice(
                init_self_attention_cache,
                [0, 0, 0, 0, 0, 0],
                slice_sizes,
            ),
        )

        row_lengths = keras.ops.sum(
            keras.ops.cast(decoder_padding_mask, "int32"),
            axis=-1,
        )
        index = keras.ops.min(row_lengths)

        def next(prompt, cache, index):
            cache_index = index - 1
            num_samples = keras.ops.shape(prompt)[0]
            prompt = keras.ops.slice(prompt, [0, cache_index], [num_samples, 1])
            # Create attention mask: True for positions 0 to cache_index, False
            # beyond.
            # Use max_sequence_length + 1 to match causal mask length in
            # autoregressive mode.
            attention_mask = (
                keras.ops.arange(max_sequence_length + 1) <= cache_index
            )
            attention_mask = keras.ops.expand_dims(
                attention_mask, axis=0
            )  # [1, max_sequence_length + 1]
            attention_mask = keras.ops.repeat(
                attention_mask, num_samples, axis=0
            )  # [batch_size, max_sequence_length + 1]

            def repeat_tensor(x):
                if keras.ops.shape(x)[0] == num_samples:
                    return x
                return keras.ops.repeat(
                    x, repeats=num_samples // batch_size, axis=0
                )

            logits, hidden_states, new_cache, _ = self.call_decoder_with_cache(
                encoder_hidden_states=repeat_tensor(encoder_hidden_states),
                encoder_padding_mask=repeat_tensor(encoder_padding_mask),
                decoder_token_ids=prompt,
                self_attention_cache=cache,
                self_attention_cache_update_index=cache_index,
                cross_attention_cache=repeat_tensor(cross_attention_cache),
                decoder_attention_mask=attention_mask,
            )
            # Get the full shape of new_cache dynamically.
            new_cache_shape = keras.ops.shape(new_cache)
            # Define slice sizes with explicit positive values, 1 for the
            # sequence dimension.
            new_cache_slice_sizes = [
                new_cache_shape[0],  # batch_size
                new_cache_shape[1],  # num_layers
                new_cache_shape[2],  # 2 (key/value)
                1,  # single token
                new_cache_shape[4],  # num_heads
                new_cache_shape[5],  # head_dim
            ]
            # Extract only the new token's cache.
            new_cache_slice = keras.ops.slice(
                new_cache,
                [0, 0, 0, cache_index, 0, 0],
                new_cache_slice_sizes,
            )
            # Update the cache at the current index with the single-token slice.
            updated_cache = keras.ops.slice_update(
                cache,
                [0, 0, 0, cache_index, 0, 0],
                new_cache_slice,
            )
            return (
                keras.ops.squeeze(logits, axis=1),
                keras.ops.squeeze(hidden_states, axis=1),
                updated_cache,
            )

        decoder_token_ids = self.sampler(
            next=next,
            prompt=decoder_token_ids,
            cache=self_attention_cache,
            index=index,
            mask=decoder_token_ids != self.preprocessor.tokenizer.pad_token_id
            if self.preprocessor is not None
            else decoder_padding_mask,
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
