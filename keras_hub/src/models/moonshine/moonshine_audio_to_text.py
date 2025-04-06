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
        self_attention_cache,
        self_attention_cache_update_index,
        decoder_attention_mask=None,
    ):
        """Process decoder inputs with attention caching for efficient
        generation.

        Args:
            encoder_hidden_states: Tensor. Encoder outputs.
            encoder_padding_mask: Tensor. Padding mask for encoder outputs.
            decoder_token_ids: Tensor. Decoder input token IDs.
            self_attention_cache: Tensor. Cache for self-attention layers.
            self_attention_cache_update_index: int. Index for cache updates.
            decoder_attention_mask: Tensor, optional. Mask for decoder attention

        Returns:
            Tuple of (logits, hidden_states, self_attention_cache).
        """
        x = self.backbone.token_embedding(decoder_token_ids)
        position = keras.ops.array(
            [self_attention_cache_update_index], dtype="int32"
        )
        rotary_embedding = self.backbone.decoder_rotary_embedding(position)
        updated_cache = []
        for i, layer in enumerate(self.backbone.decoder_blocks):
            current_cache = self_attention_cache[:, i, ...]
            x, next_cache = layer(
                inputs=[x, encoder_hidden_states, rotary_embedding],
                self_attention_cache=current_cache,
                self_attention_cache_update_index=self_attention_cache_update_index,
                decoder_attention_mask=decoder_attention_mask,
                encoder_attention_mask=encoder_padding_mask,
            )
            updated_cache.append(next_cache)
        self_attention_cache = keras.ops.stack(updated_cache, axis=1)
        hidden_states = self.backbone.decoder_post_norm(x)
        logits = self.backbone.logits(hidden_states)
        return logits, hidden_states, self_attention_cache

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

        encoder_hidden_states = self.call_encoder(
            encoder_input_values=encoder_input_values,
            padding_mask=encoder_padding_mask,
        )
        seq_len = keras.ops.shape(decoder_token_ids)[1]
        positions = keras.ops.arange(0, seq_len, dtype="int32")
        rotary_embedding = self.backbone.decoder_rotary_embedding(positions)
        x = self.backbone.token_embedding(decoder_token_ids)
        self_attention_caches = []
        for layer in self.backbone.decoder_blocks:
            x, cache = layer(
                inputs=[x, encoder_hidden_states, rotary_embedding],
                decoder_attention_mask=None,
                encoder_attention_mask=encoder_padding_mask,
            )
            self_attention_caches.append(cache)
        self_attention_cache = keras.ops.stack(self_attention_caches, axis=1)
        hidden_states = self.backbone.decoder_post_norm(x)

        row_lengths = keras.ops.sum(
            keras.ops.cast(decoder_padding_mask, "int32"), axis=-1
        )
        index = keras.ops.min(row_lengths)

        def next(prompt, cache, index):
            cache_index = index - 1
            num_samples = keras.ops.shape(prompt)[0]
            prompt = keras.ops.slice(prompt, [0, cache_index], [num_samples, 1])
            logits, hidden_states, new_cache = self.call_decoder_with_cache(
                encoder_hidden_states=encoder_hidden_states,
                encoder_padding_mask=encoder_padding_mask,
                decoder_token_ids=prompt,
                self_attention_cache=cache,
                self_attention_cache_update_index=cache_index,
                decoder_attention_mask=None,
            )
            return (
                keras.ops.squeeze(logits, axis=1),
                keras.ops.squeeze(hidden_states, axis=1),
                new_cache,
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
