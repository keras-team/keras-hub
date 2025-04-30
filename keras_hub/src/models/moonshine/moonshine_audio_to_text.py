import keras
import tensorflow as tf
from keras import tree

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.moonshine.moonshine_backbone import Arange
from keras_hub.src.models.moonshine.moonshine_backbone import MoonshineBackbone
from keras_hub.src.models.moonshine.moonshine_backbone import (
    compute_output_lengths,
)
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
    ```
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
        decoder_padding_mask=None,
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
                layers.
            decoder_padding_mask: Tensor, optional. Mask for decoder attention.

        Returns:
            Tuple: Tuple of (logits, hidden_states, self_attention_cache,
            cross_attention_cache).
        """
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
                    decoder_attention_mask=decoder_padding_mask,
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
                    decoder_attention_mask=decoder_padding_mask,
                    encoder_attention_mask=encoder_padding_mask,
                    training=False,
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
        logits = self.backbone.token_embedding(hidden_states, reverse=True)
        return (
            logits,
            hidden_states,
            self_attention_cache,
            cross_attention_cache,
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

        batch_size = keras.ops.shape(encoder_input_values)[0]
        # Calculate the length of the valid prompt before building the cache.
        row_lengths = keras.ops.sum(
            keras.ops.cast(decoder_padding_mask, "int32"),
            axis=-1,
        )
        index = keras.ops.min(row_lengths)
        # NOTE: For the JAX backend, pre-allocate the cache based on max_length.
        max_length = keras.ops.shape(decoder_token_ids)[1]

        encoder_hidden_states, encoder_attention_mask_for_decoder = (
            self.call_encoder(
                encoder_input_values=encoder_input_values,
                padding_mask=encoder_padding_mask,
            )
        )
        initial_decoder_token_ids = keras.ops.slice(
            decoder_token_ids, [0, 0], [batch_size, index]
        )
        initial_decoder_padding_mask = keras.ops.slice(
            decoder_padding_mask, [0, 0], [batch_size, index]
        )
        (
            _,
            hidden_states,
            init_self_attention_cache,
            init_cross_attention_cache,
        ) = self.call_decoder_with_cache(
            encoder_hidden_states=encoder_hidden_states,
            encoder_padding_mask=encoder_attention_mask_for_decoder,
            decoder_token_ids=initial_decoder_token_ids,
            self_attention_cache=None,
            cross_attention_cache=None,
            decoder_padding_mask=initial_decoder_padding_mask,
        )
        self_attention_cache = init_self_attention_cache
        cross_attention_cache = init_cross_attention_cache

        row_lengths = keras.ops.sum(
            keras.ops.cast(decoder_padding_mask, "int32"),
            axis=-1,
        )
        index = keras.ops.min(row_lengths)

        def next(prompt, cache, index):
            if isinstance(cache, tuple) and len(cache) == 1:
                cache = cache[0]
            elif isinstance(cache, tuple) and len(cache) == 0:
                cache = None
            cache_index = index - 1
            num_samples = keras.ops.shape(prompt)[0]
            next_token_input = keras.ops.slice(
                prompt, [0, cache_index], [num_samples, 1]
            )
            single_token_padding_mask = keras.ops.ones_like(
                next_token_input, dtype="bool"
            )

            def repeat_tensor(x):
                if keras.ops.shape(x)[0] == num_samples:
                    return x
                return keras.ops.repeat(
                    x, repeats=num_samples // batch_size, axis=0
                )

            logits, hidden_states, new_cache, _ = self.call_decoder_with_cache(
                encoder_hidden_states=repeat_tensor(encoder_hidden_states),
                encoder_padding_mask=repeat_tensor(
                    encoder_attention_mask_for_decoder
                ),
                decoder_token_ids=next_token_input,
                self_attention_cache=cache,
                self_attention_cache_update_index=cache_index,
                cross_attention_cache=repeat_tensor(cross_attention_cache),
                decoder_padding_mask=single_token_padding_mask,
            )
            return (
                logits[:, 0, :],
                hidden_states[:, 0, :],
                new_cache,
            )

        if keras.config.backend() == "jax":
            current_prompt = decoder_token_ids
            current_cache = self_attention_cache
            current_index = index
            for _ in range(max_length - index):
                if stop_token_ids is not None:
                    prompt_mask = keras.ops.cast(
                        current_prompt
                        == (
                            self.preprocessor.tokenizer.pad_token_id
                            if self.preprocessor
                            else -1
                        ),
                        dtype="bool",
                    )
                    valid_token_mask = ~prompt_mask
                    full_range = keras.ops.arange(max_length)
                    generated_range_mask = (full_range >= index) & (
                        full_range < current_index
                    )
                    check_mask = valid_token_mask & keras.ops.expand_dims(
                        generated_range_mask, 0
                    )
                    end_tokens = any_equal(
                        current_prompt, stop_token_ids, check_mask
                    )
                    prompt_done = keras.ops.any(end_tokens, axis=-1)
                    if keras.ops.all(prompt_done):
                        break

                logits, _, current_cache = next(
                    current_prompt, current_cache, current_index
                )
                probabilities = self.sampler.compute_probabilities(logits)
                next_token = self.sampler.get_next_token(probabilities)
                next_token = keras.ops.cast(next_token, current_prompt.dtype)
                next_token = next_token[:, None]
                current_prompt = keras.ops.slice_update(
                    current_prompt, [0, current_index], next_token
                )
                current_index += 1

            decoder_token_ids = current_prompt
        else:
            decoder_token_ids = self.sampler(
                next=next,
                prompt=decoder_token_ids,
                cache=self_attention_cache,
                index=index,
                mask=keras.ops.cast(
                    decoder_token_ids
                    != self.preprocessor.tokenizer.pad_token_id
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

    def make_generate_function(self):
        """Create or return the compiled generation function."""
        if self.generate_function is not None:
            return self.generate_function

        self.generate_function = self.generate_step
        if keras.config.backend() == "torch":
            import torch

            def wrapped_generate_function(
                inputs,
                stop_token_ids=None,
            ):
                with torch.no_grad():
                    return self.generate_step(inputs, stop_token_ids)

            self.generate_function = wrapped_generate_function
        elif keras.config.backend() == "tensorflow" and not self.run_eagerly:
            # `jit_compile` is a property of keras.Model after TF 2.12.
            # Use `getattr()` for backwards compatibility.
            # NOTE: Override, explicitly disabled JIT compilation for the
            # TensorFlow backend.
            self.generate_function = tf.function(
                self.generate_step, jit_compile=False
            )
        elif keras.config.backend() == "jax" and not self.run_eagerly:

            def wrapped_generate_function(
                inputs,
                stop_token_ids=None,
            ):
                inputs = tree.map_structure(keras.ops.convert_to_tensor, inputs)
                return self.generate_step(inputs, stop_token_ids)

            self.generate_function = wrapped_generate_function

        return self.generate_function
