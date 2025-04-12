import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.moonshine.moonshine_backbone import MoonshineBackbone
from keras_hub.src.models.moonshine.moonshine_seq_2_seq_lm_preprocessor import (
    MoonshineSeq2SeqLMPreprocessor,
)
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM
from keras_hub.src.utils.tensor_utils import any_equal
from keras import ops

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
        decoder_padding_mask=None,
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
                    decoder_attention_mask=decoder_padding_mask, # <<< PASS MASK HERE TOO
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
            # <<< START ADDITIONS >>>
            should_print = False
            if self_attention_cache_update_index is not None:
                if isinstance(self_attention_cache_update_index, int):
                    should_print = (self_attention_cache_update_index == 0 or self_attention_cache_update_index == 1)
                else:
                    should_print = ops.logical_or(
                        ops.equal(self_attention_cache_update_index, 0),
                        ops.equal(self_attention_cache_update_index, 1)
                    )
                    if hasattr(should_print, 'numpy'):
                        should_print = should_print.numpy()

            if should_print:
                print(f"--- [call_decoder_with_cache] ENTERED for index: {self_attention_cache_update_index} ---")
                try:
                    print(f"[call_decoder_with_cache]   decoder_token_ids (shape {ops.shape(decoder_token_ids)}): {ops.convert_to_numpy(decoder_token_ids)}")
                    print(f"[call_decoder_with_cache]   encoder_hidden_states shape: {ops.shape(encoder_hidden_states)}")
                    print(f"[call_decoder_with_cache]   encoder_padding_mask[:5]: {ops.convert_to_numpy(encoder_padding_mask[:, :5])}")
                    if self_attention_cache is not None:
                        print(f"[call_decoder_with_cache]   self_attention_cache shape: {ops.shape(self_attention_cache)}")
                    else:
                        print(f"[call_decoder_with_cache]   self_attention_cache: None")
                    if cross_attention_cache is not None:
                        print(f"[call_decoder_with_cache]   cross_attention_cache shape: {ops.shape(cross_attention_cache)}")
                    else:
                        print(f"[call_decoder_with_cache]   cross_attention_cache: None")
                    if decoder_padding_mask is not None:
                        print(f"[call_decoder_with_cache]   decoder_padding_mask: {ops.convert_to_numpy(decoder_padding_mask)}")
                    else:
                        print(f"[call_decoder_with_cache]   decoder_attention_mask: None")
                except Exception as e:
                    print(f"[call_decoder_with_cache]   Error printing inputs: {e}")
            # <<< END ADDITIONS >>>
            position = keras.ops.array(
                [self_attention_cache_update_index], dtype="int32"
            )
            # <<< START ADDITIONS >>>
            position_ids = ops.expand_dims(position, axis=0)
            batch_size = ops.shape(decoder_token_ids)[0]
            if batch_size > 1:
                 position_ids = ops.repeat(position_ids, batch_size, axis=0) # Shape (B, 1)

            if should_print:
                 try:
                     print(f"[call_decoder_with_cache]   position_ids for rotary (shape {ops.shape(position_ids)}): {ops.convert_to_numpy(position_ids)}")
                 except Exception as e:
                     print(f"[call_decoder_with_cache]   Error printing position_ids: {e}")
            # <<< END ADDITIONS >>>
            rotary_embedding = self.backbone.decoder_rotary_embedding(position)
            # <<< START ADDITIONS >>>
            if should_print:
                 try:
                     if isinstance(rotary_embedding, tuple):
                         print(f"[call_decoder_with_cache]   rotary_embedding[0] (cos) shape: {ops.shape(rotary_embedding[0])}")
                         print(f"[call_decoder_with_cache]   rotary_embedding[1] (sin) shape: {ops.shape(rotary_embedding[1])}")
                     else:
                         print(f"[call_decoder_with_cache]   rotary_embedding shape: {ops.shape(rotary_embedding)}")
                     print(f"[call_decoder_with_cache]   Initial x (token embedding) shape: {ops.shape(x)}")
                     print(f"[call_decoder_with_cache]   Initial x mean: {ops.mean(x):.4f}, max: {ops.max(x):.4f}, min: {ops.min(x):.4f}")
                 except Exception as e:
                     print(f"[call_decoder_with_cache]   Error printing rotary/x stats: {e}")
            # <<< END ADDITIONS >>>

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

                # <<< START ADDITIONS >>>
                if i == 0 and self_attention_cache_update_index == 0:
                     print(f"--- [call_decoder_with_cache, index 0] Before Decoder Layer {i} ---")
                     try:
                         print(f"[Layer {i}] Input x shape: {ops.shape(x)}")
                         print(f"[Layer {i}] Input x mean: {ops.mean(x):.4f}, max: {ops.max(x):.4f}, min: {ops.min(x):.4f}")
                         print(f"[Layer {i}] Input encoder_hidden_states shape: {ops.shape(encoder_hidden_states)}")
                         if isinstance(rotary_embedding, tuple):
                             print(f"[Layer {i}] Input rotary_embedding[0] (cos) shape: {ops.shape(rotary_embedding[0])}")
                         else:
                             print(f"[Layer {i}] Input rotary_embedding shape: {ops.shape(rotary_embedding)}")
                         print(f"[Layer {i}] Input cache_k shape: {ops.shape(cache_k)}")
                         print(f"[Layer {i}] Input cache_v shape: {ops.shape(cache_v)}")
                         print(f"[Layer {i}] Input x_attn_cache_k shape: {ops.shape(x_attn_cache_k)}")
                         print(f"[Layer {i}] Input x_attn_cache_v shape: {ops.shape(x_attn_cache_v)}")
                         print(f"[Layer {i}] Input cache_update_index: {self_attention_cache_update_index}")
                         print(f"[Layer {i}] Input encoder_padding_mask[:5]: {ops.convert_to_numpy(encoder_padding_mask[:, :5])}")
                     except Exception as e:
                         print(f"[Layer {i}] Error printing layer inputs: {e}")
                # <<< END ADDITIONS >>>

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
                # <<< START ADDITIONS >>>
                if i == 0 and self_attention_cache_update_index == 0:
                     print(f"--- [call_decoder_with_cache, index 0] After Decoder Layer {i} ---")
                     try:
                         print(f"[Layer {i}] Output x shape: {ops.shape(x)}")
                         print(f"[Layer {i}] Output x mean: {ops.mean(x):.4f}, max: {ops.max(x):.4f}, min: {ops.min(x):.4f}")
                         print(f"[Layer {i}] Output new_cache_k shape: {ops.shape(new_cache_k)}")
                         print(f"[Layer {i}] Output new_cache_v shape: {ops.shape(new_cache_v)}")
                     except Exception as e:
                         print(f"[Layer {i}] Error printing layer outputs: {e}")
                # <<< END ADDITIONS >>>
                # Update self-attention cache.
                new_self_cache = keras.ops.stack(
                    [new_cache_k, new_cache_v], axis=1
                )
                self_attention_caches.append(new_self_cache)

            # <<< START ADDITIONS >>>
            if should_print:
                print(f"--- [call_decoder_with_cache, index {self_attention_cache_update_index}] After Decoder Loop ---")
                try:
                    print(f"[AfterLoop] Final x shape: {ops.shape(x)}")
                    print(f"[AfterLoop] Final x mean: {ops.mean(x):.4f}, max: {ops.max(x):.4f}, min: {ops.min(x):.4f}")
                except Exception as e:
                    print(f"[AfterLoop] Error printing stats: {e}")
            # <<< END ADDITIONS >>>
            # [batch_size, num_layers, 2, seq_len, num_heads, head_dim].
            self_attention_cache = keras.ops.stack(
                self_attention_caches, axis=1
            )

        hidden_states = self.backbone.decoder_post_norm(x)
        # <<< START ADDITIONS >>>
        if 'should_print' in locals() and should_print:
            print(f"--- [call_decoder_with_cache, index {self_attention_cache_update_index}] After Post Norm ---")
            try:
                print(f"[AfterNorm] hidden_states shape: {ops.shape(hidden_states)}")
                print(f"[AfterNorm] hidden_states mean: {ops.mean(hidden_states):.4f}, max: {ops.max(hidden_states):.4f}, min: {ops.min(hidden_states):.4f}")
            except Exception as e:
                print(f"[AfterNorm] Error printing stats: {e}")
        # <<< END ADDITIONS >>>
        logits = self.backbone.logits(hidden_states)
        # <<< START ADDITIONS >>>
        if 'should_print' in locals() and should_print:
            print(f"--- [call_decoder_with_cache, index {self_attention_cache_update_index}] After Logits ---")
            try:
                squeezed_logits = logits
                if len(ops.shape(logits)) == 3 and ops.shape(logits)[1] == 1:
                    squeezed_logits = ops.squeeze(logits, axis=1)

                if len(ops.shape(squeezed_logits)) == 2:
                    top_k_logits, top_k_indices = ops.top_k(squeezed_logits[0, :], k=5) # Check first batch item
                    print(f"[AfterLogits] Top 5 predicted token IDs: {ops.convert_to_numpy(top_k_indices)}")
                    print(f"[AfterLogits] Top 5 predicted logits: {ops.convert_to_numpy(top_k_logits)}")
                else:
                    print(f"[AfterLogits] Unexpected logits shape after squeeze: {ops.shape(squeezed_logits)}")
            except Exception as e:
                print(f"[AfterLogits] Error printing top logits: {e}")
            print(f"--- [call_decoder_with_cache] EXITED for index: {self_attention_cache_update_index} ---")
        # <<< END ADDITIONS >>>

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
        if hasattr(self.backbone, 'encoder_dropout'):
            x = self.backbone.encoder_dropout(x, training=False)
        for transformer_layer in self.backbone.encoder_blocks:
            x = transformer_layer(
                inputs=x,
                rotary_embedding=rotary_embedding,
                attention_mask=padding_mask,
                training=False,
            )
        if hasattr(self.backbone, 'encoder_final_layer_norm'):
            x = self.backbone.encoder_final_layer_norm(x)
        return x


    def _build_cache(
        self,
        audio_inputs,
        audio_padding_mask,
        decoder_token_ids,
    ):
        """Initialize and populate attention caches with encoder and decoder
        outputs."""
        encoder_hidden_states = self.call_encoder(
            audio_inputs, padding_mask=audio_padding_mask
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
        # Calculate the length of the valid prompt before building the cache.
        row_lengths = keras.ops.sum(
            keras.ops.cast(decoder_padding_mask, "int32"),
            axis=-1,
        )
        index = keras.ops.min(row_lengths)

        encoder_hidden_states = self.call_encoder(
            encoder_input_values=encoder_input_values,
            padding_mask=encoder_padding_mask,
        )
        initial_decoder_token_ids = decoder_token_ids[:, :index]
        initial_decoder_padding_mask = decoder_padding_mask[:, :index]
        (
            _,
            hidden_states,
            init_self_attention_cache,
            init_cross_attention_cache,
        ) = self.call_decoder_with_cache(
            encoder_hidden_states=encoder_hidden_states,
            encoder_padding_mask=encoder_padding_mask,
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
            cache_index = index - 1
            num_samples = keras.ops.shape(prompt)[0]
            next_token_input = keras.ops.slice(prompt, [0, cache_index], [num_samples, 1])
            single_token_padding_mask = ops.ones_like(next_token_input, dtype="bool")

            def repeat_tensor(x):
                if keras.ops.shape(x)[0] == num_samples:
                    return x
                return keras.ops.repeat(
                    x, repeats=num_samples // batch_size, axis=0
                )

            logits, hidden_states, new_cache, _ = self.call_decoder_with_cache(
                encoder_hidden_states=repeat_tensor(encoder_hidden_states),
                encoder_padding_mask=repeat_tensor(encoder_padding_mask),
                decoder_token_ids=next_token_input,
                self_attention_cache=cache,
                self_attention_cache_update_index=cache_index,
                cross_attention_cache=repeat_tensor(cross_attention_cache),
                decoder_padding_mask=single_token_padding_mask,
            )
            # <<< START ADDITIONS >>>
            if cache_index < 2:
                print(f"--- [generate_step -> next] index {cache_index} ---")
                try:
                    print(f"  Input prompt shape to call_decoder: {ops.shape(prompt)}")
                    print(f"  Output logits shape: {ops.shape(logits)}")
                    print(f"  Output hidden_states shape: {ops.shape(hidden_states)}")
                    print(f"  Output new_cache shape: {ops.shape(new_cache)}")
                    squeezed_logits = ops.squeeze(logits, axis=1)
                    if len(ops.shape(squeezed_logits)) == 2:
                        top_k_logits, top_k_indices = ops.top_k(squeezed_logits[0, :], k=5)
                        print(f"  Top 5 predicted token IDs: {ops.convert_to_numpy(top_k_indices)}")
                        print(f"  Top 5 predicted logits: {ops.convert_to_numpy(top_k_logits)}")
                except Exception as e:
                    print(f"  Error printing next() debug info: {e}")
            # <<< END ADDITIONS >>>
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
            mask=keras.ops.cast(
                decoder_token_ids != self.preprocessor.tokenizer.pad_token_id
                if self.preprocessor is not None
                else decoder_padding_mask,
            dtype="bool"),
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

        # <<< START ADDITIONS >>>
        print("--- [generate_step] Final Output ---")
        try:
            print(f"  Final decoder_token_ids shape: {ops.shape(decoder_token_ids)}")
            print(f"  Final decoder_token_ids (first batch): {ops.convert_to_numpy(decoder_token_ids[0])}")
            print(f"  Final decoder_padding_mask shape: {ops.shape(decoder_padding_mask)}")
            print(f"  Final decoder_padding_mask (first batch): {ops.convert_to_numpy(decoder_padding_mask[0])}")
        except Exception as e:
            print(f"  Error printing final generate_step info: {e}")
        # <<< END ADDITIONS >>>

        return {
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }
