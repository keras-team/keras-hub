import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.moonshine.moonshine_audio_converter import (
    MoonshineAudioConverter,
)
from keras_hub.src.models.moonshine.moonshine_backbone import MoonshineBackbone
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM


class NotEqualMaskLayer(keras.layers.Layer):
    def call(self, inputs):
        return keras.ops.cast(keras.ops.not_equal(inputs, 0), "bool")


@keras_hub_export("keras_hub.models.MoonshineForConditionalGeneration")
class MoonshineForConditionalGeneration(Seq2SeqLM):
    """
    Moonshine model for conditional generation, tailored for Automatic Speech
    Recognition (ASR).

    This task model integrates audio preprocessing and sequence generation for
    end-to-end ASR. It processes raw audio waveforms using
    `MoonshineAudioConverter`, passes the features through the
    `MoonshineBackbone` encoder-decoder, and generates token sequences using a
    greedy decoding strategy.

    Args:
        backbone: `MoonshineBackbone`. The backbone model handling
            encoder-decoder transformations.
        audio_converter: `MoonshineAudioConverter`. Preprocessor for converting
            raw audio to features.
        tokenizer: `MoonshineTokenizer`. Tokenizer for converting token IDs to
            text.

    Examples:
    ```python
    import keras
    import numpy as np
    from keras_hub.models import MoonshineForConditionalGeneration
    from keras_hub.tokenizers import MoonshineTokenizer
    from keras_hub.models import MoonshineBackbone

    # Initialize model components.
    tokenizer = MoonshineTokenizer(
        "keras_hub/src/tests/test_data/llama2_tokenizer_full.spm",
        name="moonshine_tokenizer",
    )
    backbone = MoonshineBackbone(
        vocabulary_size=10000,
        encoder_num_layers=6,
        decoder_num_layers=6,
        hidden_dim=256,
        intermediate_dim=512,
        encoder_num_heads=8,
        decoder_num_heads=8,
        feedforward_expansion_factor=4,
        encoder_use_swiglu_activation=False,
        decoder_use_swiglu_activation=True,
        name="moonshine_backbone",
    )
    preprocessor = MoonshineAudioConverter(
        filter_dim=256, name="moonshine_audio_converter"
    )
    model = MoonshineForConditionalGeneration(
        backbone=backbone,
        audio_converter=preprocessor,
        tokenizer=tokenizer,
        name="moonshine_for_conditional_generation",
    )

    # Prepare a sample audio input (16kHz, mono).
    audio = np.random.randn(16000).astype("float32")  # 1-second audio sample
    audio = keras.ops.expand_dims(audio, axis=0)  # Shape: (1, 16000)

    # Generate transcription.
    # The outputs would be gibberish here since the components are randomly
    # initialized.
    token_ids = model.generate(audio)
    transcription = model.tokenizer.detokenize(token_ids[0])
    print("Transcription:", transcription)

    # Training example.
    x = {
        "audio": audio,
        "token_ids": keras.ops.array([[1, 2, 3]], dtype="int32"),
    }
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    y = keras.ops.array([[2, 3, 4]], dtype="int32")  # Shifted token IDs
    model.fit(x, y, epochs=1)

    # Print model summary.
    model.summary()
    ```
    """

    # References:
    # Defined and formulated based on the Hugging Face implementation of the
    # MoonshineForConditionalGeneration class (https://github.com/huggingface/transformers/blob/dcbdf7e962c4b36140cc9ee76f870016121e69e5/src/transformers/models/moonshine/modeling_moonshine.py#L1509-L1626).

    backbone_cls = MoonshineBackbone
    preprocessor_cls = MoonshineAudioConverter

    def __init__(
        self,
        backbone,
        audio_converter,
        decoder_start_token_id=1,
        end_token_id=2,
        pad_token_id=0,
        **kwargs,
    ):
        # === Layers ===
        self.backbone = backbone
        self.audio_converter = audio_converter
        self.decoder_start_token_id = decoder_start_token_id
        self.end_token_id = end_token_id
        self.pad_token_id = pad_token_id

        # === Functional Model ===
        audio_input = keras.Input(shape=(None,), name="audio", dtype="float32")
        token_ids_input = keras.Input(
            shape=(None,), name="token_ids", dtype="int32"
        )

        # Preprocess audio.
        audio_features = self.audio_converter(audio_input)
        encoder_input = audio_features["input_values"]
        encoder_mask = audio_features["attention_mask"]

        # Prepare decoder inputs.
        decoder_mask = NotEqualMaskLayer(name="decoder_mask")(token_ids_input)

        # Backbone forward pass.
        backbone_outputs = self.backbone(
            {
                "encoder_input_values": encoder_input,
                "decoder_token_ids": token_ids_input,
                "encoder_padding_mask": encoder_mask,
                "decoder_padding_mask": decoder_mask,
            }
        )
        logits = self.backbone.logits(
            backbone_outputs["decoder_sequence_output"]
        )
        super().__init__(
            inputs={
                "audio": audio_input,
                "token_ids": token_ids_input,
            },
            outputs=logits,
            **kwargs,
        )

    # Source: https://github.com/huggingface/transformers/blob/9e94801146ceeb3b215bbdb9492be74d7d7b7210/src/transformers/generation/utils.py#L1970-L2463
    def generate(
        self,
        audio_inputs,
        max_new_tokens=100,
        min_new_tokens=0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        temperature=1.0,
        top_k=0,
        do_sample=False,
        num_return_sequences=1,
        debug=False,
    ):
        """
        Generates token sequences from audio inputs using a greedy decoding
        strategy.

        This method processes raw audio input through the audio converter,
        encodes it using the backbone encoder, and incrementally generates token
        sequences using the decoder. It supports various decoding strategies and
        constraints to control the generation process.

        Args:
            audio_inputs: Array-like audio input of shape `(batch_size,
                sequence_length)`. Raw audio waveform, typically sampled at
                16kHz.
            max_new_tokens: int. Maximum number of tokens to generate. Defaults
                to 100.
            min_new_tokens: int. Minimum number of tokens to generate before
                allowing EOS token to be generated. Defaults to 0.
            repetition_penalty: float. Penalty applied to tokens that have
                already appeared in the generated sequence. Values > 1.0
                discourage repetition. Defaults to 1.0.
            no_repeat_ngram_size: int. Size of n-grams to prevent from
                repeating. If > 0, prevents any n-gram of this size from
                appearing twice. Defaults to 0.
            temperature: float. Value used to modulate next-token probability
                distribution. Lower values make the distribution more peaked
                (more deterministic), while higher values make it more uniform
                (more random). Defaults to 1.0.
            top_k: int. If > 0, only the top-k tokens with the highest
                probability are considered for next token selection. If 0, all
                tokens are considered. Defaults to 0.
            do_sample: bool. If `True`, sample next tokens according to their
                probability distribution. If `False`, select the token with
                highest probability. Defaults to `False`.
            num_return_sequences: int. Number of independently generated
                sequences to return for each input sequence. Defaults to 1.
            debug: bool. If `True`, print debugging information
                during generation. Defaults to `False`.

        Returns:
            Tensor: Tensor of shape `(batch_size * num_return_sequences,
                sequence_length)` containing the generated token IDs, where
                sequence_length varies based on when the EOS token is generated
                or max_new_tokens is reached. Sequences are padded with the pad
                token ID.
        """
        # Handle multiple sequences if specified.
        if num_return_sequences > 1:
            audio_inputs = keras.ops.repeat(
                audio_inputs, num_return_sequences, axis=0
            )
        batch_size = keras.ops.shape(audio_inputs)[0]
        # Preprocess audio inputs.
        audio_features = self.audio_converter(audio_inputs)
        encoder_input_values = audio_features["input_values"]
        encoder_padding_mask = audio_features["attention_mask"]
        # Step 1: Compute encoder outputs using dummy decoder inputs.
        dummy_decoder_token_ids = keras.ops.zeros(
            (batch_size, 1), dtype="int32"
        )
        dummy_decoder_padding_mask = keras.ops.zeros(
            (batch_size, 1), dtype="bool"
        )
        outputs = self.backbone(
            {
                "encoder_input_values": encoder_input_values,
                "decoder_token_ids": dummy_decoder_token_ids,
                "encoder_padding_mask": encoder_padding_mask,
                "decoder_padding_mask": dummy_decoder_padding_mask,
            },
            training=False,
        )
        encoder_outputs = outputs["encoder_sequence_output"]
        # Step 2: Precompute cross-attention caches for each decoder block.
        cross_attention_caches = []
        for decoder_block in self.backbone.decoder_blocks:
            dummy_query = keras.ops.zeros(
                (batch_size, 1, self.backbone.hidden_dim), dtype=self.dtype
            )
            _, x_attn_cache_k, x_attn_cache_v = decoder_block.cross_attention(
                query=dummy_query,
                key=encoder_outputs,
                value=encoder_outputs,
                attention_mask=encoder_padding_mask,
                training=False,
            )
            cross_attention_caches.append((x_attn_cache_k, x_attn_cache_v))
        # Step 3: Initialize generation with start token.
        generated_ids = keras.ops.full(
            (batch_size, 1), self.decoder_start_token_id, dtype="int32"
        )
        finished = keras.ops.zeros((batch_size,), dtype="bool")
        self_attention_caches = [(None, None)] * len(
            self.backbone.decoder_blocks
        )
        # Initialize appeared tokens for repetition penalty.
        if repetition_penalty != 1.0:
            appeared_tokens = [
                {int(keras.ops.convert_to_numpy(generated_ids[b, 0]))}
                for b in range(batch_size)
            ]
        if no_repeat_ngram_size > 0:
            banned_ngrams = [{} for _ in range(batch_size)]
        sequence_lengths = keras.ops.ones((batch_size,), dtype="int32")
        # Step 4: Generate tokens incrementally.
        for step in range(max_new_tokens):
            # Embed the last generated token.
            current_token_ids = generated_ids[:, -1:]  # Shape: (batch_size, 1)
            decoder_input = self.backbone.token_embedding(current_token_ids)
            # Compute rotary embedding for the current position.
            position = keras.ops.convert_to_tensor([step], dtype="int32")
            rotary_emb = self.backbone.decoder_rotary_embedding(position)
            # Process through decoder blocks with caching.
            x = decoder_input
            new_self_attention_caches = []
            for i, decoder_block in enumerate(self.backbone.decoder_blocks):
                cache_k, cache_v = self_attention_caches[i]
                x_attn_cache_k, x_attn_cache_v = cross_attention_caches[i]
                x, new_cache_k, new_cache_v = decoder_block(
                    [
                        x,
                        encoder_outputs,
                        cache_k,
                        cache_v,
                        x_attn_cache_k,
                        x_attn_cache_v,
                        rotary_emb,
                    ],
                    use_cache=True,
                    encoder_attention_mask=encoder_padding_mask,
                    training=False,
                )
                new_self_attention_caches.append((new_cache_k, new_cache_v))
            # Compute logits.
            x = self.backbone.decoder_post_norm(x)
            # Compute logits and select next token (greedy decoding).
            logits = self.backbone.logits(
                x
            )  # Shape: (batch_size, 1, vocab_size)
            # Apply logit modifications (temperature, penalties, etc.).
            if step < min_new_tokens:
                logits = keras.ops.where(
                    keras.ops.equal(
                        keras.ops.arange(logits.shape[-1]), self.end_token_id
                    ),
                    -float("inf"),
                    logits,
                )
            # Apply temperature scaling.
            if temperature != 1.0:
                logits = logits / temperature
            # Apply repetition penalty.
            if repetition_penalty != 1.0:
                logits_np = keras.ops.convert_to_numpy(logits)
                for b in range(batch_size):
                    if not keras.ops.convert_to_numpy(finished[b]):
                        for token_id in appeared_tokens[b]:
                            logits_np[b, 0, token_id] /= repetition_penalty
                logits = keras.ops.convert_to_tensor(
                    logits_np, dtype=logits.dtype
                )
            # Prevent repeating n-grams.
            if no_repeat_ngram_size > 0 and step >= no_repeat_ngram_size - 1:
                logits_np = keras.ops.convert_to_numpy(logits)
                for b in range(batch_size):
                    if not keras.ops.convert_to_numpy(finished[b]):
                        prev_ngram = generated_ids[
                            b, step - no_repeat_ngram_size + 1 : step + 1
                        ]
                        prev_ngram = tuple(
                            int(t)
                            for t in keras.ops.convert_to_numpy(prev_ngram)
                        )
                        for banned_token_id in banned_ngrams[b].get(
                            prev_ngram, []
                        ):
                            logits_np[b, 0, banned_token_id] = -float("inf")
                logits = keras.ops.convert_to_tensor(
                    logits_np, dtype=logits.dtype
                )
            # Apply top-k filtering.
            if top_k > 0:
                top_k_values, _ = keras.ops.top_k(logits[:, 0, :], k=top_k)
                min_values = top_k_values[:, -1:]
                logits = keras.ops.where(
                    logits[:, 0, :] < min_values,
                    -10000.0,
                    logits[:, 0, :],
                )
                logits = keras.ops.expand_dims(logits, axis=1)
            if do_sample:
                probs = keras.ops.softmax(logits, axis=-1)
                probs = keras.ops.squeeze(
                    probs, axis=1
                )  # Shape: (batch_size, vocab_size)
                next_token_ids = keras.random.categorical(probs, num_samples=1)
            else:
                next_token_ids = keras.ops.argmax(
                    logits, axis=-1
                )  # Shape: (batch_size, 1)
            # Add debugging prints.
            if debug:
                token_id = int(keras.ops.convert_to_numpy(next_token_ids[0, 0]))
                print(
                    f"Step {step}: Generated token ID for sequence: {token_id}"
                )
            force_finish = keras.ops.zeros((batch_size,), dtype="bool")
            if step > 20:
                last_10 = generated_ids[:, -10:]
                prev_10 = generated_ids[:, -20:-10]
                repetition_mask = keras.ops.all(last_10 == prev_10, axis=-1)
                force_finish = repetition_mask
            next_token_ids = keras.ops.where(
                force_finish,
                self.end_token_id,
                keras.ops.where(finished, self.end_token_id, next_token_ids),
            )
            just_finished = keras.ops.equal(next_token_ids, self.end_token_id)
            finished = keras.ops.logical_or(finished, just_finished)
            finished = keras.ops.logical_or(finished, force_finish)
            sequence_lengths = keras.ops.where(
                keras.ops.logical_not(finished),
                sequence_lengths + 1,
                sequence_lengths,
            )
            # Append next token to generated_ids.
            generated_ids = keras.ops.concatenate(
                [generated_ids, next_token_ids], axis=1
            )
            # Update n-gram bans.
            if repetition_penalty != 1.0:
                next_token_ids_np = keras.ops.convert_to_numpy(next_token_ids)
                for b in range(batch_size):
                    if not keras.ops.convert_to_numpy(finished[b]):
                        token_id = int(next_token_ids_np[b, 0])
                        appeared_tokens[b].add(token_id)
            if no_repeat_ngram_size > 0 and step >= no_repeat_ngram_size - 1:
                for b in range(batch_size):
                    if not keras.ops.convert_to_numpy(finished[b]):
                        ngram = generated_ids[
                            b, step - no_repeat_ngram_size + 1 : step + 1
                        ]
                        ngram = tuple(
                            int(t) for t in keras.ops.convert_to_numpy(ngram)
                        )
                        ngram_prefix = ngram[:-1]
                        banned_token = ngram[-1]
                        if ngram_prefix not in banned_ngrams[b]:
                            banned_ngrams[b][ngram_prefix] = []
                        banned_ngrams[b][ngram_prefix].append(banned_token)
            self_attention_caches = new_self_attention_caches
            # Break if all sequences are finished.
            if keras.ops.all(finished):
                break

        # Trim sequences at end token and pad.
        eos_mask = keras.ops.equal(generated_ids, self.end_token_id)
        has_eos = keras.ops.any(eos_mask, axis=-1)
        end_indices = keras.ops.argmax(eos_mask, axis=-1)
        end_indices = keras.ops.where(
            has_eos, end_indices, keras.ops.shape(generated_ids)[1]
        )
        all_indices = keras.ops.arange(keras.ops.shape(generated_ids)[1])
        mask = all_indices[None, :] <= end_indices[:, None]
        trimmed_ids = keras.ops.where(mask, generated_ids, self.pad_token_id)
        return trimmed_ids

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "backbone": keras.saving.serialize_keras_object(self.backbone),
                "audio_converter": keras.saving.serialize_keras_object(
                    self.audio_converter
                ),
                "decoder_start_token_id": self.decoder_start_token_id,
                "end_token_id": self.end_token_id,
                "pad_token_id": self.pad_token_id,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        backbone = keras.saving.deserialize_keras_object(config["backbone"])
        audio_converter = keras.saving.deserialize_keras_object(
            config["audio_converter"]
        )
        return cls(
            backbone=backbone,
            audio_converter=audio_converter,
            decoder_start_token_id=config.get("decoder_start_token_id", 1),
            end_token_id=config.get("end_token_id", 2),
            pad_token_id=config.get("pad_token_id", 0),
            name=config.get("name", None),
        )
