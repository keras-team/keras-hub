import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.moonshine.moonshine_audio_converter import (
    MoonshineAudioConverter,
)
from keras_hub.src.models.moonshine.moonshine_backbone import MoonshineBackbone
from keras_hub.src.models.moonshine.moonshine_tokenizer import (
    MoonshineTokenizer,
)
from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM


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
    # MoonshineForConditionalGeneration class (https://github.com/huggingface/transformers/blob/dcbdf7e962c4b36140cc9ee76f870016121e69e5/src/transformers/models/moonshine/modeling_moonshine.py#L1509).

    def __init__(self, backbone, audio_converter, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.audio_converter = audio_converter
        self.tokenizer = tokenizer
        self.decoder_start_token_id = (
            self.tokenizer.bos_token_id
        )  # Start token <s>
        self.end_token_id = self.tokenizer.eos_token_id  # End token </s>

    def call(self, inputs, training=False):
        audio = inputs["audio"]
        token_ids = inputs["token_ids"]

        # Preprocess audio.
        audio_features = self.audio_converter(audio)
        encoder_input_values = audio_features["input_values"]
        encoder_padding_mask = audio_features["attention_mask"]

        # Prepare decoder inputs.
        decoder_padding_mask = keras.ops.cast(
            keras.ops.not_equal(token_ids, 0), "bool"
        )

        # Backbone forward pass.
        outputs = self.backbone(
            {
                "encoder_input_values": encoder_input_values,
                "decoder_token_ids": token_ids,
                "encoder_padding_mask": encoder_padding_mask,
                "decoder_padding_mask": decoder_padding_mask,
            },
            training=training,
        )
        decoder_hidden_states = outputs["decoder_sequence_output"]
        logits = self.backbone.logits(decoder_hidden_states)
        return logits

    def generate(self, audio_inputs, max_new_tokens=100):
        batch_size = keras.ops.shape(audio_inputs)[0]
        audio_features = self.audio_converter(audio_inputs)
        encoder_input_values = audio_features["input_values"]
        encoder_padding_mask = audio_features["attention_mask"]

        decoder_input_ids = keras.ops.convert_to_tensor(
            [[self.decoder_start_token_id]] * batch_size, dtype="int32"
        )
        generated_ids = decoder_input_ids
        finished = keras.ops.zeros((batch_size,), dtype="bool")
        end_token_id = self.tokenizer.eos_token_id

        for step in range(max_new_tokens):
            decoder_padding_mask = keras.ops.ones_like(
                decoder_input_ids, dtype="bool"
            )
            outputs = self.backbone(
                {
                    "encoder_input_values": encoder_input_values,
                    "decoder_token_ids": decoder_input_ids,
                    "encoder_padding_mask": encoder_padding_mask,
                    "decoder_padding_mask": decoder_padding_mask,
                },
                training=False,
            )
            logits = self.backbone.logits(outputs["decoder_sequence_output"])
            next_token_ids = keras.ops.argmax(logits[:, -1, :], axis=-1)
            next_token_ids = keras.ops.where(
                finished, end_token_id, next_token_ids
            )
            generated_ids = keras.ops.concatenate(
                [generated_ids, next_token_ids[:, None]], axis=-1
            )
            decoder_input_ids = generated_ids
            # Check for end token.
            just_finished = keras.ops.logical_and(
                keras.ops.equal(next_token_ids, end_token_id),
                keras.ops.logical_not(finished),
            )
            finished = keras.ops.logical_or(finished, just_finished)
            # Break if all sequences are finished.
            if keras.ops.all(finished):
                break

        end_indices = keras.ops.argmax(
            keras.ops.equal(generated_ids, end_token_id), axis=-1
        )
        end_indices = keras.ops.where(
            keras.ops.equal(end_indices, 0),
            keras.ops.shape(generated_ids)[1],
            end_indices,
        )
        all_indices = keras.ops.arange(keras.ops.shape(generated_ids)[1])
        mask = all_indices[None, :] <= end_indices[:, None]
        trimmed_ids = keras.ops.where(
            mask, generated_ids, self.tokenizer.pad_token_id
        )
        return trimmed_ids

    def build(self, input_shape):
        super().build(input_shape)

    def get_config(self):
        return {
            "backbone": self.backbone.get_config(),
            "audio_converter": self.audio_converter.get_config(),
            "tokenizer": self.tokenizer.get_config(),
        }

    @classmethod
    def from_config(cls, config):
        backbone = MoonshineBackbone.from_config(config["backbone"])
        audio_converter = MoonshineAudioConverter.from_config(
            config["audio_converter"]
        )
        tokenizer = MoonshineTokenizer.from_config(config["tokenizer"])
        return cls(
            backbone=backbone,
            audio_converter=audio_converter,
            tokenizer=tokenizer,
        )
