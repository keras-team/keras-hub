from keras_hub.src.models.seq_2_seq_lm import Seq2SeqLM


class AudioToText(Seq2SeqLM):
    """Base class for audio-to-text models.

    `AudioToText` tasks wrap a `keras_hub.models.Backbone` (capable of
    processing audio and text features) and a
    `keras_hub.models.AudioToTextPreprocessor` to create a model for
    audio-to-text tasks like speech recognition or audio transcription.

    These models typically consist of an encoder that processes audio input
    and a decoder that generates a textual representation.

    `AudioToText` tasks provide a high-level `generate()` method for
    auto-regressively generating text from audio input. An optional text
    prompt can also be provided to the decoder to guide generation. The
    sampling strategy for generation (e.g., greedy, top-k, top-p) can be
    controlled via the `sampler` argument in the `compile()` method.

    When calling `fit()`, inputs should consist of audio data and corresponding
    target text transcriptions. The model is trained to predict the target text
    token-by-token.

    All `AudioToText` tasks include a `from_preset()` constructor which
    can be used to load pre-trained configurations and weights for specific
    audio-to-text models.
    This constructor can also be called on the base `AudioToText` class,
    which will automatically select the correct subclass based on the preset.

    Examples:
    ```python
    # Load a Moonshine backbone with pre-trained weights.
    # AudioToText is a base class. You will typically work with a specific
    # implementation, such as `keras_hub.models.MoonshineAudioToText`.
    # The following examples demonstrate common usage patterns.

    # Initialize a model from a preset using the specific subclass.
    audio_to_text = keras_hub.models.MoonshineAudioToText.from_preset(
        "moonshine_base_en"
    )

    # Initialize a model from a preset using the base class.
    audio_to_text_model_base = keras_hub.models.AudioToText.from_preset(
        "moonshine_base_en"
    )

    # Generate text from an audio input.
    audio_input_tensor = keras.random.normal((1, 16000, 1))
    generated_output = audio_to_text_model.generate(
        {"audio": audio_input_tensor}
    )

    # Generate conditioned on the `"The quick brown fox."` as an input sequence.
    prompted_output = audio_to_text_model.generate(
        {"audio": audio_input_tensor, "text": "The quick brown fox."}
    )

    # Use a different sampling strategy for generation.
    audio_to_text_model.compile(sampler="greedy")
    greedy_output = audio_to_text_model.generate(
        {"audio": audio_input_tensor}
    )
    """

    # TODO: Fill in once audio to text task model requirements are clearer.
