from keras_hub.src.models.seq_2_seq_lm_preprocessor import Seq2SeqLMPreprocessor


class AudioToTextPreprocessor(Seq2SeqLMPreprocessor):
    """Base class for audio-to-text preprocessing layers.

    `AudioToTextPreprocessor` layers wrap an audio feature extractor (specific
    to the subclass) and a `keras_hub.tokenizer.Tokenizer` to create a
    preprocessing layer for audio-to-text tasks. It is intended to be
    paired with a `keras_hub.models.AudioToText` task.

    Subclasses are expected to handle the conversion of raw audio data into
    numerical features suitable for an encoder, and raw text data into token IDs
    for a decoder.

    All `AudioToTextPreprocessor` layers take a dictionary as input,
    typically with keys like `"audio"` (for audio data) and `"text"` (for
    target transcriptions or decoder prompts).

    This layer will always output a `(x, y, sample_weight)` tuple, where `x`
    is a dictionary containing processed audio features for the encoder and
    tokenized text inputs for the decoder. `y` contains the target token IDs
    (decoder input tokens shifted by one position), and `sample_weight`
    indicates padding in `y`. The exact keys and structure of features within
    `x` will depend on the specific subclass and the paired `AudioToText` model.

    An `AudioToTextPreprocessor` includes `generate_preprocess` and
    `generate_postprocess` methods for use during inference with an
    `AudioToText` model's `generate()` method.

    All `AudioToTextPreprocessor` tasks include a `from_preset()` constructor
    which can be used to load a pre-trained configuration, including tokenizer
    vocabularies and audio feature extraction settings. Calling `from_preset()`
    on this base class can instantiate the correct subclass registered for the
    given preset.

    Examples:
    ```python
    preprocessor = keras_hub.models.AudioToTextPreprocessor.from_preset(
        "moonshine_base_en",
        decoder_sequence_length=10
    )

    # Process a single audio-text pair.
    x = {
        "audio": keras.random.normal((1, 16000, 1)),
        "text": ["the quick brown fox"]
    }
    x, y, sample_weight = preprocessor(x)

    # Process a batch of audio-text pairs.
    x = {
        "audio": keras.random.normal((2, 16000, 1)),
        "text": ["first sentence", "second sentence"]
    }
    x, y, sample_weight = preprocessor(x)

    # With a `tf.data.Dataset`.
    audio_tf = keras.ops.convert_to_tensor(batch_input["audio"])
    text_tf = batch_input["text"] # List of strings
    x = {"audio": audio_tf, "text": text_tf}
    ds = tf.data.Dataset.from_tensor_slices(x)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(2) # Batching after map

    # Generate preprocess and postprocess.
    x = preprocessor.generate_preprocess({
        "audio": keras.random.normal((1, 16000, 1)),
        "text": ["optional prompt text"]
    })
    x = preprocessor.generate_postprocess({
        "decoder_token_ids": keras.ops.array([[10, 20, 30, 2, 0]]),
        "decoder_padding_mask": keras.ops.array([
            [True, True, True, True, False]
        ])
    })
    ```
    """

    # TODO: Fill in once audio to text task model requirements are clearer.
