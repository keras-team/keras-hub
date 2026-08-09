from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.preprocessor import Preprocessor


@keras_hub_export("keras_hub.models.TextToImagePreprocessor")
class TextToImagePreprocessor(Preprocessor):
    """Base class for text to image preprocessing layers.

    `TextToImagePreprocessor` tasks wrap a `keras_hub.tokenizer.Tokenizer` to
    create a preprocessing layer for text to image tasks. It is intended to be
    paired with a `keras_hub.models.TextToImage` task.

    The exact specifics of this layer will vary depending on the subclass
    implementation per model architecture. Generally, it will take text input,
    and tokenize, then pad/truncate so it is ready to be fed to a image
    generation model (e.g. a diffusion model).

    Examples.
    ```python
    preprocessor = keras_hub.models.TextToImagePreprocessor.from_preset(
        "stable_diffusion_3_medium",
        sequence_length=256, # Optional.
    )

    # Tokenize and pad/truncate a single sentence.
    x = "The quick brown fox jumped."
    x = preprocessor(x)

    # Tokenize and pad/truncate a batch of sentences.
    x = ["The quick brown fox jumped."]
    x = preprocessor(x)
    ```
    """

    pass
