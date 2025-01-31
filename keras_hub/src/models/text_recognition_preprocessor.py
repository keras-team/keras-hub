import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.TextRecognitionPreprocessor")
class TextRecognitionPreprocessor(Preprocessor):
    """Base class for image segmentation preprocessing layers.

    `TextRecognitionPreprocessor` wraps a
    `keras_hub.layers.ImageConverter` to create a preprocessing layer for
    text recognition tasks. It is intended to be paired with a
    `keras_hub.models.TextRecognition` task.

    All `TextRecognitionPreprocessor` instances take three inputs: `x`, `y`, and
    `sample_weight`.

    - `x`: The first input, should always be included. It can be an image or
      a batch of images.
    - `y`: (Optional) text representation of the letters/words from image.
    - `sample_weight`: (Optional) Will be passed through unaltered.

    The layer will output either `x`, an `(x, y)` tuple if labels were provided,
    or an `(x, y, sample_weight)` tuple if labels and sample weight were
    provided. `x` will be the input images after all model preprocessing has
    been applied.

    All `TextRecognitionPreprocessor` tasks include a `from_preset()`
    constructor which can be used to load a pre-trained config.
    You can call the `from_preset()` constructor directly on this base class, in
    which case the correct class for your model will be automatically
    instantiated.
    """

    def __init__(
        self,
        image_converter=None,
        tokenizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_converter = image_converter
        self.tokenizer = tokenizer

    @preprocessing_function
    def call(self, x, y=None, sample_weight=None):
        if self.image_converter:
            x = self.image_converter(x)
        if y is not None:
            y = self.tokenizer(y)
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
