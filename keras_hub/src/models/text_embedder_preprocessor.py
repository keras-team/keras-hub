import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.TextEmbedderPreprocessor")
class TextEmbedderPreprocessor(Preprocessor):
    """Base class for text embedding preprocessing layers.

    `TextEmbedderPreprocessor` tasks wrap a `keras_hub.tokenizer.Tokenizer` to
    create a preprocessing layer for text embedding tasks. It is intended to be
    paired with a `keras_hub.models.TextEmbedder` task.

    All `TextEmbedderPreprocessor` take inputs three ordered inputs, `x`, `y`,
    and `sample_weight`. `x`, the first input, should always be included. It can
    be a single string, a batch of strings, or a tuple of batches of string
    segments that should be combined into a single sequence. See examples below.
    `y` and `sample_weight` are optional inputs that will be passed through
    unaltered.

    The layer will output either `x`, an `(x, y)` tuple if labels were
    provided, or an `(x, y, sample_weight)` tuple if labels and sample weight
    were provided. `x` will be a dictionary with tokenized input, the exact
    contents of the dictionary will depend on the model being used.

    All `TextEmbedderPreprocessor` tasks include a `from_preset()` constructor
    which can be used to load a pre-trained config and vocabularies. You can
    call the `from_preset()` constructor directly on this base class, in which
    case the correct class for your model will be automatically instantiated.

    Examples:
    ```python
    preprocessor = keras_hub.models.TextEmbedderPreprocessor.from_preset(
        "all_minilm_l6_v2_en",
        sequence_length=256, # Optional.
    )

    # Tokenize and pad/truncate a single sentence.
    x = "The quick brown fox jumped."
    x = preprocessor(x)

    # Tokenize and pad/truncate a batch of sentences.
    x = ["The quick brown fox jumped.", "Call me Ishmael."]
    x = preprocessor(x)
    ```
    """

    def __init__(
        self,
        tokenizer,
        sequence_length=256,
        truncate="round_robin",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.packer = None
        self.sequence_length = sequence_length
        self.truncate = truncate

        # `MultiSegmentPacker` requires TF workflow, so disable Python
        # workflow on this preprocessor layer.
        self._allow_python_workflow = False

    def build(self, input_shape):
        super().build(input_shape)
        # Defer packer creation to `build()` so that we can be sure tokenizer
        # assets have loaded when restoring a saved model.
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            truncate=self.truncate,
            sequence_length=self.sequence_length,
        )

    @preprocessing_function
    def call(self, x, y=None, sample_weight=None):
        x = x if isinstance(x, tuple) else (x,)
        x = tuple(self.tokenizer(segment) for segment in x)
        token_ids, segment_ids = self.packer(x)
        x = {
            "token_ids": token_ids,
            "padding_mask": token_ids != self.tokenizer.pad_token_id,
            "segment_ids": segment_ids,
        }
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "truncate": self.truncate,
            }
        )
        return config

    @property
    def sequence_length(self):
        """The padded length of model input sequences."""
        return self._sequence_length

    @sequence_length.setter
    def sequence_length(self, value):
        self._sequence_length = value
        if self.packer is not None:
            self.packer.sequence_length = value
