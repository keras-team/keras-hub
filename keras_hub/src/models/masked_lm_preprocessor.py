import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.masked_lm_mask_generator import (
    MaskedLMMaskGenerator,
)
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.MaskedLMPreprocessor")
class MaskedLMPreprocessor(Preprocessor):
    """Base class for masked language modeling preprocessing layers.

    `MaskedLMPreprocessor` tasks wrap a `keras_hub.tokenizer.Tokenizer` to
    create a preprocessing layer for masked language modeling tasks. It is
    intended to be paired with a `keras.models.MaskedLM` task.

    All `MaskedLMPreprocessor` take inputs a single input. This can be a single
    string, a batch of strings, or a tuple of batches of string segments that
    should be combined into a single sequence. See examples below. These inputs
    will be tokenized, combined, and masked randomly along the sequence.

    This layer will always output a `(x, y, sample_weight)` tuple, where `x`
    is a dictionary with the masked, tokenized inputs, `y` contains the tokens
    that were masked in `x`, and `sample_weight` marks where `y` contains padded
    values. The exact contents of `x` will vary depending on the model being
    used.

    All `MaskedLMPreprocessor` tasks include a `from_preset()` constructor
    which can be used to load a pre-trained config and vocabularies. You can
    call the `from_preset()` constructor directly on this base class, in which
    case the correct class for you model will be automatically instantiated.

    Examples.
    ```python
    preprocessor = keras_hub.models.MaskedLMPreprocessor.from_preset(
        "bert_base_en_uncased",
        sequence_length=256, # Optional.
    )

    # Tokenize, mask and pack a single sentence.
    x = "The quick brown fox jumped."
    x, y, sample_weight = preprocessor(x)

    # Preprocess a batch of labeled sentence pairs.
    first = ["The quick brown fox jumped.", "Call me Ishmael."]
    second = ["The fox tripped.", "Oh look, a whale."]
    x, y, sample_weight = preprocessor((first, second))

    # With a `tf.data.Dataset`.
    ds = tf.data.Dataset.from_tensor_slices((first, second))
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """

    def __init__(
        self,
        tokenizer,
        sequence_length=512,
        truncate="round_robin",
        mask_selection_rate=0.15,
        mask_selection_length=96,
        mask_token_rate=0.8,
        random_token_rate=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.packer = None
        self.sequence_length = sequence_length
        self.truncate = truncate
        self.mask_selection_rate = mask_selection_rate
        self.mask_selection_length = mask_selection_length
        self.mask_token_rate = mask_token_rate
        self.random_token_rate = random_token_rate
        self.masker = None

    def build(self, input_shape):
        super().build(input_shape)
        # Defer masker creation to `build()` so that we can be sure tokenizer
        # assets have loaded when restoring a saved model.
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            truncate=self.truncate,
            sequence_length=self.sequence_length,
        )
        self.masker = MaskedLMMaskGenerator(
            mask_selection_rate=self.mask_selection_rate,
            mask_selection_length=self.mask_selection_length,
            mask_token_rate=self.mask_token_rate,
            random_token_rate=self.random_token_rate,
            vocabulary_size=self.tokenizer.vocabulary_size(),
            mask_token_id=self.tokenizer.mask_token_id,
            unselectable_token_ids=self.tokenizer.special_token_ids,
        )

    @preprocessing_function
    def call(self, x, y=None, sample_weight=None):
        x = x if isinstance(x, tuple) else (x,)
        x = tuple(self.tokenizer(segment) for segment in x)
        token_ids, segment_ids = self.packer(x)
        padding_mask = token_ids != self.tokenizer.pad_token_id
        masker_outputs = self.masker(token_ids)
        x = {
            "token_ids": masker_outputs["token_ids"],
            "padding_mask": padding_mask,
            "segment_ids": segment_ids,
            "mask_positions": masker_outputs["mask_positions"],
        }
        y = masker_outputs["mask_ids"]
        sample_weight = masker_outputs["mask_weights"]
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "truncate": self.truncate,
                "mask_selection_rate": self.mask_selection_rate,
                "mask_selection_length": self.mask_selection_length,
                "mask_token_rate": self.mask_token_rate,
                "random_token_rate": self.random_token_rate,
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
