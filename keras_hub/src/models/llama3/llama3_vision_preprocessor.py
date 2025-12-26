import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.llama3.llama3_tokenizer import Llama3Tokenizer
from keras_hub.src.models.llama3.llama3_vision_image_converter import (
    Llama3VisionImageConverter,
)
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.Llama3VisionPreprocessor")
class Llama3VisionPreprocessor(Preprocessor):
    """Llama 3 Vision Preprocessor.

    This layer handles the preprocessing of both text and image inputs.
    It combines a `Llama3Tokenizer` for text and a `Llama3VisionImageConverter`
    for images.

    Args:
        tokenizer: A `Llama3Tokenizer` instance.
        image_converter: A `Llama3VisionImageConverter` instance.
        sequence_length: int. The fixed length of the tokenized text output.
        add_start_token: bool. Whether to add the start token to the text.
        add_end_token: bool. Whether to add the end token to the text.
        **kwargs: Arguments passed to the parent class.
    """

    tokenizer_cls = Llama3Tokenizer
    image_converter_cls = Llama3VisionImageConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        sequence_length=1024,
        add_start_token=True,
        add_end_token=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.image_converter = image_converter
        self.packer = None
        self.sequence_length = sequence_length
        self.add_start_token = add_start_token
        self.add_end_token = add_end_token

    def build(self, input_shape):
        # Create packer for text tokenization
        self.packer = StartEndPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.sequence_length,
            return_padding_mask=True,
        )
        self.built = True

    @preprocessing_function
    def call(self, x, y=None, sample_weight=None):
        """Process inputs.

        x: Can be a dict {"images": ..., "text": ...} or just "text".
        """
        # 1. Normalize Input (handle dict inputs)
        if isinstance(x, dict):
            text = x.get("text", None)
            images = x.get("images", None)
        else:
            # Assume x is text if not dict (simplification)
            text = x
            images = None

        output = {}

        # 2. Process Text
        if text is not None:
            # Tokenize (without sequence_length parameter)
            token_ids = self.tokenizer(text)
            # Pack and pad tokens
            token_ids, padding_mask = self.packer(
                token_ids,
                add_start_value=self.add_start_token,
                add_end_value=self.add_end_token,
            )
            output["token_ids"] = token_ids
            output["padding_mask"] = padding_mask

        # 3. Process Images
        if images is not None and self.image_converter is not None:
            images = self.image_converter(images)
            output["images"] = images

        # 4. Handle Labels (y) if training
        if y is not None:
            # Tokenize labels
            tokenized_y = self.tokenizer(y)
            tokenized_y, _ = self.packer(
                tokenized_y,
                add_start_value=self.add_start_token,
                add_end_value=self.add_end_token,
            )
            return keras.utils.pack_x_y_sample_weight(
                output, tokenized_y, sample_weight
            )

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "add_start_token": self.add_start_token,
                "add_end_token": self.add_end_token,
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
