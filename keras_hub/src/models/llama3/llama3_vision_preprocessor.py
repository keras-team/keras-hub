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
    """Preprocessor for the Llama 3.2 Vision model.

    This layer handles preprocessing of text and image inputs, combining
    a tokenizer for text and an image converter for images.

    Args:
        tokenizer: A `keras_hub.models.Llama3Tokenizer` instance.
        image_converter: A `keras_hub.models.Llama3VisionImageConverter`
            instance. Defaults to `None`.
        sequence_length: int. The maximum sequence length. Defaults to `1024`.
        add_start_token: bool. Whether to add start token. Defaults to `True`.
        add_end_token: bool. Whether to add end token. Defaults to `True`.
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
        if isinstance(x, dict):
            text = x.get("text", None)
            images = x.get("images", None)
        else:
            text = x
            images = None

        output = {}

        if text is not None:
            token_ids = self.tokenizer(text)
            token_ids, padding_mask = self.packer(
                token_ids,
                add_start_value=self.add_start_token,
                add_end_value=self.add_end_token,
            )
            output["token_ids"] = token_ids
            output["padding_mask"] = padding_mask

        if images is not None and self.image_converter is not None:
            images = self.image_converter(images)
            output["images"] = images

        if y is not None:
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
