import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.TextRecognitionPreprocessor")
class TextRecognitionPreprocessor(Preprocessor):
    def __init__(
        self,
        image_converter=None,
        tokenizer=None,
        sequence_length=25,
        add_start_token=True,
        add_end_token=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_converter = image_converter
        self.packer = None
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.add_start_token = add_start_token
        self.add_end_token = add_end_token

    def build(self, input_shape):
        # Defer packer creation to `build()` so that we can be sure tokenizer
        # assets have loaded when restoring a saved model.
        self.packer = StartEndPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.sequence_length,
            return_padding_mask=True,
        )
        self.built = True

    @preprocessing_function
    def call(self, x, y=None, sample_weight=None, sequence_length=None):
        sequence_length = sequence_length or self.sequence_length
        if self.image_converter:
            x = self.image_converter(x)
        if y is not None:
            token_ids = self.tokenizer(y)
            token_ids, padding_mask = self.packer(
                token_ids,
                sequence_length=sequence_length + 1,
                add_start_value=self.add_start_token,
                add_end_value=self.add_end_token,
            )

            padding_mask = keras.ops.bitwise_or(
                padding_mask,
                keras.ops.equal(token_ids, self.tokenizer.end_token_id),
            )
            x = {
                "images": x,
                "token_ids": token_ids[..., :-1],
                "padding_mask": padding_mask[..., :-1],
            }
            # Target `y` will be the next token.
            y, sample_weight = token_ids[..., 1:], padding_mask[..., 1:]
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    @preprocessing_function
    def generate_preprocess(
        self,
        x,
        sequence_length=None,
    ):
        if not self.built:
            self.build(None)
        sequence_length = sequence_length or self.sequence_length
        # +1 for <eos> at end of sequence.
        num_steps = sequence_length + 1
        images = x
        if self.image_converter:
            images = self.image_converter(images)

        images_shape = keras.ops.shape(images)
        if len(images_shape) == 3:
            batch_size = 1
        else:
            batch_size = images_shape[0]

        token_ids = ops.concatenate(
            (
                ops.full([batch_size, 1], self.tokenizer.start_token_id),
                ops.full(
                    [batch_size, num_steps - 1], self.tokenizer.pad_token_id
                ),
            ),
            axis=1,
        )

        padding_mask = ops.equal(token_ids, self.tokenizer.start_token_id)

        return {
            "images": images,
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

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
