import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.models.t5.t5_tokenizer import T5Tokenizer
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.T5Preprocessor")
class T5Preprocessor(Preprocessor):
    tokenizer_cls = T5Tokenizer

    def __init__(
        self,
        tokenizer,
        sequence_length=256,
        add_start_token=False,
        add_end_token=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
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
    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        sequence_length = sequence_length or self.sequence_length
        token_ids, padding_mask = self.packer(
            self.tokenizer(x),
            sequence_length=sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        x = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

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
