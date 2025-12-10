import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.parseq.parseq_backbone import PARSeqBackbone
from keras_hub.src.models.parseq.parseq_image_converter import (
    PARSeqImageConverter,
)
from keras_hub.src.models.parseq.parseq_tokenizer import PARSeqTokenizer
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import strip_to_ragged


@keras_hub_export("keras_hub.models.PARSeqCausalLMPreprocessor")
class PARSeqCausalLMPreprocessor(CausalLMPreprocessor):
    backbone_cls = PARSeqBackbone
    tokenizer_cls = PARSeqTokenizer
    image_converter_cls = PARSeqImageConverter

    def __init__(
        self,
        image_converter=None,
        tokenizer=None,
        sequence_length=25,
        add_start_token=True,
        add_end_token=True,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            add_start_token=add_start_token,
            add_end_token=add_end_token,
            **kwargs,
        )
        self.image_converter = image_converter

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
        """Preprocesses the input data for training.

        This method takes a dictionary containing images and text responses,
        and converts them into a format suitable for training a PARSeq model.

        Args:
            x: dict. A dictionary containing the input data. Must have keys
                "images" and "responses".
            y: The target data. Defaults to None.
            sample_weight: The sample weights. Defaults to None.
            sequence_length: int. The maximum length of the input sequence.
                Defaults to None, which uses the pre-defined sequence length.

        Returns:
            A tuple containing the preprocessed input data, target data, and
                sample weights.
        """
        sequence_length = sequence_length or self.sequence_length
        images, responses = x["images"], x["responses"]
        if self.image_converter:
            images = self.image_converter(images)
        token_ids = self.tokenizer(responses)
        token_ids, padding_mask = self.packer(
            token_ids,
            sequence_length=sequence_length + 1,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        x = {
            "images": images,
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
        """Convert strings to integer token input for generation.

        Similar to calling the layer for training, this method takes in strings
        or tensor strings, tokenizes and packs the input, and computes a padding
        mask masking all inputs not filled in with a padded value.

        Unlike calling the layer for training, this method does not compute
        labels and will never append a `tokenizer.end_token_id` to the end of
        the sequence (as generation is expected to continue at the end of the
        inputted prompt).
        """
        if not self.built:
            self.build(None)
        sequence_length = sequence_length or self.sequence_length
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
                    [batch_size, sequence_length - 1],
                    self.tokenizer.pad_token_id,
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

    @preprocessing_function
    def generate_postprocess(
        self,
        x,
    ):
        """Convert integer token output to strings for generation.

        This method reverses `generate_preprocess()`, by first removing all
        padding and start/end tokens, and then converting the integer sequence
        back to a string.
        """
        if not self.built:
            self.build(None)

        token_ids, padding_mask = x["token_ids"], x["padding_mask"]
        ids_to_strip = self.tokenizer.special_token_ids
        token_ids = strip_to_ragged(token_ids, padding_mask, ids_to_strip)
        return self.tokenizer.detokenize(token_ids)

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
