import keras
import tensorflow as tf

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor

# from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_image_converter import (
    Gemma3ImageConverter,
)
from keras_hub.src.models.gemma3.gemma3_tokenizer import Gemma3Tokenizer
from keras_hub.src.utils.tensor_utils import preprocessing_function


def get_image_placeholder_ragged_tensor(required_length, fill_value):
    required_length = tf.cast(required_length, tf.int32)
    ones_tensor = tf.ones_like(required_length, dtype=tf.int32)
    flattened_tensor = tf.repeat(ones_tensor, required_length)
    row_splits = tf.concat([[0], tf.cumsum(required_length)], axis=0)
    ragged_tensor = tf.RaggedTensor.from_row_splits(
        flattened_tensor, row_splits
    )
    ragged_tensor = ragged_tensor * fill_value
    ragged_tensor = tf.cast(ragged_tensor, tf.int32)
    return ragged_tensor


@keras_hub_export("keras_hub.models.Gemma3CausalLMPreprocessor")
class Gemma3CausalLMPreprocessor(CausalLMPreprocessor):
    # backbone_cls = Gemma3Backbone
    tokenizer_cls = Gemma3Tokenizer
    image_converter_cls = Gemma3ImageConverter

    def __init__(
        self,
        tokenizer,
        image_converter,
        sequence_length=1024,
        add_start_token=True,
        add_end_token=True,
        num_vision_tokens_per_image=256,
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
        self.num_vision_tokens_per_image = num_vision_tokens_per_image

    def build(self, input_shape):
        # Defer packer creation to `build()` so that we can be sure tokenizer
        # assets have loaded when restoring a saved model.
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sep_value=[],
            sequence_length=self.sequence_length,
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
        image_max_length = self.image_converter.image_max_length
        images = x.get("images", None)
        prompts, responses = x["prompts"], x["responses"]

        # Replace `"<start_of_image>"` in prompts with
        # `"\n\n<start_of_image> <img> * 256 <end_of_image>\n\n"`.
        prompts = tf.strings.regex_replace(
            prompts,
            "<start_of_image>",
            "\n\n<start_of_image>"
            + "<img>" * self.num_vision_tokens_per_image
            + "<end_of_image>\n\n",
        )

        # `response` cannot have any `<img>` tokens. Remove, if present.
        for token in [
            " <start_of_image>",
            "<start_of_image> ",
            "<start_of_image>",
        ]:
            responses = tf.strings.regex_replace(responses, token, "")

        # Tokenise the inputs.
        prompts = self.tokenizer(prompts)
        responses = self.tokenizer(responses)

        # Resize, rescale, pad, etc. the images.
        # NOTE: To handle the text-only case, we need to pass a dummy input
        # (with one axis=0) so as to skip the vision part.
        batch_size = tf.shape(prompts)[0]
        if images is None:
            images = tf.ones(
                shape=[
                    batch_size,
                    0,
                    image_max_length,
                    self.image_converter.image_size[0],
                    self.image_converter.image_size[1],
                    3,
                ],
                dtype="float32",
            )
            # Should be all 0 since we do not have any images in the input.
            num_valid_images = tf.zeros((batch_size,))
        else:
            images, num_valid_images = self.image_converter(images)
            images = tf.expand_dims(images, axis=1)

        # All the truncation should happen on the text token IDs and not on
        # the dummy placeholder image tokens which we will add at the end.
        # Hence, we use a packer on the text part.
        token_ids, segment_ids = self.packer(
            (prompts, responses),
            sequence_length=sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        padding_mask = token_ids != self.tokenizer.pad_token_id
        token_ids = tf.ragged.boolean_mask(token_ids, padding_mask)
        segment_ids = tf.ragged.boolean_mask(segment_ids, padding_mask)
        padding_mask = tf.ragged.boolean_mask(padding_mask, padding_mask)
        response_mask = segment_ids == 1

        # Using `num_valid_images`, we need to add dummy image tokens at the
        # end of the tokenized text. Ideally, we could have passed an image
        # padding mask to the model, but it won't work with XLA since an
        # `ops.where` on it in the interleaving layer will return different
        # number of images every time. So, we need to fix the number of images.
        vision_placeholder_tensor = get_image_placeholder_ragged_tensor(
            image_max_length - num_valid_images,
            self.tokenizer.token_to_id("<img>"),
        )
        vision_placeholder_tensor = vision_placeholder_tensor.to_tensor(
            shape=[
                batch_size,
                image_max_length * self.num_vision_tokens_per_image,
            ],
            default_value=self.tokenizer.pad_token_id,
        )

        token_ids_with_placeholder = tf.concat(
            [token_ids, vision_placeholder_tensor], axis=1
        )

        # Now, pad everything to the same length.
        # TODO: Check with @mattdangerw whether we want to pad to
        # sequence_length, i,e., whether the passed value for sequence length
        # should subsume extra vision tokens.
        desired_length = (
            sequence_length
            + image_max_length * self.num_vision_tokens_per_image
        )
        token_ids_with_placeholder = token_ids_with_placeholder.to_tensor(
            shape=[batch_size, desired_length + 1],
            default_value=self.tokenizer.pad_token_id,
        )
        padding_mask_with_placeholder = padding_mask.to_tensor(
            shape=[batch_size, desired_length + 1],
            default_value=False,
        )
        response_mask_with_placeholder = response_mask.to_tensor(
            shape=[batch_size, desired_length + 1],
            default_value=False,
        )

        text_mask = token_ids_with_placeholder != self.tokenizer.token_to_id(
            "<img>"
        )

        # The last token does not have a next token, so we truncate it out.
        x = {
            # Image
            "images": images,
            # Text
            "token_ids": token_ids_with_placeholder[..., :-1],
            "text_mask": text_mask[..., :-1],
            "response_mask": response_mask_with_placeholder[..., :-1],
            "padding_mask": padding_mask_with_placeholder[..., :-1],
        }

        # Target `y` will be the next token.
        y = token_ids_with_placeholder[..., 1:]
        # Only compute the loss for labels in the response.
        sample_weight = response_mask_with_placeholder[..., 1:]
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
        image_max_length = self.image_converter.image_max_length

        images = x.get("images", None)
        prompts = x["prompts"]

        # Replace `"<start_of_image>"` in prompts with
        # `"\n\n<start_of_image> <img> * 256 <end_of_image>\n\n"`.
        prompts = tf.strings.regex_replace(
            prompts,
            "<start_of_image>",
            "\n\n<start_of_image>"
            + "<img>" * self.num_vision_tokens_per_image
            + "<end_of_image>\n\n",
        )

        prompts = self.tokenizer(prompts)

        if "responses" in x:
            # `responses` cannot have any `<start_of_image>` tokens. Remove, if
            # present.
            responses = x["responses"]
            for token in [
                " <start_of_image>",
                "<start_of_image> ",
                "<start_of_image>",
            ]:
                responses = tf.strings.regex_replace(responses, token, "")

            responses = self.tokenizer(responses)
            segments = (prompts, responses)
        else:
            segments = (prompts,)

        # Resize, rescale, pad, etc. the images.
        # NOTE: To handle the text-only case, we need to pass a dummy input
        # (with one axis=0) so as to skip the vision part.
        batch_size = tf.shape(prompts)[0]
        if images is None:
            images = tf.ones(
                shape=[
                    batch_size,
                    0,
                    image_max_length,
                    self.image_converter.image_size[0],
                    self.image_converter.image_size[1],
                    3,
                ],
                dtype="float32",
            )
            # Should be all 0 since we do not have any images in the input.
            num_valid_images = tf.zeros((batch_size,))
        else:
            images, num_valid_images = self.image_converter(images)
            images = tf.expand_dims(images, axis=1)

        token_ids, segment_ids = self.packer(
            segments,
            sequence_length=sequence_length,
            add_end_value=False,
        )
        padding_mask = token_ids != self.tokenizer.pad_token_id
        token_ids = tf.ragged.boolean_mask(token_ids, padding_mask)
        segment_ids = tf.ragged.boolean_mask(segment_ids, padding_mask)
        padding_mask = tf.ragged.boolean_mask(padding_mask, padding_mask)
        response_mask = segment_ids == 1

        # Using `num_valid_images`, we need to add dummy image tokens at the
        # end of the tokenized text. Ideally, we could have passed an image
        # padding mask to the model, but it won't work with XLA since an
        # `ops.where` on it in the interleaving layer will return different
        # number of images every time. So, we need to fix the number of images.
        vision_placeholder_tensor = get_image_placeholder_ragged_tensor(
            image_max_length - num_valid_images,
            self.tokenizer.token_to_id("<img>"),
        )
        vision_placeholder_tensor = vision_placeholder_tensor.to_tensor(
            shape=[
                batch_size,
                image_max_length * self.num_vision_tokens_per_image,
            ],
            default_value=self.tokenizer.pad_token_id,
        )
        token_ids_with_placeholder = tf.concat(
            [token_ids, vision_placeholder_tensor], axis=1
        )

        # Now, pad everything to the same length.
        # TODO: Check with @mattdangerw whether we want to pad to
        # sequence_length, i,e., whether the passed value for sequence length
        # should subsume extra vision tokens.
        desired_length = (
            sequence_length
            + image_max_length * self.num_vision_tokens_per_image
        )
        token_ids_with_placeholder = token_ids_with_placeholder.to_tensor(
            shape=[batch_size, desired_length],
            default_value=self.tokenizer.pad_token_id,
        )
        padding_mask_with_placeholder = padding_mask.to_tensor(
            shape=[batch_size, desired_length],
            default_value=False,
        )
        response_mask_with_placeholder = response_mask.to_tensor(
            shape=[batch_size, desired_length],
            default_value=False,
        )

        text_mask = token_ids_with_placeholder != self.tokenizer.token_to_id(
            "<img>"
        )

        output = {
            # Image
            "images": images,
            # Text
            "token_ids": token_ids_with_placeholder,
            "text_mask": text_mask,
            "response_mask": response_mask_with_placeholder,
            "padding_mask": padding_mask_with_placeholder,
        }
        return output

    def get_config(self):
        config = super().get_config()
        config["num_vision_tokens_per_image"] = self.num_vision_tokens_per_image
        return config
