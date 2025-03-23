import keras
import tensorflow as tf

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.models.gemma3.gemma3_image_converter import (
    Gemma3ImageConverter,
)
from keras_hub.src.models.gemma3.gemma3_tokenizer import Gemma3Tokenizer
from keras_hub.src.utils.tensor_utils import preprocessing_function

START_OF_IMAGE_TOKEN = "<start_of_image>"
IMAGE_PLACEHOLDER_TOKEN = "<img>"
END_OF_IMAGE_TOKEN = "<end_of_image>"


@keras_hub_export("keras_hub.models.Gemma3CausalLMPreprocessor")
class Gemma3CausalLMPreprocessor(CausalLMPreprocessor):
    backbone_cls = Gemma3Backbone
    tokenizer_cls = Gemma3Tokenizer
    image_converter_cls = Gemma3ImageConverter

    def __init__(
        self,
        tokenizer,
        image_converter,
        sequence_length=1024,
        add_start_token=True,
        add_end_token=True,
        text_only_model=False,
        max_images_per_prompt=2,
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
        self.text_only_model = text_only_model
        self.max_images_per_prompt = max_images_per_prompt
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

    def _format_output(
        self,
        images,
        token_ids,
        text_mask,
        response_mask,
        padding_mask,
        return_labels=False,
        text_only_input=False,
    ):
        if return_labels:
            token_ids = token_ids[..., :-1]
            text_mask = text_mask[..., :-1]
            response_mask = response_mask[..., :-1]
            padding_mask = padding_mask[..., :-1]

        batch_size, sequence_length = tf.shape(text_mask)

        if text_only_input:
            vision_indices = tf.ones(
                shape=[
                    batch_size,
                    0,
                ],
                dtype=tf.int32,
            )
        else:
            sequence_length = tf.shape(text_mask)[-1]
            flat_text_mask = tf.reshape(
                text_mask, (batch_size * sequence_length)
            )
            vision_indices = tf.where(tf.logical_not(flat_text_mask))
            vision_indices = tf.reshape(vision_indices, (batch_size, -1))

        # The last token does not have a next token, so we truncate it out.
        x = {
            # Image
            "images": images,
            # Text
            "token_ids": token_ids,
            "vision_indices": vision_indices,
            "text_mask": text_mask,
            "response_mask": response_mask,
            "padding_mask": padding_mask,
        }

        if return_labels:
            # Target `y` will be the next token.
            y = token_ids[..., 1:]
            # Only compute the loss for labels in the response.
            sample_weight = response_mask[..., 1:]
            return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
        else:
            return x

    def _pad_image_list_ragged(
        self,
        image_list,
        image_height,
        image_width,
        max_images_per_prompt=2,
        pad_value=0,
    ):
        """Pads image input to `max_images_per_prompt`."""

        if isinstance(image_list, tf.RaggedTensor):
            ragged_images = image_list
        elif isinstance(image_list, tf.Tensor):
            ragged_images = tf.RaggedTensor.from_tensor(image_list)
        else:
            ragged_images = tf.ragged.constant(image_list)

        batch_size = ragged_images.nrows()
        num_images = ragged_images.row_lengths()
        padded_images_dense = ragged_images.to_tensor(
            shape=[
                batch_size,
                max_images_per_prompt,
                image_height,
                image_width,
                3,
            ],
            default_value=tf.cast(pad_value, ragged_images.dtype),
        )

        return padded_images_dense, num_images

    def _get_image_placeholder_ragged_tensor(self, required_length, fill_value):
        """Identifies the number of dummy placeholder tokens to pad input with.

        Depending on the number of images provided per sample, and the
        allowed number of images, this method identifies the number of vision
        placeholder tokens we need to pad tokens with. This is necessary to
        ensure the same number of image tokens in every sample so as to not
        cause dynamic shape issues with XLA in the interleaving layer.
        """
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

    @preprocessing_function
    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        sequence_length = sequence_length or self.sequence_length
        images = x.get("images", None)
        prompts, responses = x["prompts"], x["responses"]

        # Replace `"<start_of_image>"` in prompts with
        # `"\n\n<start_of_image> <img> * 256 <end_of_image>\n\n"`.
        if not self.text_only_model:
            prompts = tf.strings.regex_replace(
                prompts,
                START_OF_IMAGE_TOKEN,
                f"\n\n{START_OF_IMAGE_TOKEN}"
                + IMAGE_PLACEHOLDER_TOKEN * self.num_vision_tokens_per_image
                + f"{END_OF_IMAGE_TOKEN}\n\n",
            )

        # Tokenise the inputs.
        prompts = self.tokenizer(prompts)
        responses = self.tokenizer(responses)

        # All the truncation should happen on the text token IDs and not on
        # the dummy placeholder image tokens which we will add at the end.
        # Hence, we use a packer on the text part.
        token_ids, segment_ids = self.packer(
            (prompts, responses),
            sequence_length=sequence_length
            if images is not None
            else sequence_length + 1,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )

        # If it is a text only model, return immediately.
        if self.text_only_model:
            # The last token does not have a next token, so we truncate it out.
            response_mask = segment_ids == 1
            padding_mask = token_ids != self.tokenizer.pad_token_id
            x = {
                "token_ids": token_ids[..., :-1],
                "padding_mask": padding_mask[..., :-1],
                "response_mask": response_mask,
            }

            # Target `y` will be the next token.
            y = token_ids[..., 1:]
            # Only compute the loss for labels in the response.
            sample_weight = response_mask[..., 1:]
            return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

        # NOTE: To handle the text-only input case, we need to pass an empty
        # tensor so as to skip the vision part.
        batch_size = tf.shape(prompts)[0]
        if images is None:
            images = tf.ones(
                shape=[
                    batch_size,
                    0,
                    self.image_converter.image_size[0],
                    self.image_converter.image_size[1],
                    3,
                ],
                dtype="float32",
            )

            text_mask = tf.ones_like(token_ids, dtype=bool)
            padding_mask = token_ids != self.tokenizer.pad_token_id
            response_mask = segment_ids == 1

            return self._format_output(
                images=images,
                token_ids=token_ids,
                text_mask=text_mask,
                response_mask=response_mask,
                padding_mask=padding_mask,
                return_labels=True,
                text_only_input=True,
            )

        # Pad images.
        first_image_shape = tf.shape(images[0], out_type=tf.int64)
        images, num_valid_images = self._pad_image_list_ragged(
            image_list=images,
            image_height=first_image_shape[-3],
            image_width=first_image_shape[-2],
            max_images_per_prompt=self.max_images_per_prompt,
            pad_value=tf.constant(0, dtype=tf.int32),
        )

        # Resize, rescale, etc. the images.
        padded_images_shape = tf.shape(images)
        images = tf.reshape(
            images,
            [
                -1,
                padded_images_shape[-3],
                padded_images_shape[-2],
                padded_images_shape[-1],
            ],
        )
        images = self.image_converter(images)
        height = (
            self.image_size[0]
            if self.image_converter.image_size
            else first_image_shape[-3]
        )
        width = (
            self.image_size[1]
            if self.image_converter.image_size
            else first_image_shape[-2]
        )
        images = tf.reshape(
            images,
            [
                padded_images_shape[0],
                self.max_images_per_prompt,
                height,
                width,
                3,
            ],
        )

        # Format tokens.
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
        vision_placeholder_tensor = self._get_image_placeholder_ragged_tensor(
            (self.max_images_per_prompt - num_valid_images)
            * self.num_vision_tokens_per_image,
            self.tokenizer.token_to_id("<img>"),
        )
        vision_placeholder_tensor = vision_placeholder_tensor.to_tensor(
            shape=[
                batch_size,
                self.max_images_per_prompt * self.num_vision_tokens_per_image,
            ],
            default_value=self.tokenizer.pad_token_id,
        )

        token_ids_with_placeholder = tf.concat(
            [token_ids, vision_placeholder_tensor], axis=1
        )

        # Now, pad everything to the same length.
        desired_length = (
            sequence_length
            + self.max_images_per_prompt * self.num_vision_tokens_per_image
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

        return self._format_output(
            images=images,
            token_ids=token_ids_with_placeholder,
            text_mask=text_mask,
            response_mask=response_mask_with_placeholder,
            padding_mask=padding_mask_with_placeholder,
            return_labels=True,
        )

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

        if isinstance(x, dict):
            images = x.get("images", None)
            # TODO: do we even need `responses` for generation? Makes sense for
            # finetuning (i.e., `call()`).
            responses = x.get("responses", None)
            prompts = x["prompts"]
        else:
            images = None
            responses = None
            prompts = x

        if not self.text_only_model:
            # Replace `"<start_of_image>"` in prompts with
            # `"\n\n<start_of_image> <img> * 256 <end_of_image>\n\n"`.
            prompts = tf.strings.regex_replace(
                prompts,
                START_OF_IMAGE_TOKEN,
                f"\n\n{START_OF_IMAGE_TOKEN}"
                + IMAGE_PLACEHOLDER_TOKEN * self.num_vision_tokens_per_image
                + f"{END_OF_IMAGE_TOKEN}\n\n",
            )

        prompts = self.tokenizer(prompts)

        if responses is not None:
            responses = self.tokenizer(responses)
            segments = (prompts, responses)
        else:
            segments = (prompts,)

        token_ids, segment_ids = self.packer(
            segments,
            sequence_length=sequence_length,
            add_end_value=False,
        )

        # If it is a text only model, return immediately.
        if self.text_only_model:
            response_mask = segment_ids == 1
            padding_mask = token_ids != self.tokenizer.pad_token_id
            return {
                "token_ids": token_ids,
                "padding_mask": padding_mask,
                "response_mask": response_mask,
            }

        # NOTE: To handle the text-only input case, we need to pass an empty
        # tensor so as to skip the vision part of the model.
        batch_size = tf.shape(prompts)[0]
        if images is None:
            images = tf.ones(
                shape=[
                    batch_size,
                    0,
                    self.image_converter.image_size[0],
                    self.image_converter.image_size[1],
                    3,
                ],
                dtype="float32",
            )

            text_mask = tf.ones_like(token_ids, dtype=bool)
            padding_mask = token_ids != self.tokenizer.pad_token_id
            response_mask = segment_ids == 1

            return self._format_output(
                images=images,
                token_ids=token_ids,
                text_mask=text_mask,
                response_mask=response_mask,
                padding_mask=padding_mask,
                return_labels=False,
                text_only_input=True,
            )

        # Pad images.
        first_image_shape = tf.shape(images[0], out_type=tf.int64)
        images, num_valid_images = self._pad_image_list_ragged(
            image_list=images,
            image_height=first_image_shape[-3],
            image_width=first_image_shape[-2],
            max_images_per_prompt=self.max_images_per_prompt,
            pad_value=0,
        )

        # Resize, rescale, etc. the images.
        padded_images_shape = tf.shape(images)
        images = tf.reshape(
            images,
            [
                -1,
                padded_images_shape[-3],
                padded_images_shape[-2],
                padded_images_shape[-1],
            ],
        )
        images = self.image_converter(images)
        height = (
            self.image_size[0]
            if self.image_converter.image_size
            else first_image_shape[-3]
        )
        width = (
            self.image_size[1]
            if self.image_converter.image_size
            else first_image_shape[-2]
        )
        images = tf.reshape(
            images,
            [
                padded_images_shape[0],
                self.max_images_per_prompt,
                height,
                width,
                3,
            ],
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
        vision_placeholder_tensor = self._get_image_placeholder_ragged_tensor(
            (self.max_images_per_prompt - num_valid_images)
            * self.num_vision_tokens_per_image,
            self.tokenizer.token_to_id("<img>"),
        )
        vision_placeholder_tensor = vision_placeholder_tensor.to_tensor(
            shape=[
                batch_size,
                self.max_images_per_prompt * self.num_vision_tokens_per_image,
            ],
            default_value=self.tokenizer.pad_token_id,
        )
        token_ids_with_placeholder = tf.concat(
            [token_ids, vision_placeholder_tensor], axis=1
        )

        # Now, pad everything to the same length.
        desired_length = (
            sequence_length
            + self.max_images_per_prompt * self.num_vision_tokens_per_image
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

        return self._format_output(
            images=images,
            token_ids=token_ids_with_placeholder,
            text_mask=text_mask,
            response_mask=response_mask_with_placeholder,
            padding_mask=padding_mask_with_placeholder,
            return_labels=False,
        )

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "num_vision_tokens_per_image": self.num_vision_tokens_per_image,
                "max_images_per_prompt": self.max_images_per_prompt,
                "text_only_model": self.text_only_model,
            }
        )
        return config
