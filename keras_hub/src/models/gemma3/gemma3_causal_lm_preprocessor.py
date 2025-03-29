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
from keras_hub.src.utils.tensor_utils import strip_to_ragged

START_OF_IMAGE_TOKEN = "<start_of_image>"
IMAGE_PLACEHOLDER_TOKEN = "<img>"
END_OF_IMAGE_TOKEN = "<end_of_image>"


@keras_hub_export("keras_hub.models.Gemma3CausalLMPreprocessor")
class Gemma3CausalLMPreprocessor(CausalLMPreprocessor):
    """Gemma3 Causal LM preprocessor.

    This preprocessing layer is meant for use with
    `keras_hub.models.Gemma3CausalLM`. By default, it will take in batches of
    images and strings, and return outputs in a `(x, y, sample_weight)` format,
    where the `y` label is the next token id in the `x` sequence.

    There is only one mode this layer currently supports, i.e.,
    `image_converter` is `None`. We preprocess the text like any other
    Causal LM preprocessor, i.e., tokenisation, padding, etc. The sequence
    is padded to `sequence_length`.

    For use with generation, the layer also exposes two methods
    `generate_preprocess()` and `generate_postprocess()`. When this preprocessor
    is attached to a `keras_hub.models.GemmaCausalLM` instance, these methods
    will be called implicitly in `generate()`. They can also be called
    standalone (e.g. to precompute preprocessing inputs for generation in a
    separate process).

    Args:
        tokenizer: A `keras_hub.models.GemmaTokenizer` instance.
        image_converter: A `keras_hub.layers.ImageConverter` instance. Defaults
            to `None`.
        sequence_length: The length of the packed inputs. Defaults to 1024.
        add_start_token: If `True`, the preprocessor will prepend the tokenizer
            start token to each input sequence. Defaults to `True`.
        add_end_token: If `True`, the preprocessor will append the tokenizer
            end token to each input sequence. Defaults to `True`.
        max_images_per_prompt: int. Permissible number of images per sample in
            the batch. Defaults to 2.
        num_vision_tokens_per_image: int. Number of vision placeholder tokens
            per image. Defaults to 256.

    Call arguments:
        x: A string, `tf.Tensor` or list of python strings.
        y: Label data. Should always be `None` as the layer generates labels.
        sample_weight: Label weights. Should always be `None` as the layer
            generates label weights.
        sequence_length: Pass to override the configured `sequence_length` of
            the layer.

    Examples:
    ```python
    # Load the preprocessor from a preset.
    preprocessor = keras_hub.models.Gemma3CausalLMPreprocessor.from_preset(
        "gemma3_4b_en"
    )

    # Text-only input.
    preprocessor(
        "prompts": ["The quick brown fox jumped."],
        "responses": [""],
    )

    # Images (pass one image)
    max_images_per_prompt = 2
    preprocessor(
        "prompts": ["The quick brown fox jumped."],
        "responses": [""],
        "images": [np.ones((2, 896, 896, 3)).astype("float32")],
        "num_valid_images": np.array([1,], dtype=np.int32)
    )
    ```
    """

    backbone_cls = Gemma3Backbone
    tokenizer_cls = Gemma3Tokenizer
    image_converter_cls = Gemma3ImageConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        sequence_length=1024,
        add_start_token=True,
        add_end_token=True,
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

        if image_converter is not None:
            raise ValueError(
                "Currently, only the text version of the Gemma3 model is "
                "supported."
            )

        self.image_converter = image_converter
        self.max_images_per_prompt = max_images_per_prompt
        self.num_vision_tokens_per_image = num_vision_tokens_per_image

        self.text_only_model = self.image_converter is None

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
            # Target `y` will be the next token.
            y = token_ids[..., 1:]
            # Only compute the loss for labels in the response.
            sample_weight = response_mask[..., 1:]

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
            "padding_mask": padding_mask,
        }

        if return_labels:
            return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
        else:
            return x

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

        # Extract text part of the input.
        prompts, responses = x["prompts"], x["responses"]

        # Extract images from the input.
        images = x.get("images", None)
        num_valid_images = x.get("num_valid_images", None)

        if self.text_only_model:
            if images is not None or num_valid_images is not None:
                raise ValueError(
                    "`image_converter` cannot be None when `images` or"
                    " `num_valid_images` is not None."
                )
        else:
            # Replace `"<start_of_image>"` in prompts with
            # `"\n\n<start_of_image> <img> * 256 <end_of_image>\n\n"`.
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

        # All truncation should happen on the text token IDs and not on
        # the dummy placeholder image tokens which we will add at the end.
        # Hence, we use a packer only on the text part first, and then
        # add the padded dummy placeholder tokens separately.
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
            }

            # Target `y` will be the next token.
            y = token_ids[..., 1:]
            # Only compute the loss for labels in the response.
            sample_weight = response_mask[..., 1:]
            return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

        # Vision preprocessing
        batch_size = tf.shape(prompts)[0]
        if images is None:
            # To handle the text-only input case, we need to pass an empty
            # tensor so as to skip the vision part of the model.
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

        original_image_shape = tf.shape(images)
        if num_valid_images is None:
            num_valid_images = tf.fill(
                dims=(batch_size,),
                value=self.max_images_per_prompt,
            )

        # Image inputs checks.
        if original_image_shape[1] != self.max_images_per_prompt:
            raise ValueError(
                "The number of images per sample should be the same as "
                "`max_images_per_prompt`. Received: "
                f"images.shape = {original_image_shape}, "
                f"max_images_per_prompt = {self.max_images_per_prompt}"
            )
        if tf.cast(
            tf.math.reduce_sum(
                tf.cast(
                    tf.math.greater(
                        num_valid_images, self.max_images_per_prompt
                    ),
                    dtype=tf.int32,
                )
            ),
            dtype=bool,
        ):
            raise ValueError(
                "`num_valid_images` should have values <= "
                "self.max_images_per_prompt. Received: "
                f"num_valid_images = {num_valid_images}, ",
                f"max_images_per_prompt = {self.max_images_per_prompt}",
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
            else original_image_shape[-3]
        )
        width = (
            self.image_size[1]
            if self.image_converter.image_size
            else original_image_shape[-2]
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

        if isinstance(x, dict):
            images = x.get("images", None)
            num_valid_images = x.get("num_valid_images", None)
            # TODO: do we even need `responses` for generation? Makes sense for
            # finetuning (i.e., `call()`).
            responses = x.get("responses", None)
            prompts = x["prompts"]
        else:
            images = None
            num_valid_images = None
            responses = None
            prompts = x

        if self.text_only_model:
            if images is not None or num_valid_images is not None:
                raise ValueError(
                    "`image_converter` cannot be None when `images` or"
                    " `num_valid_images` is not None."
                )
        else:
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
            }

        # Vision preprocessing
        batch_size = tf.shape(prompts)[0]
        if images is None:
            # To handle the text-only input case, we need to pass an empty
            # tensor so as to skip the vision part of the model.
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
        original_image_shape = tf.shape(images)
        if num_valid_images is None:
            num_valid_images = tf.fill(
                dims=(batch_size,),
                value=self.max_images_per_prompt,
            )

        # Image inputs checks.
        if original_image_shape[1] != self.max_images_per_prompt:
            raise ValueError(
                "The number of images per sample should be the same as "
                "`max_images_per_prompt`. Received: "
                f"images.shape = {original_image_shape}, "
                f"max_images_per_prompt = {self.max_images_per_prompt}"
            )
        if tf.cast(
            tf.math.reduce_sum(
                tf.cast(
                    tf.math.greater(
                        num_valid_images, self.max_images_per_prompt
                    ),
                    dtype=tf.int32,
                )
            ),
            dtype=bool,
        ):
            raise ValueError(
                "`num_valid_images` should have values <= "
                "self.max_images_per_prompt. Received: "
                f"num_valid_images = {num_valid_images}, ",
                f"max_images_per_prompt = {self.max_images_per_prompt}",
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
            else original_image_shape[-3]
        )
        width = (
            self.image_size[1]
            if self.image_converter.image_size
            else original_image_shape[-2]
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
            }
        )
        return config

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
        ids_to_strip += [self.tokenizer.token_to_id("<end_of_image>")]
        token_ids = strip_to_ragged(token_ids, padding_mask, ids_to_strip)
        return self.tokenizer.detokenize(token_ids)
