import keras
import numpy as np
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


@keras_hub_export("keras_hub.models.Gemma3CausalLMPreprocessor")
class Gemma3CausalLMPreprocessor(CausalLMPreprocessor):
    """Gemma3 Causal LM preprocessor.

    This preprocessing layer is meant for use with
    `keras_hub.models.Gemma3CausalLM`. It can be configured in two ways:
    text-only and text + vision, based on whether the passed value of
    `image_converter` is None. For the former, it takes in batches of strings,
    whereas for the latter, it takes in batches of images and strings. It
    returns outputs in a `(x, y, sample_weight)` format, where the `y` label is
    the next token id in the `x` sequence. `sample_weight` is 0 for "prompt"
    tokens, and 1 for "response" tokens, so that the loss is computed only on
    the "response" tokens.

    For the text + vision case, this layer replaces instance of
    `<start_of_image>` token in the prompt with `num_vision_tokens_per_image`
    placeholder tokens. It also returns indices of where these vision tokens
    are present so that in the model, image embeddings can be placed in the
    right position in the sequence of text embeddings. Note that if
    `max_images_per_prompt` is 2, you can pass either 0, 1, 2 images per sample.
    The value 0 corresponds to text-only input.

    For use with generation, the layer also exposes two methods
    `generate_preprocess()` and `generate_postprocess()`. When this preprocessor
    is attached to a `keras_hub.models.Gemma3CausalLM` instance, these methods
    will be called implicitly in `generate()`. They can also be called
    standalone (e.g. to precompute preprocessing inputs for generation in a
    separate process).

    Args:
        tokenizer: A `keras_hub.models.Gemma3Tokenizer` instance.
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
    # === Language Gemma3 model ===
    # Load the preprocessor from a preset.
    preprocessor = keras_hub.models.Gemma3CausalLMPreprocessor.from_preset(
        "gemma3_instruct_1b"
    )

    # Unbatched inputs.
    preprocessor(
        {
            "prompts": "What is the capital of India?",
            "responses": "New Delhi",
        }
    )

    # Batched inputs.
    preprocessor(
        {
            "prompts": [
                "What is the capital of India?",
                "What is the capital of Spain?"
            ],
            "responses": ["New Delhi", "Madrid"],
        }
    )

    # Apply preprocessing to a `tf.data.Dataset`.
    features = {
        "prompts": [
            "What is the capital of India?",
            "What is the capital of Spain?"
        ],
        "responses": ["New Delhi", "Madrid"],
    }

    ds = tf.data.Dataset.from_tensor_slices(features)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Prepare tokens for generation (no end token).
    preprocessor.generate_preprocess(["The quick brown fox jumped."])

    # Map generation outputs back to strings.
    preprocessor.generate_postprocess({
        'token_ids': np.array([[2, 818, 3823, 8864, 37423, 32694, 236761, 0]]),
        'padding_mask': np.array([[ 1, 1, 1, 1, 1, 1, 1, 0]]),
    })

    # === Vision and Language Gemma3 model ===
    # Load the preprocessor from a preset.
    preprocessor = keras_hub.models.Gemma3CausalLMPreprocessor.from_preset(
        "gemma3_instruct_4b"
    )

    # text-only inputs (unbatched)
    preprocessor(
        {
            "prompts": "What is the capital of India?",
            "responses": "New Delhi",
        }
    )

    # text-only inputs (batched)
    preprocessor(
        {
            "prompts": [
                "What is the capital of India?",
                "What is the capital of Spain?"
            ],
            "responses": ["New Delhi", "Madrid"],
        }
    )

    # Unbatched inputs, with one image.
    preprocessor(
        {
            "prompts": "this is a lily <start_of_image>",
            "responses": "pristine!",
            "images": np.ones((896, 896, 3), dtype="float32")
        }
    )

    # Unbatched inputs, with two images.
    preprocessor(
        {
            "prompts": "lily: <start_of_image>, sunflower: <start_of_image>",
            "responses": "pristine!",
            "images": [
                np.ones((896, 896, 3), dtype="float32"),
                np.ones((896, 896, 3), dtype="float32")
            ],
        }
    )

    # Batched inputs, one image per prompt.
    preprocessor(
        {
            "prompts": [
                "this is a lily: <start_of_image>",
                "this is a sunflower: <start_of_image>"
            ],
            "responses": ["pristine!", "radiant!"],
            "images": [
                np.ones((896, 896, 3), dtype="float32"),
                np.ones((896, 896, 3), dtype="float32")
            ]
        }
    )

    # Can also be written this way.
    preprocessor(
        {
            "prompts": [
                "this is a lily: <start_of_image>",
                "this is a sunflower: <start_of_image>"
            ],
            "responses": ["pristine!", "radiant!"],
            "images": [
                [np.ones((896, 896, 3), dtype="float32")],
                [np.ones((896, 896, 3), dtype="float32")]
            ]
        }
    )

    # Different number of images in every sample.
    preprocessor(
        {
            "prompts": [
                "Who is this singer: <start_of_image>?",
                "Who are these musicians <start_of_image>, <start_of_image>?"
            ],
            "responses": ["Arijit Singh", "John Lennon, Paul Mccartney"],
            "images": [
                [
                    np.ones((896, 896, 3), dtype="float32"),
                    np.ones((896, 896, 3), dtype="float32")
                ],
                [np.ones((896, 896, 3), dtype="float32")]
            ]
        }
    )

    # Apply preprocessing to a `tf.data.Dataset`.
    inputs = {
        "prompts": [
            "Who are these two: <start_of_image>, <start_of_image>",
            "Who is this: <start_of_image>?",
            "What is the capital of India?"
        ],
        "responses": [
            "John Lennon, Paul Mccartney",
            "Arijit Singh",
            "New Delhi"
        ],
        "images": (
            tf.ragged.constant(
                [
                    [np.ones((10, 10, 3)), np.ones((10, 10, 3))],
                    [np.ones((10, 10, 3))],
                    [],
                ]
            )
        )
    }
    ds = tf.data.Dataset.from_tensor_slices(inputs)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
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

        # Ensure `max_images_per_prompt * num_vision_tokens_per_image` is
        # greater than `sequence_length`.
        if (
            image_converter is not None
            and sequence_length
            <= max_images_per_prompt * num_vision_tokens_per_image
        ):
            raise ValueError(
                "`sequence_length` should be greater than "
                "`max_images_per_prompt * num_vision_tokens_per_image`."
                f"Received: `sequence_length` = {sequence_length}"
                f"`max_images_per_prompt` = {max_images_per_prompt}"
                "`num_vision_tokens_per_image` = "
                f"{num_vision_tokens_per_image}"
            )

        self.image_converter = image_converter
        self.max_images_per_prompt = max_images_per_prompt
        self.num_vision_tokens_per_image = num_vision_tokens_per_image

        # The preprocessor and model are "text-only" if `self.image_converter`
        # is `None`.
        self.text_only_model = self.image_converter is None

        self.image_placeholder = self.tokenizer.image_placeholder
        self.start_of_image_token = self.tokenizer.start_of_image_token
        self.end_of_image_token = self.tokenizer.end_of_image_token

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

    def _get_vision_indices(self, vision_mask):
        """Computes indices given vision mask, and pads with 0.

        If `vision_mask` is

        ```
        [
            [False, True, True], [False, True, False], [False, False, False]
        ]
        ```

        , then the output will be:

        ```
        [
            [1, 2, 0], [1, 0, 0], [0, 0, 0]
        ]
        ```
        """
        batch_size, sequence_length = vision_mask.shape

        vision_mask_flattened = tf.reshape(vision_mask, [-1])
        vision_indices = tf.where(vision_mask_flattened)[..., 0]
        vision_indices = tf.cast(vision_indices, dtype=tf.int32)

        row_lengths = tf.math.reduce_sum(
            tf.cast(vision_mask, dtype=vision_indices.dtype), axis=1
        )

        batched_vision_indices = tf.RaggedTensor.from_row_lengths(
            values=vision_indices,
            row_lengths=row_lengths,
        )

        to_subtract = tf.math.scalar_mul(
            scalar=tf.cast(sequence_length, dtype=tf.int32),
            x=tf.range(
                start=0,
                limit=tf.shape(vision_mask)[0],
                dtype=tf.int32,
            ),
        )

        # All indices should be independent of other samples in the batch. If
        # not, and if we do sharding along the batch dimension for data
        # parallel, things might get weird.
        batched_vision_indices = tf.math.subtract(
            batched_vision_indices,
            tf.expand_dims(to_subtract, axis=-1),
        )

        # Pad the indices.
        batched_vision_indices = batched_vision_indices.to_tensor(
            shape=[
                batch_size,
                self.max_images_per_prompt * self.num_vision_tokens_per_image,
            ],
            default_value=0,
        )
        return batched_vision_indices

    def _format_output(
        self,
        images,
        token_ids,
        vision_mask,
        response_mask,
        padding_mask,
        return_labels=False,
        text_only_input=False,
        batched=False,
    ):
        if return_labels:
            # Target `y` will be the next token.
            y = token_ids[..., 1:]
            # Only compute the loss for labels in the response.
            sample_weight = response_mask[..., 1:]

            # The last token does not have a next token. So, remove it.
            token_ids = token_ids[..., :-1]
            vision_mask = vision_mask[..., :-1]
            response_mask = response_mask[..., :-1]
            padding_mask = padding_mask[..., :-1]

        batch_size = tf.shape(vision_mask)[0]

        if text_only_input:
            vision_indices = tf.ones(
                shape=[
                    batch_size,
                    0,
                ],
                dtype=tf.int32,
            )
        else:
            vision_indices = self._get_vision_indices(vision_mask=vision_mask)

        x = {
            # Image
            "images": images if batched else tf.squeeze(images, axis=0),
            # Text
            "token_ids": (
                token_ids if batched else tf.squeeze(token_ids, axis=0)
            ),
            "vision_indices": (
                vision_indices
                if batched
                else tf.squeeze(vision_indices, axis=0)
            ),
            # This mask is redundant information. But easier to compute it here
            # than the model forward pass.
            "vision_mask": (
                vision_mask if batched else tf.squeeze(vision_mask, axis=0)
            ),
            "padding_mask": (
                padding_mask if batched else tf.squeeze(padding_mask, axis=0)
            ),
        }

        if return_labels:
            if not batched:
                y = tf.squeeze(y, axis=0)
                sample_weight = tf.squeeze(sample_weight, 0)

            return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
        else:
            return x

    def _preprocess_images(self, images, batched):
        desired_height = self.image_converter.image_size[0]
        desired_width = self.image_converter.image_size[1]

        # Images can be lists/ragged tensors. We need to pad them/truncate them.
        if isinstance(images, (list, np.ndarray)):
            images = tf.ragged.constant(images)
        elif isinstance(images, tf.RaggedTensor):
            pass
        elif isinstance(images, tf.Tensor):
            images = tf.RaggedTensor.from_tensor(images)
        else:
            # Attempt to convert anyway. This handles the case where
            # the inputs might be `jax.Array`, `torch.Tensor`. To check the
            # type, we will have to import all three frameworks, which is
            # undesirable.
            try:
                images = tf.RaggedTensor.from_tensor(images)
            except:  # noqa: E722
                raise ValueError(
                    "`images` should be a list, ragged tensor, dense tensor."
                    f"Received: `type(images)` = {type(images)}"
                )

        if not batched:
            images = tf.expand_dims(images, axis=0)

        # If the input is a list of images, instead of list of lists of images.
        if len(images.shape) == 4:
            images = tf.expand_dims(images, axis=1)

        # Convert to dense tensor.
        images = images.to_tensor(
            shape=[None, self.max_images_per_prompt, None, None, 3],
            default_value=0,
        )

        # Resize, rescale, etc. the images.
        original_images_shape = tf.shape(images)

        # Before passing through image converter, we need to collapse the
        # first two dimensions (`batch_size`, `max_images_per_prompt`) into one.
        images = tf.reshape(
            images,
            [
                -1,
                original_images_shape[-3],
                original_images_shape[-2],
                original_images_shape[-1],
            ],
        )
        images = self.image_converter(images)

        if keras.config.backend() == "torch" and not isinstance(
            images, tf.Tensor
        ):
            images = images.cpu()

        # Recover the rank.
        images = tf.reshape(
            images,
            [
                original_images_shape[0],
                self.max_images_per_prompt,
                desired_height,
                desired_width,
                original_images_shape[-1],
            ],
        )
        return images

    @preprocessing_function
    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        sequence_length = sequence_length or self.sequence_length

        # === Input extraction and validation ===

        # Extract text part of the input.
        prompts, responses = x["prompts"], x["responses"]
        tf.debugging.assert_shapes([(prompts, ("N",)), (responses, ("N",))])

        # Find out if the input is batched/not batched. Uprank if not batched.
        # In other preprocessors, we don't have to do this, but here, all
        # the following logic (indices, etc.) uses tensors with a batch dim.
        # We will squeeze these back at the end.
        batched = True
        if isinstance(prompts, str):
            batched = False
            prompts = [prompts]
            responses = [responses]
        if isinstance(prompts, tf.Tensor) and len(prompts.shape) == 0:
            batched = False
            prompts = tf.expand_dims(prompts, axis=0)
            responses = tf.expand_dims(responses, axis=0)

        # Extract images from the input.
        images = x.get("images", None)

        # There are 8 cases, based on values of
        # a = `self.text_only_model`, b = `images` is `None`, and whether
        # c = `<start_of_image>` token is present in `prompts`.
        # F F F, F F T -> Raise error if #`<start_of_image>` <0,  or
        # > `max_images_per_prompt`.
        # F T F -> Return empty images and vision indices
        # F T T -> Return empty images and vision indices to the model.
        # T F F, T F T -> Raise error.
        # T T F -> Only token IDs and padding mask are returned.
        # T T T -> Only token IDs and padding mask are returned.

        if self.text_only_model and images is not None:
            raise ValueError(
                "The initialized preprocessor/model is text-only, but "
                " `images` is not `None`."
            )

        # Add image placeholder tokens. Replace `"<start_of_image>"` in
        # prompts with
        # `"\n\n<start_of_image> <img> * 256 <end_of_image>\n\n"`.
        if not self.text_only_model:
            prompts = tf.strings.regex_replace(
                prompts,
                self.start_of_image_token,
                f"\n\n{self.start_of_image_token}"
                + self.image_placeholder * self.num_vision_tokens_per_image
                + f"{self.end_of_image_token}\n\n",
            )

        # === Tokenization, padding, etc. ===

        # Tokenise the inputs.
        prompts = self.tokenizer(prompts)
        responses = self.tokenizer(responses)

        # Padding.
        token_ids, segment_ids = self.packer(
            (prompts, responses),
            sequence_length=sequence_length + 1,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        response_mask = segment_ids == 1
        padding_mask = token_ids != self.tokenizer.pad_token_id

        # === Text Model ===
        if self.text_only_model:
            # The last token does not have a next token, so we truncate it out.
            x = {
                "token_ids": token_ids[..., :-1],
                "padding_mask": padding_mask[..., :-1],
            }

            # Target `y` will be the next token.
            y = token_ids[..., 1:]
            # Only compute the loss for labels in the response.
            sample_weight = response_mask[..., 1:]

            # Squeeze if not batched.
            if not batched:
                x["token_ids"] = tf.squeeze(x["token_ids"], axis=0)
                x["padding_mask"] = tf.squeeze(x["padding_mask"], axis=0)
                y = tf.squeeze(y, axis=0)
                sample_weight = tf.squeeze(sample_weight, axis=0)

            return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

        # === Vision processing ===

        batch_size = tf.shape(prompts)[0]
        desired_height = self.image_converter.image_size[0]
        desired_width = self.image_converter.image_size[1]
        if images is None:
            # == Branch: vision model, with `None` value for `images` ==

            # To handle the text-only input case, we need to pass an empty
            # tensor so as to skip the vision layers of the model.

            # TODO: Once functional models accept `None` inputs, consider
            # passing this as `None` directly.
            images = tf.ones(
                shape=[
                    batch_size,
                    0,
                    desired_height,
                    desired_width,
                    3,
                ],
                dtype="float32",
            )

            vision_mask = tf.zeros_like(token_ids, dtype=bool)

            return self._format_output(
                images=images,
                token_ids=token_ids,
                vision_mask=vision_mask,
                response_mask=response_mask,
                padding_mask=padding_mask,
                return_labels=True,
                text_only_input=True,
                batched=batched,
            )

        # == Branch: vision model, with non-`None` value for `images` ==

        images = self._preprocess_images(images=images, batched=batched)

        vision_mask = token_ids == self.tokenizer.image_placeholder_id

        return self._format_output(
            images=images,
            token_ids=token_ids,
            vision_mask=vision_mask,
            response_mask=response_mask,
            padding_mask=padding_mask,
            return_labels=True,
            text_only_input=False,
            batched=batched,
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

        # Extract inputs.
        if isinstance(x, dict):
            images = x.get("images", None)

            # TODO: do we even need `responses` for generation? Makes sense for
            # finetuning only (i.e., `call()`).
            responses = x.get("responses", None)
            prompts = x["prompts"]
        else:
            images = None
            responses = None
            prompts = x

        # Find out if the input is batched/not batched. Uprank if not batched.
        # In other preprocessors, we don't have to do this, but here, all
        # the following logic (indices, etc.) uses tensors with a batch dim.
        # We will squeeze these back at the end.
        batched = True
        if isinstance(prompts, str):
            batched = False
            prompts = [prompts]
            if responses is not None:
                responses = [responses]
        if isinstance(prompts, tf.Tensor) and len(prompts.shape) == 0:
            batched = False
            prompts = tf.expand_dims(prompts, axis=0)
            if responses is not None:
                responses = tf.expand_dims(responses, axis=0)

        # We have the same 8 cases here, as in `call()`.
        if self.text_only_model and images is not None:
            raise ValueError(
                "The initialized preprocessor/model is text-only, but "
                " `images` is not `None`."
            )

        # Add image placeholder tokens. Replace `"<start_of_image>"` in
        # prompts with
        # `"\n\n<start_of_image> <img> * 256 <end_of_image>\n\n"`.
        if not self.text_only_model:
            prompts = tf.strings.regex_replace(
                prompts,
                self.start_of_image_token,
                f"\n\n{self.start_of_image_token}"
                + self.image_placeholder * self.num_vision_tokens_per_image
                + f"{self.end_of_image_token}\n\n",
            )

        # === Tokenization, padding, etc. ===
        prompts = self.tokenizer(prompts)

        if responses is not None:
            responses = self.tokenizer(responses)
            segments = (prompts, responses)
        else:
            segments = (prompts,)

        # Padding.
        token_ids, segment_ids = self.packer(
            segments,
            sequence_length=sequence_length,
            add_end_value=False,
        )
        response_mask = segment_ids == 1
        padding_mask = token_ids != self.tokenizer.pad_token_id

        # === Text Model ===
        if self.text_only_model:
            return {
                "token_ids": (
                    token_ids if batched else tf.squeeze(token_ids, axis=0)
                ),
                "padding_mask": (
                    padding_mask
                    if batched
                    else tf.squeeze(padding_mask, axis=0)
                ),
            }

        # === Vision processing ===

        batch_size = tf.shape(prompts)[0]
        desired_height = self.image_converter.image_size[0]
        desired_width = self.image_converter.image_size[1]
        if images is None:
            # == Branch: vision model, with `None` value for `images` ==

            # To handle the text-only input case, we need to pass an empty
            # tensor so as to skip the vision layers of the model.

            # TODO: Once functional models accept `None` inputs, consider
            # passing this as `None` directly.
            images = tf.ones(
                shape=[
                    batch_size,
                    0,
                    desired_height,
                    desired_width,
                    3,
                ],
                dtype="float32",
            )

            vision_mask = tf.zeros_like(token_ids, dtype=bool)

            return self._format_output(
                images=images,
                token_ids=token_ids,
                vision_mask=vision_mask,
                response_mask=response_mask,
                padding_mask=padding_mask,
                return_labels=False,
                text_only_input=True,
                batched=batched,
            )

        # == Branch: vision model, with non-`None` value for `images` ==
        images = self._preprocess_images(images=images, batched=batched)

        vision_mask = token_ids == self.tokenizer.image_placeholder_id

        return self._format_output(
            images=images,
            token_ids=token_ids,
            vision_mask=vision_mask,
            response_mask=response_mask,
            padding_mask=padding_mask,
            return_labels=False,
            text_only_input=False,
            batched=batched,
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

        # We do not want to strip SoI token because it is provided by the user.
        if self.tokenizer.start_of_image_token_id in ids_to_strip:
            ids_to_strip.remove(self.tokenizer.start_of_image_token_id)

        token_ids = strip_to_ragged(token_ids, padding_mask, ids_to_strip)
        return self.tokenizer.detokenize(token_ids)

    @property
    def max_images_per_prompt(self):
        return self._max_images_per_prompt

    @max_images_per_prompt.setter
    def max_images_per_prompt(self, value):
        self._max_images_per_prompt = value
