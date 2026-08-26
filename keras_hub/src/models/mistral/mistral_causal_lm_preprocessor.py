import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.mistral.mistral_backbone import MistralBackbone
from keras_hub.src.models.mistral.mistral_image_converter import (
    Mistral3ImageConverter,
)
from keras_hub.src.models.mistral.mistral_tokenizer import MistralTokenizer
from keras_hub.src.models.mistral.mistral_vision_encoder import (
    MISTRAL3_DEFAULT_SPATIAL_MERGE_SIZE,
)
from keras_hub.src.models.mistral.mistral_vision_encoder import (
    compute_image_placeholder_indices,
)


def _expand_image_placeholders(
    prompt,
    image_sizes,
    patch_size,
    spatial_merge_size,
    image_token,
    image_break_token,
    image_end_token,
):
    """Expands each `image_token` occurrence in `prompt` into an image block.

    Mirrors HF's Pixtral/Mistral3 processor: every literal occurrence of
    `image_token` in `prompt` is replaced by a grid of `image_token`s (one
    per merged vision-patch row/column for that image), with each row
    terminated by `image_break_token`, and the very last row's trailing
    `image_break_token` swapped for `image_end_token`.

    Args:
        prompt: str. The raw prompt, containing zero or more literal
            occurrences of `image_token`.
        image_sizes: list of `(height, width)` tuples, one per occurrence of
            `image_token` in `prompt`, in order.
        patch_size: int. The vision encoder's patch size.
        spatial_merge_size: int. The multimodal projector's spatial merge
            size.
        image_token: str. The literal image placeholder token, e.g.
            `"[IMG]"`.
        image_break_token: str. The literal row-break token, e.g.
            `"[IMG_BREAK]"`.
        image_end_token: str. The literal image-end token, e.g.
            `"[IMG_END]"`.

    Returns:
        The expanded prompt string.
    """
    segments = prompt.split(image_token)
    num_occurrences = len(segments) - 1
    if num_occurrences != len(image_sizes):
        raise ValueError(
            "The number of `image_token` occurrences in `prompt` must "
            "match `len(image_sizes)`. Received: "
            f"{num_occurrences} occurrences of {image_token!r} in "
            f"{prompt!r}, but `image_sizes` has length {len(image_sizes)}."
        )

    merge = patch_size * spatial_merge_size
    blocks = []
    for height, width in image_sizes:
        num_width_tokens = width // merge
        num_height_tokens = height // merge
        row = image_token * num_width_tokens + image_break_token
        block = row * num_height_tokens
        # Swap the final row's trailing break token for the end token.
        block = block[: -len(image_break_token)] + image_end_token
        blocks.append(block)

    expanded = segments[0]
    for segment, block in zip(segments[1:], blocks):
        expanded += block + segment
    return expanded


@keras_hub_export("keras_hub.models.MistralCausalLMPreprocessor")
class MistralCausalLMPreprocessor(CausalLMPreprocessor):
    """Mistral Causal LM preprocessor.

    This preprocessing layer is meant for use with
    `keras_hub.models.MistralCausalLM`. By default, it will take in batches of
    strings, and return outputs in a `(x, y, sample_weight)` format, where the
    `y` label is the next token id in the `x` sequence.

    For use with generation, the layer also exposes two methods
    `generate_preprocess()` and `generate_postprocess()`. When this preprocessor
    is attached to a `keras_hub.models.MistralCausalLM` instance, these methods
    will be called implicitly in `generate()`. They can also be called
    standalone (e.g. to precompute preprocessing inputs for generation in a
    separate process).

    This preprocessor can also be configured for multimodal (image + text)
    use with the Mistral3/Pixtral architecture, by passing an
    `image_converter`. In that case, `x` for `call()`/`generate_preprocess()`
    should be a dict with `"prompts"` (and optionally `"responses"`) and
    `"images"` keys, where `"images"` is a list (one entry per prompt) of
    lists of raw images (each `(height, width, 3)`).

    Args:
        tokenizer: A `keras_hub.models.MistralTokenizer` instance.
        image_converter: A `keras_hub.layers.Mistral3ImageConverter`
            instance. If `None` (the default), this preprocessor is
            text-only.
        sequence_length: The length of the packed inputs.
        add_start_token: If `True`, the preprocessor will prepend the tokenizer
            start token to each input sequence. Default is `True`.
        add_end_token: If `True`, the preprocessor will append the tokenizer
            end token to each input sequence. Default is `True`.
        spatial_merge_size: int. The multimodal projector's spatial merge
            size, used to compute how many image placeholder tokens each
            image expands to. Only used when `image_converter` is set.
            Defaults to `2`, matching `Mistral3PatchMerger`'s default.

    Call arguments:
        x: A string, `tf.Tensor` or list of python strings, or (when
            configured for multimodal use) a dict with `"prompts"` and
            `"images"` keys.
        y: Label data. Should always be `None` as the layer generates labels.
        sample_weight: Label weights. Should always be `None` as the layer
            generates label weights.
        sequence_length: Pass to override the configured `sequence_length` of
            the layer.

    Examples:
    ```python
    # Load the preprocessor from a preset.
    preprocessor = keras_hub.models.MistralCausalLMPreprocessor.from_preset(
        "mistral_base_en"
    )

    # Tokenize and pack a single sentence.
    sentence = tf.constant("League of legends")
    preprocessor(sentence)
    # Same output.
    preprocessor("League of legends")

    # Tokenize a batch of sentences.
    sentences = tf.constant(["Taco tuesday", "Fish taco please!"])
    preprocessor(sentences)
    # Same output.
    preprocessor(["Taco tuesday", "Fish taco please!"])

    # Map a dataset to preprocess a single sentence.
    features = tf.constant(
        [
            "Avatar 2 is amazing!",
            "Well, I am not sure.",
        ]
    )
    labels = tf.constant([1, 0])
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Map a dataset to preprocess unlabled sentences.
    ds = tf.data.Dataset.from_tensor_slices(features)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """

    backbone_cls = MistralBackbone
    tokenizer_cls = MistralTokenizer
    image_converter_cls = Mistral3ImageConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        sequence_length=1024,
        add_start_token=True,
        add_end_token=True,
        spatial_merge_size=MISTRAL3_DEFAULT_SPATIAL_MERGE_SIZE,
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
        self.spatial_merge_size = spatial_merge_size
        # The preprocessor and model are "text-only" if `self.image_converter`
        # is `None`.
        self.text_only_model = self.image_converter is None
        if not self.text_only_model and not tokenizer.has_vision_tokens:
            raise ValueError(
                "`MistralCausalLMPreprocessor` was given an `image_converter`, "
                "but its `tokenizer` was not built with `has_vision_tokens="
                "True`. The multimodal preprocessing path needs the "
                "tokenizer's `image_placeholder_token`/`image_break_token`/"
                "`image_end_token` special tokens to expand image "
                "placeholders in prompts."
            )

    def _build_multimodal_inputs(self, prompts, images_per_prompt):
        """Expands image placeholders and produces vision model inputs.

        Args:
            prompts: list of str. One raw prompt per batch element, possibly
                containing literal `image_token` occurrences to expand.
            images_per_prompt: list of lists of raw images (each
                `(height, width, 3)`), the same length as `prompts`. A
                prompt with no images uses an empty list.

        Returns:
            A tuple `(expanded_prompts, pixel_values, image_sizes)`.
            `pixel_values`/`image_sizes` are `None` when the batch has no
            images at all, matching HF's `Mistral3Model.forward()`, which
            only invokes the vision tower `if pixel_values is not None`
            rather than feeding it an empty/dummy batch.
        """
        # Flatten all images across all prompts into one ordered list,
        # batch-row-major then per-prompt left-to-right. This exact order
        # must match: the order images are consumed while expanding each
        # prompt's placeholders, the order `Mistral3ImageConverter`
        # processes them in, and the order features get scattered back into
        # token positions. There is no padding slot to absorb a mismatch.
        flat_images = []
        for images in images_per_prompt:
            flat_images.extend(images)

        if len(flat_images) == 0:
            return list(prompts), None, None

        pixel_values, image_sizes = self.image_converter(flat_images)

        expanded_prompts = []
        offset = 0
        for prompt, images in zip(prompts, images_per_prompt):
            num_images = len(images)
            sizes_slice = [
                tuple(image_sizes[offset + i]) for i in range(num_images)
            ]
            offset += num_images
            expanded_prompts.append(
                _expand_image_placeholders(
                    prompt,
                    sizes_slice,
                    patch_size=self.image_converter.patch_size,
                    spatial_merge_size=self.spatial_merge_size,
                    image_token=self.tokenizer.image_placeholder_token,
                    image_break_token=self.tokenizer.image_break_token,
                    image_end_token=self.tokenizer.image_end_token,
                )
            )
        return expanded_prompts, pixel_values, image_sizes

    def _extract_multimodal_inputs(self, x):
        """Normalizes `x` into `(prompts, images_per_prompt, batched)`."""
        if isinstance(x, dict):
            prompts = x["prompts"]
            images = x.get("images", None)
        else:
            prompts = x
            images = None

        batched = True
        if isinstance(prompts, str):
            batched = False
            prompts = [prompts]
            if images is not None:
                images = [images]

        if images is None:
            images_per_prompt = [[] for _ in prompts]
        else:
            images_per_prompt = [
                list(per_prompt_images) for per_prompt_images in images
            ]
        return list(prompts), images_per_prompt, batched

    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        if self.text_only_model:
            return super().call(
                x,
                y=y,
                sample_weight=sample_weight,
                sequence_length=sequence_length,
            )

        sequence_length = sequence_length or self.sequence_length
        prompts, images_per_prompt, batched = self._extract_multimodal_inputs(x)
        prompts, pixel_values, image_sizes = self._build_multimodal_inputs(
            prompts, images_per_prompt
        )
        if pixel_values is None:
            raise ValueError(
                "Mistral3's multimodal preprocessor requires at least one "
                "image per batch when `image_converter` is set; got a "
                "batch with zero images."
            )

        tokenized = self.tokenizer(prompts)
        # Pad with one extra token to account for the truncation below.
        token_ids, padding_mask = self.packer(
            tokenized,
            sequence_length=sequence_length + 1,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )

        model_token_ids = token_ids[..., :-1]
        model_padding_mask = padding_mask[..., :-1]
        placeholder_indices = compute_image_placeholder_indices(
            keras.ops.convert_to_numpy(model_token_ids),
            self.tokenizer.image_placeholder_token_id,
        )[None, :]

        out_x = {
            "token_ids": model_token_ids,
            "padding_mask": model_padding_mask,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
            "placeholder_indices": placeholder_indices,
        }
        # Target `y` will be the next token.
        y = token_ids[..., 1:]
        sample_weight = padding_mask[..., 1:]

        if not batched:
            out_x["token_ids"] = keras.ops.squeeze(out_x["token_ids"], axis=0)
            out_x["padding_mask"] = keras.ops.squeeze(
                out_x["padding_mask"], axis=0
            )
            y = keras.ops.squeeze(y, axis=0)
            sample_weight = keras.ops.squeeze(sample_weight, axis=0)

        return keras.utils.pack_x_y_sample_weight(out_x, y, sample_weight)

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
        if self.text_only_model:
            return super().generate_preprocess(
                x, sequence_length=sequence_length
            )

        if not self.built:
            self.build(None)

        prompts, images_per_prompt, batched = self._extract_multimodal_inputs(x)
        prompts, pixel_values, image_sizes = self._build_multimodal_inputs(
            prompts, images_per_prompt
        )

        tokenized = self.tokenizer(prompts)
        token_ids, padding_mask = self.packer(
            tokenized, sequence_length=sequence_length, add_end_value=False
        )

        out_x = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }
        if pixel_values is not None:
            placeholder_indices = compute_image_placeholder_indices(
                keras.ops.convert_to_numpy(token_ids),
                self.tokenizer.image_placeholder_token_id,
            )[None, :]
            out_x["pixel_values"] = pixel_values
            out_x["image_sizes"] = image_sizes
            out_x["placeholder_indices"] = placeholder_indices
        if not batched:
            out_x["token_ids"] = keras.ops.squeeze(out_x["token_ids"], axis=0)
            out_x["padding_mask"] = keras.ops.squeeze(
                out_x["padding_mask"], axis=0
            )
        return out_x

    def get_config(self):
        config = super().get_config()
        config.update({"spatial_merge_size": self.spatial_merge_size})
        return config
