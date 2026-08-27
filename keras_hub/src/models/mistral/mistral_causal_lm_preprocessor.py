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
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
except ImportError:
    tf = None


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
            Defaults to `2`.

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

    def _compute_image_block_ids(self, height, width):
        """Builds the token-ID block a single image expands to.

        Mirrors HF's Pixtral/Mistral3 processor: an image contributes one
        `image_placeholder_token_id` per merged vision-patch row/column,
        each row terminated by `image_break_token_id`, with the last row's
        trailing break token swapped for `image_end_token_id`.

        Args:
            height: int. The image's resized height, in pixels.
            width: int. The image's resized width, in pixels.

        Returns:
            list of int. The token IDs this image expands to.
        """
        merge = self.image_converter.patch_size * self.spatial_merge_size
        num_width_tokens = width // merge
        num_height_tokens = height // merge
        row = [self.tokenizer.image_placeholder_token_id] * num_width_tokens
        row.append(self.tokenizer.image_break_token_id)
        block = row * num_height_tokens
        block[-1] = self.tokenizer.image_end_token_id
        return block

    def _tokenize_with_image_blocks(self, prompt, image_sizes):
        """Tokenizes `prompt`, splicing in each image's block token ids.

        Tokenizes the whole, unexpanded `prompt` in one call rather than
        per-segment, since SentencePiece's leading-space handling differs
        per call and wouldn't reproduce whole-string tokenization.

        Args:
            prompt: str. The raw prompt, containing zero or more literal
                occurrences of the image placeholder token.
            image_sizes: list of `(height, width)` tuples, one per
                occurrence of the placeholder token in `prompt`, in order.

        Returns:
            list of int. The complete token ID sequence for `prompt`.
        """
        base_ids = self.tokenizer(prompt)
        if hasattr(base_ids, "numpy"):
            base_ids = base_ids.numpy().tolist()
        else:
            base_ids = list(base_ids)

        placeholder_id = self.tokenizer.image_placeholder_token_id
        num_occurrences = base_ids.count(placeholder_id)
        if num_occurrences != len(image_sizes):
            raise ValueError(
                "The number of image placeholder token occurrences in "
                "`prompt` must match `len(image_sizes)`. Received: "
                f"{num_occurrences} occurrences in {prompt!r}, but "
                f"`image_sizes` has length {len(image_sizes)}."
            )

        token_ids = []
        image_idx = 0
        for token_id in base_ids:
            if token_id == placeholder_id:
                height, width = image_sizes[image_idx]
                token_ids.extend(self._compute_image_block_ids(height, width))
                image_idx += 1
            else:
                token_ids.append(token_id)
        return token_ids

    def _build_multimodal_inputs(self, prompts, images_per_prompt):
        """Tokenizes prompts and produces vision model inputs.

        Args:
            prompts: list of str. One raw prompt per batch element, possibly
                containing literal image-placeholder-token occurrences to
                expand.
            images_per_prompt: list of lists of raw images (each
                `(height, width, 3)`), the same length as `prompts`. A
                prompt with no images uses an empty list.

        Returns:
            A tuple `(tokenized, pixel_values, image_sizes)`. When the batch
            has no images, `tokenized` is `prompts` unchanged and
            `pixel_values`/`image_sizes` are `None` (matching HF, which
            skips the vision tower entirely rather than passing it an
            empty batch). Otherwise, `tokenized` is a list of per-example
            token ID lists, with each image placeholder already expanded.
        """
        # Flatten into one ordered list, batch-row-major then per-prompt
        # left-to-right. This order must match the order images are
        # consumed while expanding placeholders, the order
        # `Mistral3ImageConverter` processes them in, and the order
        # features get scattered back into token positions.
        flat_images = []
        for images in images_per_prompt:
            flat_images.extend(images)

        if len(flat_images) == 0:
            return list(prompts), None, None

        pixel_values, image_sizes = self.image_converter(flat_images)

        tokenized = []
        offset = 0
        for prompt, images in zip(prompts, images_per_prompt):
            num_images = len(images)
            sizes_slice = [
                tuple(image_sizes[offset + i]) for i in range(num_images)
            ]
            offset += num_images
            tokenized.append(
                self._tokenize_with_image_blocks(prompt, sizes_slice)
            )
        return tokenized, pixel_values, image_sizes

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
        elif tf is not None and isinstance(prompts, tf.Tensor):
            # `@preprocessing_function` converts raw inputs to `tf.Tensor`s
            # eagerly; decode back to Python strings, since image-placeholder
            # expansion below is per-example and variable-length.
            if prompts.shape.rank == 0:
                batched = False
                prompts = [prompts]
                if images is not None:
                    images = [images]
            prompts = [p.numpy().decode("utf-8") for p in prompts]

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
        tokenized, pixel_values, image_sizes = self._build_multimodal_inputs(
            prompts, images_per_prompt
        )
        if pixel_values is None:
            raise ValueError(
                "Mistral3's multimodal preprocessor requires at least one "
                "image per batch when `image_converter` is set; got a "
                "batch with zero images."
            )

        tokenized = tf.ragged.constant(tokenized, dtype="int32")
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

    @preprocessing_function
    def generate_preprocess(
        self,
        x,
        sequence_length=None,
    ):
        """Convert strings to integer token ids for generation.

        `x` may be a string, list of strings, or a dict with a `"prompts"`
        key (and, for multimodal presets, an `"images"` key). Returns a dict
        with `token_ids` and `padding_mask`, plus `pixel_values`,
        `image_sizes`, and `placeholder_indices` when images are present.
        """
        images = x.get("images") if isinstance(x, dict) else None
        if images is None:
            if isinstance(x, dict):
                x = x["prompts"]
            return super().generate_preprocess(
                x, sequence_length=sequence_length
            )

        if self.text_only_model:
            raise ValueError(
                "The initialized preprocessor/model is text-only, but "
                "`images` is not `None`."
            )
        if not self.built:
            self.build(None)

        prompts, images_per_prompt, batched = self._extract_multimodal_inputs(x)
        tokenized, pixel_values, image_sizes = self._build_multimodal_inputs(
            prompts, images_per_prompt
        )
        if pixel_values is not None:
            tokenized = tf.ragged.constant(tokenized, dtype="int32")
        else:
            tokenized = self.tokenizer(tokenized)
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
