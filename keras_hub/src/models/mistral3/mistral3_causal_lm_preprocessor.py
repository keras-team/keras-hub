import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.mistral3.mistral3_backbone import Mistral3Backbone
from keras_hub.src.models.mistral3.mistral3_image_converter import (
    Mistral3ImageConverter,
)
from keras_hub.src.models.mistral3.mistral3_tokenizer import Mistral3Tokenizer
from keras_hub.src.models.mistral3.mistral3_vision_encoder import (
    MISTRAL3_DEFAULT_SPATIAL_MERGE_SIZE,
)
from keras_hub.src.models.mistral3.mistral3_vision_encoder import (
    compute_image_placeholder_indices,
)
from keras_hub.src.utils.tensor_utils import convert_to_numpy
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.Mistral3CausalLMPreprocessor")
class Mistral3CausalLMPreprocessor(CausalLMPreprocessor):
    """Mistral3 Causal LM preprocessor.

    This preprocessing layer is meant for use with
    `keras_hub.models.Mistral3CausalLM`. It takes in batches of prompts and
    (optionally, per-prompt) images and returns outputs in a
    `(x, y, sample_weight)` format, where the `y` label is the next token id
    in the `x` sequence.

    `x` for `call()`/`generate_preprocess()` should be a dict with
    `"prompts"` (and optionally `"responses"`) and, for multimodal inputs, an
    `"images"` key. Images are matched to prompts by their `"[IMG]"`
    placeholder occurrences, consumed in order — `"images"` can be any
    reasonable nesting (a single image, a batched array, flat or
    per-prompt-grouped lists), as long as the total image count matches the
    total placeholder count. Omitting `"images"` (or passing `x` as a plain
    string/list of strings) preprocesses as plain text, matching HF's
    `Mistral3ForConditionalGeneration`, which also supports text-only calls.

    For use with generation, the layer also exposes two methods
    `generate_preprocess()` and `generate_postprocess()`. When this preprocessor
    is attached to a `keras_hub.models.Mistral3CausalLM` instance, these methods
    will be called implicitly in `generate()`. They can also be called
    standalone (e.g. to precompute preprocessing inputs for generation in a
    separate process).

    Args:
        tokenizer: A `keras_hub.models.Mistral3Tokenizer` instance.
        image_converter: A `keras_hub.layers.Mistral3ImageConverter`
            instance.
        sequence_length: The length of the packed inputs.
        add_start_token: If `True`, the preprocessor will prepend the tokenizer
            start token to each input sequence. Default is `True`.
        add_end_token: If `True`, the preprocessor will append the tokenizer
            end token to each input sequence. Default is `True`.
        spatial_merge_size: int. The multimodal projector's spatial merge
            size, used to compute how many image placeholder tokens each
            image expands to. Defaults to `2`.

    Call arguments:
        x: A dict with `"prompts"` and, optionally, `"images"` keys.
        y: Label data. Should always be `None` as the layer generates labels.
        sample_weight: Label weights. Should always be `None` as the layer
            generates label weights.
        sequence_length: Pass to override the configured `sequence_length` of
            the layer.
    """

    backbone_cls = Mistral3Backbone
    tokenizer_cls = Mistral3Tokenizer
    image_converter_cls = Mistral3ImageConverter

    def __init__(
        self,
        tokenizer,
        image_converter,
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

    def _tokenize_base(self, prompt):
        """Tokenizes `prompt` whole (not split around placeholders), since
        SentencePiece's leading-space handling differs per call.

        Args:
            prompt: str. The raw prompt text.

        Returns:
            list of int. `prompt`'s token IDs, placeholders not expanded.
        """
        base_ids = self.tokenizer(prompt)
        return convert_to_numpy(base_ids).tolist()

    def _expand_image_blocks(self, base_ids, image_sizes):
        """Splices each image's block token ids into `base_ids`.

        Args:
            base_ids: list of int, from `_tokenize_base`.
            image_sizes: list of `(height, width)` tuples, one per
                placeholder occurrence in `base_ids`, in order.

        Returns:
            list of int. The complete token ID sequence.
        """
        placeholder_id = self.tokenizer.image_placeholder_token_id
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

    def _tokenize_multimodal_prompts(self, prompts, image_sizes):
        """Tokenizes `prompts` and splices in each image's block token ids.

        Args:
            prompts: list of str.
            image_sizes: list of `(height, width)` tuples, one per
                placeholder occurrence across `prompts`, in order.

        Returns:
            list of list of int. Token ids per prompt, placeholders
            expanded.
        """
        placeholder_id = self.tokenizer.image_placeholder_token_id
        base_ids_per_prompt = [self._tokenize_base(p) for p in prompts]
        occurrence_counts = [
            base_ids.count(placeholder_id) for base_ids in base_ids_per_prompt
        ]
        total_occurrences = sum(occurrence_counts)
        if total_occurrences != len(image_sizes):
            raise ValueError(
                "The total number of image placeholder token occurrences "
                "across `prompts` must match the number of images "
                f"provided. Received: {total_occurrences} occurrence(s) "
                f"across {len(prompts)} prompt(s), but {len(image_sizes)} "
                "image(s)."
            )

        tokenized = []
        offset = 0
        for base_ids, num_occurrences in zip(
            base_ids_per_prompt, occurrence_counts
        ):
            sizes_slice = image_sizes[offset : offset + num_occurrences]
            offset += num_occurrences
            tokenized.append(self._expand_image_blocks(base_ids, sizes_slice))
        return tokenized

    def _convert_images(self, flat_images):
        """Runs `self.image_converter`, or signals an empty image batch.

        Args:
            flat_images: list of raw images, or a `tf.Tensor` stacking
                them on its leading axis (see `_flatten_images`).

        Returns:
            `(pixel_values, image_sizes)`, or `(None, None)` if
            `flat_images` is empty.
        """
        # `len()` fails on a `tf.Tensor` with an unknown leading dim; use
        # the static shape, treating unknown as non-empty.
        if isinstance(flat_images, list):
            num_images = len(flat_images)
        else:
            num_images = flat_images.shape[0]
        if num_images == 0:
            return None, None
        return self.image_converter(flat_images)

    def _build_multimodal_inputs(self, prompts, flat_images):
        """Tokenizes prompts and produces vision model inputs.

        Images are matched to prompts by consuming `flat_images`
        left-to-right as placeholder tokens are encountered, not by any
        caller-supplied grouping. Tokenization runs inside `tf.py_function`
        since it needs concrete Python values, which `prompts` may not be
        (e.g. inside `tf.data.Dataset.map`).

        Args:
            prompts: list of str, or a `tf.Tensor` of str. The raw prompts.
            flat_images: list of raw images, or a `tf.Tensor` stacking
                them on its leading axis, in placeholder-occurrence order
                across `prompts`.

        Returns:
            `(tokenized, pixel_values, image_sizes)`. For an image-free
            batch, `tokenized` is `prompts` unchanged and
            `pixel_values`/`image_sizes` are `None`. Otherwise `tokenized`
            is a ragged int32 tensor of token ids.
        """
        pixel_values, image_sizes = self._convert_images(flat_images)
        if pixel_values is None:
            return prompts, None, None

        def _encode(prompts_tensor, image_sizes_tensor):
            prompts_list = [p.decode("utf-8") for p in prompts_tensor.numpy()]
            image_sizes_list = [
                tuple(size) for size in image_sizes_tensor.numpy().tolist()
            ]
            tokenized = self._tokenize_multimodal_prompts(
                prompts_list, image_sizes_list
            )
            return tf.ragged.constant(tokenized, dtype="int32")

        prompts_tensor = (
            prompts
            if isinstance(prompts, tf.Tensor)
            else tf.constant(prompts, dtype=tf.string)
        )
        tokenized = tf.py_function(
            _encode,
            [prompts_tensor, image_sizes],
            Tout=tf.RaggedTensorSpec(
                shape=[None, None], dtype="int32", ragged_rank=1
            ),
        )
        return tokenized, pixel_values, image_sizes

    def _flatten_images(self, images):
        """Flattens `images` so all images sit on one leading axis.

        A `tf.Tensor` is folded via `tf.reshape` (graph safe); arbitrary
        Python nesting is flattened by iteration (eager only).

        Returns:
            Either a list of individual images or a single tensor
            stacking every image on its leading axis. Both support
            `len()` and are accepted by `self.image_converter`.
        """
        if images is None:
            return []
        if tf is not None and isinstance(images, tf.Tensor):
            if images.shape.rank == 3:
                return tf.expand_dims(images, axis=0)
            image_shape = images.shape[-3:].as_list()
            return tf.reshape(images, [-1] + image_shape)
        if hasattr(images, "shape") and len(images.shape) == 3:
            return [images]
        if hasattr(images, "shape") and len(images.shape) == 4:
            return list(images)
        flat_images = []
        for item in images:
            flat_images.extend(self._flatten_images(item))
        return flat_images

    def _build_multimodal_outputs(self, prompts, image_sizes, sequence_length):
        """Builds `_call_multimodal_python`'s per-example outputs.

        Tokenization, packing, and placeholder-index computation each
        depend on the previous step's concrete output, so they run
        together in one `tf.py_function`.

        Args:
            prompts: list of str, or a `tf.Tensor` of str.
            image_sizes: int tensor `(num_images, 2)`, from
                `self.image_converter`.
            sequence_length: int.

        Returns:
            `(model_token_ids, model_padding_mask, y, sample_weight,
            placeholder_indices)`.
        """

        def _build(prompts_tensor, image_sizes_tensor):
            prompts_list = [p.decode("utf-8") for p in prompts_tensor.numpy()]
            image_sizes_list = [
                tuple(size) for size in image_sizes_tensor.numpy().tolist()
            ]
            tokenized = self._tokenize_multimodal_prompts(
                prompts_list, image_sizes_list
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
            y = token_ids[..., 1:]
            sample_weight = padding_mask[..., 1:]
            placeholder_indices = compute_image_placeholder_indices(
                convert_to_numpy(model_token_ids),
                self.tokenizer.image_placeholder_token_id,
            )
            return (
                model_token_ids,
                model_padding_mask,
                y,
                sample_weight,
                placeholder_indices,
            )

        prompts_tensor = (
            prompts
            if isinstance(prompts, tf.Tensor)
            else tf.constant(prompts, dtype=tf.string)
        )
        (
            model_token_ids,
            model_padding_mask,
            y,
            sample_weight,
            placeholder_indices,
        ) = tf.py_function(
            _build,
            [prompts_tensor, image_sizes],
            Tout=[tf.int32, tf.bool, tf.int32, tf.bool, tf.int32],
        )
        # `tf.py_function` outputs have unknown rank unless set explicitly,
        # which breaks the model's shape inference. `placeholder_indices`'
        # last dim is data-dependent, so only its rank is fixed.
        model_token_ids.set_shape([None, sequence_length])
        model_padding_mask.set_shape([None, sequence_length])
        y.set_shape([None, sequence_length])
        sample_weight.set_shape([None, sequence_length])
        placeholder_indices.set_shape([None, None])
        return (
            model_token_ids,
            model_padding_mask,
            y,
            sample_weight,
            placeholder_indices,
        )

    def _extract_multimodal_inputs(self, x):
        """Normalizes `x` into `(prompts, flat_images, batched)`."""
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
        elif tf is not None and isinstance(prompts, tf.Tensor):
            if prompts.shape.rank == 0:
                batched = False
                prompts = tf.expand_dims(prompts, 0)
        else:
            prompts = list(prompts)

        return prompts, self._flatten_images(images), batched

    def _call_multimodal_python(
        self, x, y=None, sample_weight=None, sequence_length=None
    ):
        sequence_length = sequence_length or self.sequence_length
        prompts, flat_images, batched = self._extract_multimodal_inputs(x)
        pixel_values, image_sizes = self._convert_images(flat_images)
        if pixel_values is None:
            raise ValueError(
                'Mistral3\'s preprocessor was passed an `"images"` key but '
                "found zero images across the batch."
            )

        (
            model_token_ids,
            model_padding_mask,
            y,
            sample_weight,
            placeholder_indices,
        ) = self._build_multimodal_outputs(
            prompts, image_sizes, sequence_length
        )

        out_x = {
            "token_ids": model_token_ids,
            "padding_mask": model_padding_mask,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
            "placeholder_indices": placeholder_indices,
        }

        if not batched:
            out_x["token_ids"] = keras.ops.squeeze(out_x["token_ids"], axis=0)
            out_x["padding_mask"] = keras.ops.squeeze(
                out_x["padding_mask"], axis=0
            )
            y = keras.ops.squeeze(y, axis=0)
            sample_weight = keras.ops.squeeze(sample_weight, axis=0)

        return keras.utils.pack_x_y_sample_weight(out_x, y, sample_weight)

    @preprocessing_function
    def _call_multimodal_tf(
        self, x, y=None, sample_weight=None, sequence_length=None
    ):
        return self._call_multimodal_python(
            x,
            y=y,
            sample_weight=sample_weight,
            sequence_length=sequence_length,
        )

    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        images = x.get("images") if isinstance(x, dict) else None
        if images is None:
            # Mistral3 (like the HF model it wraps) supports plain text-only
            # calls: no image inputs are added to the output in that case.
            prompts = x["prompts"] if isinstance(x, dict) else x
            return super().call(
                prompts,
                y=y,
                sample_weight=sample_weight,
                sequence_length=sequence_length,
            )

        if not self._allow_python_workflow or in_tf_function():
            return self._call_multimodal_tf(
                x,
                y=y,
                sample_weight=sample_weight,
                sequence_length=sequence_length,
            )
        return self._call_multimodal_python(
            x,
            y=y,
            sample_weight=sample_weight,
            sequence_length=sequence_length,
        )

    @preprocessing_function
    def generate_preprocess(
        self,
        x,
        sequence_length=None,
    ):
        """Convert prompts (and optional images) to model inputs for generation.

        `x` may be a string, list of strings, or a dict with a `"prompts"`
        key and an `"images"` key. Returns a dict with `token_ids` and
        `padding_mask`, plus `pixel_values`, `image_sizes`, and
        `placeholder_indices` when images are present.
        """
        images = x.get("images") if isinstance(x, dict) else None
        if images is None:
            # Mistral3 (like the HF model it wraps) supports plain text-only
            # generation: no image inputs are added to the output in that
            # case.
            prompts = x["prompts"] if isinstance(x, dict) else x
            return super().generate_preprocess(
                prompts, sequence_length=sequence_length
            )

        if not self.built:
            self.build(None)

        prompts, flat_images, batched = self._extract_multimodal_inputs(x)
        tokenized, pixel_values, image_sizes = self._build_multimodal_inputs(
            prompts, flat_images
        )
        if pixel_values is None:
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
            )
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
