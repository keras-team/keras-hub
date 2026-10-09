import keras
import numpy as np

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
from keras_hub.src.utils.tensor_utils import (
    convert_preprocessing_outputs_python,
)
from keras_hub.src.utils.tensor_utils import convert_to_numpy
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tf


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
            instance or `None`.
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

    def _extract_multimodal_inputs(self, x):
        """Normalizes `x` into `(prompts, flat_images, batched)`."""
        prompts = x["prompts"]
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
        return prompts, self._flatten_images(x["images"]), batched

    def _tokenize_and_pack_python(
        self, prompts, image_sizes, sequence_length, add_labels
    ):
        """Tokenizes and packs multimodal prompts with Python and NumPy.

        Args:
            prompts: list of str, or an array of str or bytes.
            image_sizes: int array `(num_images, 2)`, from
                `self.image_converter`.
            sequence_length: int. The packed sequence length.
            add_labels: bool. If `True`, the method also returns the
                shifted labels and sample weights for `call()`.

        Returns:
            list of NumPy arrays. The list is `[token_ids, padding_mask,
            y, sample_weight, placeholder_indices]` if `add_labels` is
            `True`, else `[token_ids, padding_mask, placeholder_indices]`.
        """
        prompts = [
            prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
            for prompt in convert_to_numpy(prompts).tolist()
        ]
        image_sizes = [
            tuple(size) for size in convert_to_numpy(image_sizes).tolist()
        ]
        tokenized = self._tokenize_multimodal_prompts(prompts, image_sizes)
        if add_labels:
            # Pad with one extra token to account for the truncation below.
            token_ids, padding_mask = self.packer(
                tokenized,
                sequence_length=sequence_length + 1,
                add_start_value=self.add_start_token,
                add_end_value=self.add_end_token,
            )
        else:
            token_ids, padding_mask = self.packer(
                tokenized,
                sequence_length=sequence_length,
                add_end_value=False,
            )
        token_ids = convert_to_numpy(token_ids).astype("int32")
        padding_mask = convert_to_numpy(padding_mask).astype("bool")
        if add_labels:
            outputs = [
                token_ids[..., :-1],
                padding_mask[..., :-1],
                token_ids[..., 1:],
                padding_mask[..., 1:],
            ]
        else:
            outputs = [token_ids, padding_mask]
        placeholder_indices = compute_image_placeholder_indices(
            outputs[0], self.tokenizer.image_placeholder_token_id
        )
        return outputs + [placeholder_indices]

    def _tokenize_and_pack_tf(
        self, prompts, image_sizes, sequence_length, add_labels
    ):
        """Runs `_tokenize_and_pack_python` inside `tf.py_function`.

        Tokenization needs concrete Python values. Inside `tf.function` or
        `tf.data.Dataset.map`, `prompts` and `image_sizes` are symbolic.
        """
        if not isinstance(prompts, tf.Tensor):
            prompts = tf.constant(prompts, dtype=tf.string)
        num_pairs = 2 if add_labels else 1
        outputs = tf.py_function(
            lambda p, s: self._tokenize_and_pack_python(
                p, s, sequence_length, add_labels
            ),
            [prompts, image_sizes],
            Tout=[tf.int32, tf.bool] * num_pairs + [tf.int32],
        )
        # `tf.py_function` outputs have unknown rank. The last dim of
        # `placeholder_indices` is data dependent.
        for output in outputs[:-1]:
            output.set_shape([None, sequence_length])
        outputs[-1].set_shape([None, None])
        return outputs

    def _squeeze_batch(self, tensors):
        """Removes the leading batch axis from each tensor in `tensors`.

        The Python path gives NumPy arrays. The TensorFlow path gives
        `tf.Tensor`s, which can be symbolic inside a graph.
        """
        return [
            tf.squeeze(t, axis=0)
            if tf is not None and isinstance(t, tf.Tensor)
            else np.squeeze(t, axis=0)
            for t in tensors
        ]

    def _call_multimodal(self, x, sequence_length, tokenize_and_pack):
        """Builds `call()` outputs for inputs with an `"images"` key.

        Args:
            x: dict with `"prompts"` and `"images"` keys.
            sequence_length: int or `None`.
            tokenize_and_pack: `_tokenize_and_pack_python` or
                `_tokenize_and_pack_tf`.

        Returns:
            `(x, y, sample_weight)`.
        """
        sequence_length = sequence_length or self.sequence_length
        prompts, flat_images, batched = self._extract_multimodal_inputs(x)
        pixel_values, image_sizes = self._convert_images(flat_images)
        if pixel_values is None:
            raise ValueError(
                'Mistral3\'s preprocessor was passed an `"images"` key but '
                "found zero images across the batch."
            )
        token_ids, padding_mask, y, sample_weight, placeholder_indices = (
            tokenize_and_pack(
                prompts, image_sizes, sequence_length, add_labels=True
            )
        )
        if not batched:
            token_ids, padding_mask, y, sample_weight = self._squeeze_batch(
                [token_ids, padding_mask, y, sample_weight]
            )
        out_x = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
            "placeholder_indices": placeholder_indices,
        }
        return keras.utils.pack_x_y_sample_weight(out_x, y, sample_weight)

    def _call_python(self, x, y=None, sample_weight=None, sequence_length=None):
        if not isinstance(x, dict) or x.get("images") is None:
            # Mistral3 (like the HF model it wraps) supports plain text-only
            # calls: no image inputs are added to the output in that case.
            prompts = x["prompts"] if isinstance(x, dict) else x
            return super()._call_python(
                prompts,
                y=y,
                sample_weight=sample_weight,
                sequence_length=sequence_length,
            )
        outputs = self._call_multimodal(
            x, sequence_length, self._tokenize_and_pack_python
        )
        return convert_preprocessing_outputs_python(outputs)

    @preprocessing_function
    def _call_tf(self, x, y=None, sample_weight=None, sequence_length=None):
        if not isinstance(x, dict) or x.get("images") is None:
            prompts = x["prompts"] if isinstance(x, dict) else x
            return super()._call_python(
                prompts,
                y=y,
                sample_weight=sample_weight,
                sequence_length=sequence_length,
            )
        return self._call_multimodal(
            x, sequence_length, self._tokenize_and_pack_tf
        )

    def _generate_preprocess_multimodal(
        self, x, sequence_length, tokenize_and_pack
    ):
        """Builds `generate_preprocess()` outputs for inputs with images.

        Args:
            x: dict with `"prompts"` and `"images"` keys.
            sequence_length: int or `None`.
            tokenize_and_pack: `_tokenize_and_pack_python` or
                `_tokenize_and_pack_tf`.

        Returns:
            A dict of model inputs. An image-free batch gives only
            `token_ids` and `padding_mask`.
        """
        if not self.built:
            self.build(None)
        sequence_length = sequence_length or self.sequence_length
        prompts, flat_images, batched = self._extract_multimodal_inputs(x)
        pixel_values, image_sizes = self._convert_images(flat_images)
        if pixel_values is None:
            return super()._generate_preprocess_python(
                x["prompts"], sequence_length=sequence_length
            )
        token_ids, padding_mask, placeholder_indices = tokenize_and_pack(
            prompts, image_sizes, sequence_length, add_labels=False
        )
        if not batched:
            token_ids, padding_mask = self._squeeze_batch(
                [token_ids, padding_mask]
            )
        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
            "placeholder_indices": placeholder_indices,
        }

    def _generate_preprocess_python(self, x, sequence_length=None):
        if not isinstance(x, dict) or x.get("images") is None:
            # Mistral3 supports plain text-only generation: no image inputs
            # are added to the output in that case.
            prompts = x["prompts"] if isinstance(x, dict) else x
            return super()._generate_preprocess_python(
                prompts, sequence_length=sequence_length
            )
        outputs = self._generate_preprocess_multimodal(
            x, sequence_length, self._tokenize_and_pack_python
        )
        return convert_preprocessing_outputs_python(outputs)

    @preprocessing_function
    def _generate_preprocess_tf(self, x, sequence_length=None):
        if not isinstance(x, dict) or x.get("images") is None:
            prompts = x["prompts"] if isinstance(x, dict) else x
            return super()._generate_preprocess_python(
                prompts, sequence_length=sequence_length
            )
        return self._generate_preprocess_multimodal(
            x, sequence_length, self._tokenize_and_pack_tf
        )

    def get_config(self):
        config = super().get_config()
        config.update({"spatial_merge_size": self.spatial_merge_size})
        return config
