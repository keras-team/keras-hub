import re

import keras
import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.block_diffusion_lm_preprocessor import (
    BlockDiffusionLMPreprocessor,
)
from keras_hub.src.models.diffusion_gemma.diffusion_gemma_backbone import (
    DiffusionGemmaBackbone,
)
from keras_hub.src.models.diffusion_gemma.diffusion_gemma_image_converter import (  # noqa: E501
    DiffusionGemmaImageConverter,
)
from keras_hub.src.models.diffusion_gemma.diffusion_gemma_tokenizer import (
    DiffusionGemmaTokenizer,
)
from keras_hub.src.utils.tensor_utils import (
    convert_preprocessing_outputs_python,
)
from keras_hub.src.utils.tensor_utils import convert_to_numpy
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import strip_to_ragged
from keras_hub.src.utils.tensor_utils import strip_to_ragged_python
from keras_hub.src.utils.tensor_utils import tf


@keras_hub_export("keras_hub.models.DiffusionGemmaBlockDiffusionLMPreprocessor")
class DiffusionGemmaBlockDiffusionLMPreprocessor(BlockDiffusionLMPreprocessor):
    """Preprocessing layer for DiffusionGemma diffusion language model tasks.

    Tokenizes and packs prompt strings, optionally expanding image
    placeholders, then appends `canvas_length` placeholder (pad) tokens
    to form the full model input for the block-diffusion generation loop.

    This layer supports two operational modes:
    - **Text-only**: no converters provided; plain string or dict with
      ``"prompts"`` key accepted.
    - **Image**: pass an ``image_converter``; the prompt may contain
      ``<|image|>`` placeholders that are expanded to per-patch soft tokens.

    The preprocessor accepts input as plain strings **or** as dicts with the
    following keys:

    - ``"prompts"`` (required): string or batch of strings.
    - ``"responses"`` (optional): string or batch of strings used only during
      training (the ``call()`` path).
    - ``"images"`` (optional): image tensor(s) matching ``<|image|>``
      occurrences in the prompts.

    During generation (``generate_preprocess()``), only the packed prompt
    tokens are returned.  The canvas is initialised inside ``generate_step``
    via ``_init_canvas`` so that ``_encode_prompt`` always receives the prompt
    alone, matching the HuggingFace encoder/decoder split.

    Args:
        tokenizer: A `keras_hub.models.DiffusionGemmaTokenizer` instance.
        image_converter: A `keras_hub.layers.DiffusionGemmaImageConverter`
            instance. Defaults to `None`.
        sequence_length: int. Maximum prompt sequence length. Defaults to
            `1024`.
        canvas_length: int. Number of canvas tokens appended after the packed
            prompt during generation preprocessing. Defaults to `256`.
        add_start_token: bool. Whether to prepend the BOS token. Defaults to
            `True`.
        add_end_token: bool. Whether to append the EOS token after the prompt.
            Defaults to `True`.
        max_images_per_prompt: int. Maximum number of images per sample.
            Defaults to `2`.
        num_vision_tokens_per_image: int. Vision placeholder tokens per image.
            Defaults to `280`.
        stop_token_ids: Optional tuple of token IDs. Stripped from decoded
            text during `generate_postprocess()`, in addition to the
            tokenizer's special tokens. Defaults to `None`.
    """

    backbone_cls = DiffusionGemmaBackbone
    tokenizer_cls = DiffusionGemmaTokenizer
    image_converter_cls = DiffusionGemmaImageConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        sequence_length=1024,
        canvas_length=256,
        add_start_token=True,
        add_end_token=True,
        max_images_per_prompt=2,
        num_vision_tokens_per_image=280,
        stop_token_ids=None,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            canvas_length=canvas_length,
            add_start_token=add_start_token,
            add_end_token=add_end_token,
            **kwargs,
        )

        self.stop_token_ids = (
            tuple(stop_token_ids) if stop_token_ids is not None else None
        )

        if (
            image_converter is not None
            and sequence_length
            <= max_images_per_prompt * num_vision_tokens_per_image
        ):
            raise ValueError(
                "`sequence_length` should be greater than "
                "`max_images_per_prompt * num_vision_tokens_per_image`."
            )

        self.image_converter = image_converter
        self.max_images_per_prompt = max_images_per_prompt
        self.num_vision_tokens_per_image = num_vision_tokens_per_image

        self.text_only_model = self.image_converter is None

        if self.image_converter is None:
            self.image_placeholder = None
            self.start_of_image_token = None
            self.end_of_image_token = None
        else:
            self.image_placeholder = "<|image|>"
            self.start_of_image_token = "<|image>"
            self.end_of_image_token = "<image|>"

    def build(self, input_shape):
        # Use MultiSegmentPacker so training supports (prompt, response) pairs.
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sep_value=[],
            sequence_length=self.sequence_length,
            padding_side="left",
        )
        self.built = True

    def _get_vision_indices(self, vision_mask, max_tokens=None):
        use_tf = self._use_tf_workflow()
        ops = tf if use_tf else keras.ops
        shape = ops.shape(vision_mask)
        batch_size, sequence_length = shape[0], shape[1]
        if max_tokens is None:
            max_tokens = sequence_length

        arange = (
            tf.range(sequence_length, dtype="int32")
            if use_tf
            else keras.ops.arange(sequence_length, dtype="int32")
        )
        positions = ops.broadcast_to(arange, (batch_size, sequence_length))
        sentinel = ops.cast(sequence_length, "int32")
        padded_positions = ops.where(vision_mask, positions, sentinel)
        batched_vision_indices = ops.sort(padded_positions, axis=1)[
            :, :max_tokens
        ]
        return ops.where(
            batched_vision_indices == sentinel,
            ops.zeros_like(batched_vision_indices),
            batched_vision_indices,
        )

    def _preprocess_images(self, images, batched):
        if self._use_tf_workflow():
            return self._preprocess_images_tf(images, batched)
        return self._preprocess_images_python(images, batched)

    def _preprocess_images_python(self, images, batched):
        if isinstance(images, np.ndarray):
            array = images
            if not batched:
                if array.ndim == 3:
                    rows = [[array]]
                else:
                    rows = [list(array)]
            elif array.ndim == 4:
                rows = [[image] for image in array]
            else:
                rows = [[image for image in row] for row in array]
        else:
            if not isinstance(images, list):
                images = convert_to_numpy(images)
            if not batched:
                rows = [images]
            elif images and isinstance(images[0], np.ndarray):
                rows = [[image] for image in images]
            else:
                rows = images

        max_patches = (
            self.image_converter.resizing.max_soft_tokens
            * self.image_converter.resizing.pooling_kernel_size**2
        )
        patch_dim = self.image_converter.patch_size**2 * 3
        pooling_kernel_size = self.image_converter.resizing.pooling_kernel_size
        pixel_values_rows = []
        pixel_position_ids_rows = []
        real_token_count_rows = []

        for row in rows:
            row_pixel_values = []
            row_pixel_position_ids = []
            row_real_counts = []
            for image in row[: self.max_images_per_prompt]:
                image = ops.convert_to_tensor(image)
                output = self.image_converter(ops.expand_dims(image, axis=0))
                pixel_values = output["pixel_values"][0]
                pixel_position_ids = output["pixel_position_ids"][0]
                is_real_patch = pixel_position_ids[..., 0] != -1
                real_patch_count = int(
                    convert_to_numpy(ops.sum(ops.cast(is_real_patch, "int32")))
                )
                row_pixel_values.append(pixel_values)
                row_pixel_position_ids.append(pixel_position_ids)
                row_real_counts.append(
                    real_patch_count // pooling_kernel_size**2
                )

            while len(row_pixel_values) < self.max_images_per_prompt:
                row_pixel_values.append(
                    ops.zeros([max_patches, patch_dim], dtype="float32")
                )
                row_pixel_position_ids.append(
                    ops.full([max_patches, 2], -1, dtype="int32")
                )
                row_real_counts.append(0)

            pixel_values_rows.append(ops.stack(row_pixel_values))
            pixel_position_ids_rows.append(ops.stack(row_pixel_position_ids))
            real_token_count_rows.append(row_real_counts)

        return {
            "pixel_values": ops.stack(pixel_values_rows),
            "pixel_position_ids": ops.stack(pixel_position_ids_rows),
            "real_token_count": ops.convert_to_tensor(
                real_token_count_rows, dtype="int32"
            ),
        }

    def _preprocess_images_tf(self, images, batched):
        """Resize and patchify each image.

        The method returns real soft-token counts for each image.
        The Python path preserves each image's aspect ratio.
        """
        if isinstance(images, np.ndarray):
            images = tf.convert_to_tensor(images)
        elif isinstance(images, list):
            images = tf.ragged.constant(images)
        elif not isinstance(images, (tf.Tensor, tf.RaggedTensor)):
            images = tf.convert_to_tensor(images)

        if isinstance(images, tf.RaggedTensor):
            if not batched:
                images = tf.expand_dims(images, axis=0)
            if len(images.shape) == 4:
                images = tf.expand_dims(images, axis=1)
        elif isinstance(images, tf.Tensor):
            if not batched:
                images = tf.expand_dims(images, axis=0)
            if len(images.shape) == 3:
                images = tf.expand_dims(images, axis=0)
            if len(images.shape) == 4:
                images = tf.expand_dims(images, axis=1)
        else:
            raise ValueError(
                "`images` should be a list, ragged tensor, or dense tensor."
                f" Received: `type(images)` = {type(images)}"
            )

        max_patches = (
            self.image_converter.resizing.max_soft_tokens
            * self.image_converter.resizing.pooling_kernel_size**2
        )
        patch_dim = self.image_converter.patch_size**2 * 3
        pooling_kernel_size = self.image_converter.resizing.pooling_kernel_size

        if in_tf_function():
            return self._preprocess_images_shared_canvas(
                images, max_patches, patch_dim, pooling_kernel_size
            )

        if isinstance(images, tf.RaggedTensor):
            row_lengths = [int(n) for n in images.row_lengths().numpy()]
        else:
            row_lengths = [int(images.shape[1])] * int(images.shape[0])

        pixel_values_rows = []
        pixel_position_ids_rows = []
        real_token_count_rows = []
        for row_index, num_real_images in enumerate(row_lengths):
            row_pixel_values = []
            row_pixel_position_ids = []
            row_real_counts = []
            for image_index in range(num_real_images):
                image = images[row_index][image_index]
                # A single image's own (H, W) is always fully known, but
                # indexing into a ragged structure can still hand back a
                # RaggedTensor wrapper instead of resolving to dense.
                if isinstance(image, tf.RaggedTensor):
                    image = image.to_tensor()
                else:
                    image = tf.convert_to_tensor(image)
                out = self.image_converter.call(tf.expand_dims(image, axis=0))
                pixel_values = out["pixel_values"][0]
                pixel_position_ids = out["pixel_position_ids"][0]
                if keras.config.backend() == "torch":
                    if not isinstance(pixel_values, tf.Tensor):
                        pixel_values = tf.convert_to_tensor(
                            pixel_values.cpu().numpy()
                        )
                    if not isinstance(pixel_position_ids, tf.Tensor):
                        pixel_position_ids = tf.convert_to_tensor(
                            pixel_position_ids.cpu().numpy()
                        )
                is_real_patch = pixel_position_ids[..., 0] != -1
                real_patch_count = int(
                    tf.reduce_sum(tf.cast(is_real_patch, tf.int32))
                )
                row_pixel_values.append(pixel_values)
                row_pixel_position_ids.append(pixel_position_ids)
                row_real_counts.append(
                    real_patch_count // pooling_kernel_size**2
                )
            # Pad this prompt's image slots up to max_images_per_prompt.
            while len(row_pixel_values) < self.max_images_per_prompt:
                row_pixel_values.append(
                    tf.zeros([max_patches, patch_dim], dtype=tf.float32)
                )
                row_pixel_position_ids.append(
                    tf.fill([max_patches, 2], tf.constant(-1, tf.int32))
                )
                row_real_counts.append(0)
            pixel_values_rows.append(
                tf.stack(row_pixel_values[: self.max_images_per_prompt])
            )
            pixel_position_ids_rows.append(
                tf.stack(row_pixel_position_ids[: self.max_images_per_prompt])
            )
            real_token_count_rows.append(
                row_real_counts[: self.max_images_per_prompt]
            )

        return {
            "pixel_values": tf.stack(pixel_values_rows),
            "pixel_position_ids": tf.stack(pixel_position_ids_rows),
            "real_token_count": tf.convert_to_tensor(
                real_token_count_rows, dtype=tf.int32
            ),
        }

    def _preprocess_images_shared_canvas(
        self, images, max_patches, patch_dim, pooling_kernel_size
    ):
        """Graph-mode fallback: resize every image to one shared canvas
        size before converting, since a real per-image loop cannot trace.
        Every image gets the same soft-token count regardless of its real
        aspect ratio, unlike the eager path this method backs up.
        """
        if isinstance(images, tf.RaggedTensor):
            images = images.to_tensor(
                shape=[None, self.max_images_per_prompt, None, None, 3],
                default_value=0,
            )

        original_images_shape = tf.shape(images)
        flat_images = tf.reshape(
            images,
            [
                -1,
                original_images_shape[-3],
                original_images_shape[-2],
                original_images_shape[-1],
            ],
        )
        images_dict = self.image_converter.call(flat_images)
        pixel_values = images_dict["pixel_values"]
        pixel_position_ids = images_dict["pixel_position_ids"]

        if keras.config.backend() == "torch":
            if not isinstance(pixel_values, tf.Tensor):
                pixel_values = pixel_values.cpu()
            if not isinstance(pixel_position_ids, tf.Tensor):
                pixel_position_ids = pixel_position_ids.cpu()

        pixel_values = tf.reshape(
            pixel_values,
            [
                original_images_shape[0],
                original_images_shape[1],
                max_patches,
                patch_dim,
            ],
        )
        pixel_position_ids = tf.reshape(
            pixel_position_ids,
            [
                original_images_shape[0],
                original_images_shape[1],
                max_patches,
                2,
            ],
        )
        is_real_patch = pixel_position_ids[..., 0] != -1
        real_patch_count = tf.reduce_sum(
            tf.cast(is_real_patch, tf.int32), axis=-1
        )
        real_token_count = real_patch_count // pooling_kernel_size**2

        return {
            "pixel_values": pixel_values,
            "pixel_position_ids": pixel_position_ids,
            "real_token_count": real_token_count,
        }

    def _build_multimodal_output(
        self,
        token_ids,
        padding_mask,
        vision_mask,
        pixel_values,
        pixel_position_ids,
        batched,
    ):
        """Assemble the output dict from processed tensors."""
        use_tf = self._use_tf_workflow()
        ops = tf if use_tf else keras.ops
        shape = ops.shape(token_ids)
        batch_size, seq_len = shape[0], shape[1]
        position_ids = (
            tf.range(seq_len, dtype="int32")
            if use_tf
            else keras.ops.arange(seq_len, dtype="int32")
        )
        position_ids = ops.expand_dims(position_ids, axis=0)
        position_ids = ops.tile(position_ids, [batch_size, 1])

        if self.text_only_model:
            vision_indices = ops.ones([batch_size, 0], dtype="int32")
        else:
            if self.image_converter is not None:
                max_vision_tokens = (
                    self.max_images_per_prompt
                    * self.num_vision_tokens_per_image
                )
            else:
                max_vision_tokens = None
            vision_indices = self._get_vision_indices(
                vision_mask=vision_mask, max_tokens=max_vision_tokens
            )

        if pixel_values is None:
            patch_dim = 3 * self.image_converter.patch_size**2
            pixel_values = ops.zeros(
                (batch_size, 0, 1, patch_dim), dtype="float32"
            )
        if pixel_position_ids is None:
            pixel_position_ids = ops.zeros((batch_size, 0, 1, 2), dtype="int32")

        if batched:
            return {
                "token_ids": token_ids,
                "padding_mask": padding_mask,
                "position_ids": position_ids,
                "pixel_values": pixel_values,
                "pixel_position_ids": pixel_position_ids,
                "vision_indices": vision_indices,
                "vision_mask": vision_mask,
            }
        return {
            "token_ids": ops.reshape(token_ids, ops.shape(token_ids)[1:]),
            "padding_mask": ops.reshape(
                padding_mask, ops.shape(padding_mask)[1:]
            ),
            "position_ids": ops.reshape(
                position_ids, ops.shape(position_ids)[1:]
            ),
            "pixel_values": ops.reshape(
                pixel_values, ops.shape(pixel_values)[1:]
            ),
            "pixel_position_ids": ops.reshape(
                pixel_position_ids, ops.shape(pixel_position_ids)[1:]
            ),
            "vision_indices": ops.reshape(
                vision_indices, ops.shape(vision_indices)[1:]
            ),
            "vision_mask": ops.reshape(vision_mask, ops.shape(vision_mask)[1:]),
        }

    def _expand_one_prompt_per_occurrence(self, prompt, counts):
        """Expand one prompt's `<|image|>` occurrences to their own real
        counts, in order. `counts[i]` is occurrence `i`'s real token count;
        unused slots (fewer real occurrences than `max_images_per_prompt`)
        are never read, since padded segments are empty strings.
        """
        segments = tf.strings.split(prompt, sep=self.image_placeholder)
        num_segments = tf.shape(segments)[0]
        pad_needed = tf.maximum(
            self.max_images_per_prompt + 1 - num_segments, 0
        )
        segments = tf.concat([segments, tf.fill([pad_needed], "")], axis=0)
        segments = segments[: self.max_images_per_prompt + 1]

        result = segments[0]
        for i in range(self.max_images_per_prompt):
            has_occurrence = i + 1 < num_segments
            block = tf.cond(
                has_occurrence,
                lambda i=i: (
                    self.start_of_image_token
                    + tf.strings.reduce_join(
                        tf.repeat(
                            self.image_placeholder,
                            tf.maximum(counts[i], 0),
                        )
                    )
                    + self.end_of_image_token
                ),
                lambda: tf.constant("", dtype=tf.string),
            )
            # segments[i + 1] is already "" for a padded (non-existent)
            # occurrence, so appending it unconditionally is safe.
            result = result + block + segments[i + 1]
        return result

    def _expand_and_tokenize_prompts(
        self, prompts, batched, real_token_counts=None
    ):
        """Expand image placeholders in prompts and return token IDs.

        With `real_token_counts` (shape `(batch, max_images_per_prompt)`,
        one real per-image soft-token count per occurrence, in order),
        each `<|image|>` occurrence expands to its own image's real count.
        Without it (no raw `images`, e.g. precomputed `pixel_values`),
        every occurrence expands to the fixed `num_vision_tokens_per_image`
        cap, since no per-image size is known in that case.
        """
        if not self._use_tf_workflow():
            return self._expand_and_tokenize_prompts_python(
                prompts, real_token_counts=real_token_counts
            )

        if self.image_converter is not None:
            if real_token_counts is not None:
                prompts = tf.map_fn(
                    lambda args: self._expand_one_prompt_per_occurrence(
                        args[0], args[1]
                    ),
                    (prompts, real_token_counts),
                    fn_output_signature=tf.TensorSpec(
                        shape=[], dtype=tf.string
                    ),
                )
            else:
                num_tokens = self.num_vision_tokens_per_image
                img_pattern = re.escape(self.image_placeholder)
                prompts = tf.strings.regex_replace(
                    prompts,
                    img_pattern,
                    f"{self.start_of_image_token}"
                    + self.image_placeholder * num_tokens
                    + f"{self.end_of_image_token}",
                )

        return self.tokenizer(prompts)

    def _expand_and_tokenize_prompts_python(
        self, prompts, real_token_counts=None
    ):
        if isinstance(prompts, str):
            prompts = [prompts]
        elif isinstance(prompts, np.ndarray):
            prompts = prompts.tolist()
        elif not isinstance(prompts, list):
            prompts = list(convert_to_numpy(prompts))

        if self.image_converter is not None:
            if real_token_counts is not None:
                real_token_counts = convert_to_numpy(real_token_counts)
                expanded_prompts = []
                for prompt, counts in zip(prompts, real_token_counts):
                    segments = prompt.split(self.image_placeholder)
                    result = segments[0]
                    for index, segment in enumerate(segments[1:]):
                        count = int(counts[index])
                        block = (
                            self.start_of_image_token
                            + self.image_placeholder * count
                            + self.end_of_image_token
                        )
                        result += block + segment
                    expanded_prompts.append(result)
                prompts = expanded_prompts
            else:
                block = (
                    self.start_of_image_token
                    + self.image_placeholder * self.num_vision_tokens_per_image
                    + self.end_of_image_token
                )
                prompts = [
                    prompt.replace(self.image_placeholder, block)
                    for prompt in prompts
                ]

        return self.tokenizer(prompts)

    def _resolve_vision(
        self,
        token_ids,
        preprocessed_images,
        pixel_values,
        pixel_position_ids,
        batch_size,
        batched,
    ):
        """Return (pixel_values, pixel_position_ids, vision_mask)."""
        ops = tf if self._use_tf_workflow() else keras.ops
        if preprocessed_images is not None:
            pixel_values = preprocessed_images["pixel_values"]
            pixel_position_ids = preprocessed_images["pixel_position_ids"]
            vision_mask = token_ids == self.tokenizer.image_placeholder_id
        elif pixel_values is not None:
            pixel_values = (
                pixel_values if batched else ops.expand_dims(pixel_values, 0)
            )
            pixel_position_ids = (
                pixel_position_ids
                if batched
                else ops.expand_dims(pixel_position_ids, 0)
            )
            vision_mask = token_ids == self.tokenizer.image_placeholder_id
        else:
            if self.image_converter is not None:
                patch_dim = self.image_converter.patch_size**2 * 3
                pixel_values = ops.ones(
                    [batch_size, 0, 0, patch_dim], dtype="float32"
                )
                pixel_position_ids = ops.zeros(
                    [batch_size, 0, 0, 2], dtype="int32"
                )
            else:
                pixel_values = None
                pixel_position_ids = None
            vision_mask = ops.zeros_like(token_ids, dtype="bool")

        return pixel_values, pixel_position_ids, vision_mask

    def _call_python(self, x, y=None, sample_weight=None, sequence_length=None):
        sequence_length = sequence_length or self.sequence_length

        # Accept plain strings for backward compatibility.
        if not isinstance(x, dict):
            prompts = x
            responses = None
            images = None
            pixel_values = pixel_position_ids = None
        else:
            prompts = x["prompts"]
            responses = x.get("responses", None)
            images = x.get("images", None)
            pixel_values = x.get("pixel_values", None)
            pixel_position_ids = x.get("pixel_position_ids", None)

        batched = True
        if isinstance(prompts, str):
            batched = False
            prompts = [prompts]
            if responses is not None:
                responses = [responses]
        if (
            tf is not None
            and isinstance(prompts, tf.Tensor)
            and len(prompts.shape) == 0
        ):
            batched = False
            prompts = tf.expand_dims(prompts, axis=0)
            if responses is not None:
                responses = tf.expand_dims(responses, axis=0)

        # Resolve images before expanding text, so each image's real
        # per-image soft-token count (aspect-ratio dependent) is known
        # before its `<|image|>` occurrence is expanded.
        preprocessed_images = None
        real_token_counts = None
        if images is not None and self.image_converter is not None:
            preprocessed_images = self._preprocess_images(images, batched)
            real_token_counts = preprocessed_images["real_token_count"]

        prompts_tok = self._expand_and_tokenize_prompts(
            prompts, batched, real_token_counts=real_token_counts
        )

        if responses is not None:
            responses_tok = self.tokenizer(responses)
            segments = (prompts_tok, responses_tok)
        else:
            segments = (prompts_tok,)

        token_ids, segment_ids = self.packer(
            segments,
            sequence_length=sequence_length + 1,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        response_mask = segment_ids == 1
        padding_mask = token_ids != self.tokenizer.pad_token_id

        # Text-only shortcut.
        if self.text_only_model:
            label_ids = token_ids[..., 1:]
            sw = (
                response_mask[..., 1:]
                if responses is not None
                # A label counts only if both it and the token it is
                # predicted from are real. Otherwise a left-padded row's
                # first real label gets weight 1 with no real context.
                else ops.logical_and(
                    padding_mask[..., 1:], padding_mask[..., :-1]
                )
            )
            trimmed_token_ids = token_ids[..., :-1]
            trimmed_padding_mask = padding_mask[..., :-1]
            batch_size = ops.shape(trimmed_token_ids)[0]
            seq_len = ops.shape(trimmed_token_ids)[1]
            position_ids = ops.arange(seq_len, dtype="int32")
            position_ids = ops.expand_dims(position_ids, axis=0)
            position_ids = ops.tile(position_ids, [batch_size, 1])
            out_x = {
                "token_ids": trimmed_token_ids,
                "padding_mask": trimmed_padding_mask,
                "position_ids": position_ids,
            }
            if not batched:
                out_x["token_ids"] = ops.reshape(
                    out_x["token_ids"], ops.shape(out_x["token_ids"])[1:]
                )
                out_x["padding_mask"] = ops.reshape(
                    out_x["padding_mask"], ops.shape(out_x["padding_mask"])[1:]
                )
                out_x["position_ids"] = ops.reshape(
                    out_x["position_ids"], ops.shape(out_x["position_ids"])[1:]
                )
                label_ids = ops.reshape(label_ids, ops.shape(label_ids)[1:])
                sw = ops.reshape(sw, ops.shape(sw)[1:])
            return convert_preprocessing_outputs_python(
                keras.utils.pack_x_y_sample_weight(out_x, label_ids, sw)
            )

        batch_size = ops.shape(token_ids)[0]

        pixel_values, pixel_position_ids, vision_mask = self._resolve_vision(
            token_ids[..., :-1],
            preprocessed_images,
            pixel_values,
            pixel_position_ids,
            batch_size,
            batched,
        )

        # Build labels before trimming.
        label_ids = token_ids[..., 1:]
        sw = (
            response_mask[..., 1:]
            if responses is not None
            # A label counts only if both it and the token it is predicted
            # from are real. Otherwise a left-padded row's first real
            # label gets weight 1 with no real context.
            else ops.logical_and(padding_mask[..., 1:], padding_mask[..., :-1])
        )
        if responses is None:
            # A vision placeholder label carries no real next-token
            # signal. Without this, placeholder ids can dominate the loss.
            label_is_vision_placeholder = (
                label_ids == self.tokenizer.image_placeholder_id
            )
            sw = ops.logical_and(
                sw, ops.logical_not(label_is_vision_placeholder)
            )
        token_ids = token_ids[..., :-1]
        padding_mask = padding_mask[..., :-1]

        out_x = self._build_multimodal_output(
            token_ids=token_ids,
            padding_mask=padding_mask,
            vision_mask=vision_mask,
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
            batched=batched,
        )

        if not batched:
            label_ids = ops.reshape(label_ids, ops.shape(label_ids)[1:])
            sw = ops.reshape(sw, ops.shape(sw)[1:])

        return convert_preprocessing_outputs_python(
            keras.utils.pack_x_y_sample_weight(out_x, label_ids, sw)
        )

    @preprocessing_function
    def _call_tf(self, x, y=None, sample_weight=None, sequence_length=None):
        sequence_length = sequence_length or self.sequence_length

        # Accept plain strings for backward compatibility.
        if not isinstance(x, dict):
            prompts = x
            responses = None
            images = None
            pixel_values = pixel_position_ids = None
        else:
            prompts = x["prompts"]
            responses = x.get("responses", None)
            images = x.get("images", None)
            pixel_values = x.get("pixel_values", None)
            pixel_position_ids = x.get("pixel_position_ids", None)

        batched = True
        if isinstance(prompts, str):
            batched = False
            prompts = [prompts]
            if responses is not None:
                responses = [responses]
        if (
            tf is not None
            and isinstance(prompts, tf.Tensor)
            and len(prompts.shape) == 0
        ):
            batched = False
            prompts = tf.expand_dims(prompts, axis=0)
            if responses is not None:
                responses = tf.expand_dims(responses, axis=0)

        # Resolve images before expanding text, so each image's real
        # per-image soft-token count (aspect-ratio dependent) is known
        # before its `<|image|>` occurrence is expanded.
        preprocessed_images = None
        real_token_counts = None
        if images is not None and self.image_converter is not None:
            preprocessed_images = self._preprocess_images(images, batched)
            real_token_counts = preprocessed_images["real_token_count"]

        prompts_tok = self._expand_and_tokenize_prompts(
            prompts, batched, real_token_counts=real_token_counts
        )

        if responses is not None:
            responses_tok = self.tokenizer(responses)
            segments = (prompts_tok, responses_tok)
        else:
            segments = (prompts_tok,)

        token_ids, segment_ids = self.packer(
            segments,
            sequence_length=sequence_length + 1,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        response_mask = segment_ids == 1
        padding_mask = token_ids != self.tokenizer.pad_token_id

        # Text-only shortcut.
        if self.text_only_model:
            label_ids = token_ids[..., 1:]
            sw = (
                response_mask[..., 1:]
                if responses is not None
                # A label counts only if both it and the token it is
                # predicted from are real. Otherwise a left-padded row's
                # first real label gets weight 1 with no real context.
                else tf.logical_and(
                    padding_mask[..., 1:], padding_mask[..., :-1]
                )
            )
            trimmed_token_ids = token_ids[..., :-1]
            trimmed_padding_mask = padding_mask[..., :-1]
            batch_size = tf.shape(trimmed_token_ids)[0]
            seq_len = tf.shape(trimmed_token_ids)[1]
            position_ids = tf.range(seq_len, dtype="int32")
            position_ids = tf.expand_dims(position_ids, axis=0)
            position_ids = tf.tile(position_ids, [batch_size, 1])
            out_x = {
                "token_ids": trimmed_token_ids,
                "padding_mask": trimmed_padding_mask,
                "position_ids": position_ids,
            }
            if not batched:
                out_x["token_ids"] = tf.reshape(
                    out_x["token_ids"], tf.shape(out_x["token_ids"])[1:]
                )
                out_x["padding_mask"] = tf.reshape(
                    out_x["padding_mask"], tf.shape(out_x["padding_mask"])[1:]
                )
                out_x["position_ids"] = tf.reshape(
                    out_x["position_ids"], tf.shape(out_x["position_ids"])[1:]
                )
                label_ids = tf.reshape(label_ids, tf.shape(label_ids)[1:])
                sw = tf.reshape(sw, tf.shape(sw)[1:])
            return convert_preprocessing_outputs_python(
                keras.utils.pack_x_y_sample_weight(out_x, label_ids, sw)
            )

        batch_size = tf.shape(token_ids)[0]

        pixel_values, pixel_position_ids, vision_mask = self._resolve_vision(
            token_ids[..., :-1],
            preprocessed_images,
            pixel_values,
            pixel_position_ids,
            batch_size,
            batched,
        )

        # Build labels before trimming.
        label_ids = token_ids[..., 1:]
        sw = (
            response_mask[..., 1:]
            if responses is not None
            # A label counts only if both it and the token it is predicted
            # from are real. Otherwise a left-padded row's first real
            # label gets weight 1 with no real context.
            else tf.logical_and(padding_mask[..., 1:], padding_mask[..., :-1])
        )
        if responses is None:
            # A vision placeholder label carries no real next-token
            # signal. Without this, placeholder ids can dominate the loss.
            label_is_vision_placeholder = (
                label_ids == self.tokenizer.image_placeholder_id
            )
            sw = tf.logical_and(sw, tf.logical_not(label_is_vision_placeholder))
        token_ids = token_ids[..., :-1]
        padding_mask = padding_mask[..., :-1]

        out_x = self._build_multimodal_output(
            token_ids=token_ids,
            padding_mask=padding_mask,
            vision_mask=vision_mask,
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
            batched=batched,
        )

        if not batched:
            label_ids = tf.reshape(label_ids, tf.shape(label_ids)[1:])
            sw = tf.reshape(sw, tf.shape(sw)[1:])

        return convert_preprocessing_outputs_python(
            keras.utils.pack_x_y_sample_weight(out_x, label_ids, sw)
        )

    def call(self, x, y=None, sample_weight=None, sequence_length=None):
        if self._use_tf_workflow():
            return self._call_tf(
                x,
                y=y,
                sample_weight=sample_weight,
                sequence_length=sequence_length,
            )
        return self._call_python(
            x,
            y=y,
            sample_weight=sample_weight,
            sequence_length=sequence_length,
        )

    def _generate_preprocess_python(self, x, sequence_length=None):
        """Convert prompt inputs to model-ready tensors for generation.

        Expands image placeholders, tokenizes, and packs to
        ``sequence_length``.  The canvas is initialised separately inside
        `generate_step` via `_init_canvas`, so no canvas tokens are appended
        here.

        Args:
            x: A string, batch of strings, or a dict with key ``"prompts"``
                and optionally ``"images"``, ``"pixel_values"``,
                ``"pixel_position_ids"``.
            sequence_length: Optional int. Maximum prompt sequence length.
                Defaults to ``self.sequence_length``. Output is left-padded to
                this length.

        Returns:
            A dict with ``"token_ids"``, ``"padding_mask"``, and multimodal
            fields when converters are configured.

        Raises:
            ValueError: If a prompt is longer than ``sequence_length``.
        """
        if not self.built:
            self.build(None)

        seq_len = sequence_length or self.sequence_length

        if isinstance(x, dict):
            prompts = x["prompts"]
            images = x.get("images", None)
            pixel_values = x.get("pixel_values", None)
            pixel_position_ids = x.get("pixel_position_ids", None)
        else:
            prompts = x
            images = None
            pixel_values = pixel_position_ids = None

        batched = True
        if isinstance(prompts, str):
            batched = False
            prompts = [prompts]
        if (
            tf is not None
            and isinstance(prompts, tf.Tensor)
            and len(prompts.shape) == 0
        ):
            batched = False
            prompts = tf.expand_dims(prompts, axis=0)

        if self.text_only_model and (
            pixel_values is not None or images is not None
        ):
            raise ValueError(
                "The initialized preprocessor/model is text-only, but "
                "`images`/`pixel_values` is not `None`."
            )

        # Resolve images before expanding text, so each image's real
        # per-image soft-token count (aspect-ratio dependent) is known
        # before its `<|image|>` occurrence is expanded.
        preprocessed_images = None
        real_token_counts = None
        if images is not None and self.image_converter is not None:
            preprocessed_images = self._preprocess_images(images, batched)
            real_token_counts = preprocessed_images["real_token_count"]

        prompts_tok = self._expand_and_tokenize_prompts(
            prompts, batched, real_token_counts=real_token_counts
        )

        # The packer keeps the start of the prompt and drops the tail if it
        # doesn't fit. That silently cuts off the generation cue at the end
        # of the prompt, so raise instead of truncating quietly.
        reserved = 1 if self.add_start_token else 0
        max_prompt_length = seq_len - reserved
        if self._use_tf_workflow():
            prompt_lengths = prompts_tok.row_lengths()
            too_long = bool(ops.any(prompt_lengths > max_prompt_length))
            longest = int(ops.max(prompt_lengths))
        else:
            prompt_lengths = [len(row) for row in prompts_tok]
            too_long = any(
                length > max_prompt_length for length in prompt_lengths
            )
            longest = max(prompt_lengths, default=0)
        if too_long:
            raise ValueError(
                "A prompt is too long for `sequence_length`. The longest "
                f"prompt is {longest} tokens, but only {max_prompt_length} "
                "fit (after reserving space for special tokens). Pass a "
                "larger `sequence_length` to `generate()`."
            )

        token_ids, segment_ids = self.packer(
            (prompts_tok,),
            sequence_length=seq_len,
            add_start_value=self.add_start_token,
            add_end_value=False,
        )
        padding_mask = token_ids != self.tokenizer.pad_token_id

        if self.text_only_model:
            batch_size = ops.shape(token_ids)[0]
            seq_len = ops.shape(token_ids)[1]
            position_ids = ops.arange(seq_len, dtype="int32")
            position_ids = ops.expand_dims(position_ids, axis=0)
            position_ids = ops.tile(position_ids, [batch_size, 1])
            if not batched:
                token_ids = ops.reshape(token_ids, ops.shape(token_ids)[1:])
                padding_mask = ops.reshape(
                    padding_mask, ops.shape(padding_mask)[1:]
                )
                position_ids = ops.reshape(
                    position_ids, ops.shape(position_ids)[1:]
                )
            return convert_preprocessing_outputs_python(
                {
                    "token_ids": token_ids,
                    "padding_mask": padding_mask,
                    "position_ids": position_ids,
                }
            )

        batch_size = ops.shape(token_ids)[0]

        pixel_values, pixel_position_ids, vision_mask = self._resolve_vision(
            token_ids,
            preprocessed_images,
            pixel_values,
            pixel_position_ids,
            batch_size,
            batched,
        )

        # Canvas tokens are initialised inside generate_step via _init_canvas.
        return convert_preprocessing_outputs_python(
            self._build_multimodal_output(
                token_ids=token_ids,
                padding_mask=padding_mask,
                vision_mask=vision_mask,
                pixel_values=pixel_values,
                pixel_position_ids=pixel_position_ids,
                batched=batched,
            )
        )

    @preprocessing_function
    def _generate_preprocess_tf(self, x, sequence_length=None):
        """Convert prompt inputs to model-ready tensors for generation.

        Expands image placeholders, tokenizes, and packs to
        ``sequence_length``.  The canvas is initialised separately inside
        `generate_step` via `_init_canvas`, so no canvas tokens are appended
        here.

        Args:
            x: A string, batch of strings, or a dict with key ``"prompts"``
                and optionally ``"images"``, ``"pixel_values"``,
                ``"pixel_position_ids"``.
            sequence_length: Optional int. Maximum prompt sequence length.
                Defaults to ``self.sequence_length``. Output is left-padded to
                this length.

        Returns:
            A dict with ``"token_ids"``, ``"padding_mask"``, and multimodal
            fields when converters are configured.

        Raises:
            ValueError: If a prompt is longer than ``sequence_length``.
        """
        if not self.built:
            self.build(None)

        seq_len = sequence_length or self.sequence_length

        if isinstance(x, dict):
            prompts = x["prompts"]
            images = x.get("images", None)
            pixel_values = x.get("pixel_values", None)
            pixel_position_ids = x.get("pixel_position_ids", None)
        else:
            prompts = x
            images = None
            pixel_values = pixel_position_ids = None

        batched = True
        if isinstance(prompts, str):
            batched = False
            prompts = [prompts]
        if (
            tf is not None
            and isinstance(prompts, tf.Tensor)
            and len(prompts.shape) == 0
        ):
            batched = False
            prompts = tf.expand_dims(prompts, axis=0)

        if self.text_only_model and (
            pixel_values is not None or images is not None
        ):
            raise ValueError(
                "The initialized preprocessor/model is text-only, but "
                "`images`/`pixel_values` is not `None`."
            )

        # Resolve images before expanding text, so each image's real
        # per-image soft-token count (aspect-ratio dependent) is known
        # before its `<|image|>` occurrence is expanded.
        preprocessed_images = None
        real_token_counts = None
        if images is not None and self.image_converter is not None:
            preprocessed_images = self._preprocess_images(images, batched)
            real_token_counts = preprocessed_images["real_token_count"]

        prompts_tok = self._expand_and_tokenize_prompts(
            prompts, batched, real_token_counts=real_token_counts
        )

        # The packer keeps the start of the prompt and drops the tail if it
        # doesn't fit. That silently cuts off the generation cue at the end
        # of the prompt, so raise instead of truncating quietly.
        reserved = 1 if self.add_start_token else 0
        max_prompt_length = seq_len - reserved
        if self._use_tf_workflow():
            prompt_lengths = prompts_tok.row_lengths()
            too_long = bool(tf.reduce_any(prompt_lengths > max_prompt_length))
            longest = int(tf.reduce_max(prompt_lengths))
        else:
            prompt_lengths = [len(row) for row in prompts_tok]
            too_long = any(
                length > max_prompt_length for length in prompt_lengths
            )
            longest = max(prompt_lengths, default=0)
        if too_long:
            raise ValueError(
                "A prompt is too long for `sequence_length`. The longest "
                f"prompt is {longest} tokens, but only {max_prompt_length} "
                "fit (after reserving space for special tokens). Pass a "
                "larger `sequence_length` to `generate()`."
            )

        token_ids, segment_ids = self.packer(
            (prompts_tok,),
            sequence_length=seq_len,
            add_start_value=self.add_start_token,
            add_end_value=False,
        )
        padding_mask = token_ids != self.tokenizer.pad_token_id

        if self.text_only_model:
            batch_size = tf.shape(token_ids)[0]
            seq_len = tf.shape(token_ids)[1]
            position_ids = tf.range(seq_len, dtype="int32")
            position_ids = tf.expand_dims(position_ids, axis=0)
            position_ids = tf.tile(position_ids, [batch_size, 1])
            if not batched:
                token_ids = tf.reshape(token_ids, tf.shape(token_ids)[1:])
                padding_mask = tf.reshape(
                    padding_mask, tf.shape(padding_mask)[1:]
                )
                position_ids = tf.reshape(
                    position_ids, tf.shape(position_ids)[1:]
                )
            return convert_preprocessing_outputs_python(
                {
                    "token_ids": token_ids,
                    "padding_mask": padding_mask,
                    "position_ids": position_ids,
                }
            )

        batch_size = tf.shape(token_ids)[0]

        pixel_values, pixel_position_ids, vision_mask = self._resolve_vision(
            token_ids,
            preprocessed_images,
            pixel_values,
            pixel_position_ids,
            batch_size,
            batched,
        )

        # Canvas tokens are initialised inside generate_step via _init_canvas.
        return convert_preprocessing_outputs_python(
            self._build_multimodal_output(
                token_ids=token_ids,
                padding_mask=padding_mask,
                vision_mask=vision_mask,
                pixel_values=pixel_values,
                pixel_position_ids=pixel_position_ids,
                batched=batched,
            )
        )

    def generate_preprocess(self, x, sequence_length=None):
        if self._use_tf_workflow():
            return self._generate_preprocess_tf(
                x, sequence_length=sequence_length
            )
        return self._generate_preprocess_python(
            x, sequence_length=sequence_length
        )

    def _generate_postprocess_python(self, x):
        if not self.built:
            self.build(None)
        # Keep the start-of-image marker in the detokenized output when
        # images are enabled, unlike the standard special tokens.
        ids_to_strip = list(getattr(self.tokenizer, "special_token_ids", []))
        if self.image_converter is not None:
            soi_id = getattr(self.tokenizer, "start_of_image_token_id", None)
            if soi_id is not None and soi_id in ids_to_strip:
                ids_to_strip.remove(soi_id)
        if self.stop_token_ids is not None:
            ids_to_strip.extend(self.stop_token_ids)
        if isinstance(x, dict):
            token_ids = x["token_ids"]
            mask = ops.cast(x["padding_mask"], "bool")
        else:
            token_ids = x
            mask = ops.ones_like(token_ids, dtype="bool")
        was_1d = ops.ndim(token_ids) == 1
        token_ids = strip_to_ragged_python(token_ids, mask, ids_to_strip)
        if was_1d:
            return self.tokenizer.detokenize([token_ids])[0]
        return self.tokenizer.detokenize(token_ids)

    @preprocessing_function
    def _generate_postprocess_tf(self, x):
        if not self.built:
            self.build(None)
        # Keep the start-of-image marker in the detokenized output when
        # images are enabled, unlike the standard special tokens.
        ids_to_strip = list(getattr(self.tokenizer, "special_token_ids", []))
        if self.image_converter is not None:
            soi_id = getattr(self.tokenizer, "start_of_image_token_id", None)
            if soi_id is not None and soi_id in ids_to_strip:
                ids_to_strip.remove(soi_id)
        if self.stop_token_ids is not None:
            ids_to_strip.extend(self.stop_token_ids)
        if isinstance(x, dict):
            token_ids = x["token_ids"]
            mask = ops.cast(x["padding_mask"], "bool")
        else:
            token_ids = x
            mask = ops.ones_like(token_ids, dtype="bool")
        token_ids = strip_to_ragged(token_ids, mask, ids_to_strip)
        return self.tokenizer.detokenize(token_ids)

    @property
    def max_images_per_prompt(self):
        return self._max_images_per_prompt

    @max_images_per_prompt.setter
    def max_images_per_prompt(self, value):
        self._max_images_per_prompt = value

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_converter": None
                if self.image_converter is None
                else keras.layers.serialize(self.image_converter),
                "num_vision_tokens_per_image": self.num_vision_tokens_per_image,
                "max_images_per_prompt": self.max_images_per_prompt,
                "stop_token_ids": self.stop_token_ids,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config.update(
            {
                "image_converter": None
                if config.get("image_converter") is None
                else keras.layers.deserialize(config["image_converter"]),
            }
        )
        return super().from_config(config)
