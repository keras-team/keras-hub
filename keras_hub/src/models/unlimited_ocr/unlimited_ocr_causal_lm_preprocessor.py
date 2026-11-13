import keras
import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_backbone import (
    UnlimitedOCRBackbone,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_image_converter import (
    UnlimitedOCRImageConverter,
)
from keras_hub.src.models.unlimited_ocr.unlimited_ocr_tokenizer import (
    UnlimitedOCRTokenizer,
)
from keras_hub.src.utils.tensor_utils import (
    convert_preprocessing_outputs_python,
)
from keras_hub.src.utils.tensor_utils import convert_to_numpy
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tf


@keras_hub_export("keras_hub.models.UnlimitedOCRCausalLMPreprocessor")
class UnlimitedOCRCausalLMPreprocessor(CausalLMPreprocessor):
    """Unlimited-OCR causal LM preprocessor.

    This layer is for use with `keras_hub.models.UnlimitedOCRCausalLM`. It
    takes prompts and images. It returns the model inputs, and for training
    the labels in a `(x, y, sample_weight)` format. The label `y` is the next
    token id in the `x` sequence.

    `x` should be a dictionary with the key `"prompts"` and, for images, the
    key `"images"`. Each `"<image>"` in the prompts takes the next image.
    `"images"` can be a single image, a batched array, or nested lists of
    images. The number of images must equal the number of `"<image>"` tokens
    in all prompts. The layer replaces each `"<image>"` token with one token
    for every image feature. Without `"images"`, or with a plain string or
    list of strings, the layer preprocesses plain text and adds empty image
    inputs.

    The layer needs an `image_converter`. It sets the empty image inputs of
    text-only prompts, and it gives the grid of tiles of each image.

    The image outputs have the same batch axis as the token outputs. The
    layer outputs these arrays, where `max_images` and `max_tiles` are the
    largest counts in the batch:

    - `"pixel_values"`: `(batch, max_images, global_size, global_size, 3)`.
    - `"pixel_values_local"`: `(batch, max_tiles, tile_size, tile_size, 3)`.
      The tiles of the images of a sample follow each other.
    - `"patches_grid"`: `(batch, max_images, 2)`. The columns and rows of
      tiles of each image.

    A sample with fewer images, or tiles, than the largest count has zero
    padding after its real images, and the grid `(0, 0)` for each padding
    image. A single prompt, not in a batch, has no batch axis.

    For generation, the layer has the methods `generate_preprocess()` and
    `generate_postprocess()`. `keras_hub.models.UnlimitedOCRCausalLM` calls
    them in `generate()`.

    Args:
        tokenizer: A `keras_hub.models.UnlimitedOCRTokenizer` instance.
        image_converter: A `keras_hub.layers.UnlimitedOCRImageConverter`
            instance.
        sequence_length: int. The length of the packed inputs. The sequence
            must hold all image tokens of the prompt. A page with tiles
            needs a few thousand tokens. Defaults to `1024`.
        add_start_token: bool. If `True`, the layer prepends the start token
            to each sequence. Defaults to `True`.
        add_end_token: bool. If `True`, the layer appends the end token to
            each sequence in `call()`. Defaults to `True`.

    Call arguments:
        x: A dictionary with the keys `"prompts"` and `"images"`.
        y: Label data. This must be `None`, because the layer makes labels.
        sample_weight: Label weights. This must be `None`.
        sequence_length: int. Overrides the `sequence_length` of the layer.

    Example:
    ```python
    cls = keras_hub.models.UnlimitedOCRCausalLMPreprocessor
    preprocessor = cls.from_preset(
        "unlimited_ocr", sequence_length=4096
    )
    preprocessor.generate_preprocess(
        {
            "prompts": ["<image>document parsing."],
            "images": [np.random.randint(0, 256, (900, 700, 3))],
        }
    )
    ```
    """

    backbone_cls = UnlimitedOCRBackbone
    tokenizer_cls = UnlimitedOCRTokenizer
    image_converter_cls = UnlimitedOCRImageConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        sequence_length=1024,
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

    def _flatten_images(self, images):
        """Flattens `images` to a list of `(height, width, 3)` arrays."""
        if images is None:
            return []
        if tf is not None and isinstance(images, tf.Tensor):
            images = images.numpy()
        if hasattr(images, "shape"):
            images = convert_to_numpy(images)
            if images.ndim == 3:
                return [images]
            if images.ndim in (4, 5):
                return list(images.reshape(-1, *images.shape[-3:]))
            raise ValueError(
                "`images` must have 3, 4 or 5 dimensions when it is an "
                f"array. Received shape: {images.shape}"
            )
        flat_images = []
        for item in images:
            flat_images.extend(self._flatten_images(item))
        return flat_images

    def _expand_image_tokens(self, token_ids, num_tokens):
        """Replaces each image token with the tokens of its features."""
        image_token_id = self.tokenizer.image_token_id
        expanded = []
        image_index = 0
        for token_id in token_ids:
            if token_id == image_token_id:
                expanded.extend([image_token_id] * num_tokens[image_index])
                image_index += 1
            else:
                expanded.append(token_id)
        return expanded

    def _empty_image_inputs(self):
        if self.image_converter is None:
            raise ValueError(
                "`UnlimitedOCRCausalLMPreprocessor` needs an "
                "`image_converter`, also for text-only prompts. The model "
                "has image inputs that the converter sizes."
            )
        converter = self.image_converter
        return (
            np.zeros(
                (0, converter.global_size, converter.global_size, 3), "float32"
            ),
            np.zeros(
                (0, converter.tile_size, converter.tile_size, 3), "float32"
            ),
            np.zeros((0, 2), "int32"),
        )

    def _group_images_by_prompt(
        self, pixel_values, pixel_values_local, patches_grid, image_counts
    ):
        """Gives the image arrays a batch axis, with zero padding.

        Args:
            pixel_values: array of shape `(num_images, size, size, 3)`.
            pixel_values_local: array of shape `(num_tiles, size, size, 3)`.
            patches_grid: array of shape `(num_images, 2)`.
            image_counts: list of ints. The number of images of each prompt.

        Returns:
            A `(pixel_values, pixel_values_local, patches_grid)` tuple. The
            arrays have shapes `(batch, max_images, size, size, 3)`,
            `(batch, max_tiles, size, size, 3)` and `(batch, max_images, 2)`.
        """
        tiles_per_image = patches_grid[:, 0] * patches_grid[:, 1]
        tile_counts = []
        image_offset = 0
        for count in image_counts:
            tiles = tiles_per_image[image_offset : image_offset + count]
            tile_counts.append(int(tiles.sum()))
            image_offset += count
        batch_size = len(image_counts)
        max_images = max(image_counts, default=0)
        max_tiles = max(tile_counts, default=0)
        grouped_views = np.zeros(
            (batch_size, max_images) + pixel_values.shape[1:], "float32"
        )
        grouped_tiles = np.zeros(
            (batch_size, max_tiles) + pixel_values_local.shape[1:], "float32"
        )
        grouped_grid = np.zeros((batch_size, max_images, 2), "int32")
        image_offset = tile_offset = 0
        for index, (count, num_tiles) in enumerate(
            zip(image_counts, tile_counts)
        ):
            image_end = image_offset + count
            tile_end = tile_offset + num_tiles
            grouped_views[index, :count] = pixel_values[image_offset:image_end]
            grouped_grid[index, :count] = patches_grid[image_offset:image_end]
            grouped_tiles[index, :num_tiles] = pixel_values_local[
                tile_offset:tile_end
            ]
            image_offset, tile_offset = image_end, tile_end
        return grouped_views, grouped_tiles, grouped_grid

    def _group_images_by_prompt_tf(
        self, pixel_values, pixel_values_local, patches_grid, image_counts
    ):
        """Computes `_group_images_by_prompt` with TensorFlow ops.

        Args:
            pixel_values: float tensor of shape `(num_images, size, size, 3)`.
            pixel_values_local: float tensor of shape
                `(num_tiles, size, size, 3)`.
            patches_grid: int tensor of shape `(num_images, 2)`.
            image_counts: int tensor of shape `(batch,)`. The number of
                images of each prompt.

        Returns:
            A `(pixel_values, pixel_values_local, patches_grid)` tuple.
        """
        batch_size = tf.shape(image_counts)[0]
        tiles_per_image = patches_grid[:, 0] * patches_grid[:, 1]
        image_prompt = tf.repeat(tf.range(batch_size), image_counts)
        tile_counts = tf.math.unsorted_segment_sum(
            tiles_per_image, image_prompt, batch_size
        )

        def gather(values, counts):
            # Appends one zero row. Padding slots point to this row.
            max_count = tf.reduce_max(tf.concat([counts, [0]], axis=0))
            start = tf.cumsum(counts, exclusive=True)
            slot = tf.range(max_count)[None, :]
            index = tf.where(
                slot < counts[:, None],
                start[:, None] + slot,
                tf.shape(values)[0],
            )
            zero_shape = tf.concat([[1], tf.shape(values)[1:]], axis=0)
            zero = tf.zeros(zero_shape, values.dtype)
            padded = tf.concat([values, zero], axis=0)
            return tf.gather(padded, index)

        return (
            gather(pixel_values, image_counts),
            gather(pixel_values_local, tile_counts),
            gather(patches_grid, image_counts),
        )

    def _process_python(self, prompts, images, sequence_length, add_labels):
        """Tokenizes, expands and packs the inputs with Python and NumPy.

        Args:
            prompts: list of str, or an array of str or bytes.
            images: list of `(height, width, 3)` arrays.
            sequence_length: int. The packed length.
            add_labels: bool. If `True`, the method also returns the labels
                and the sample weights.

        Returns:
            A list of NumPy arrays. The list is `[token_ids, padding_mask, y,
            sample_weight, pixel_values, pixel_values_local, patches_grid]`
            if `add_labels` is `True`. Else the list is `[token_ids,
            padding_mask, pixel_values, pixel_values_local, patches_grid]`.
        """
        if not self.built:
            self.build(None)
        prompts = [
            prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
            for prompt in convert_to_numpy(prompts).tolist()
        ]
        if images:
            if self.image_converter is None:
                raise ValueError(
                    "`images` need an `image_converter`. Received images, "
                    "but the preprocessor has `image_converter=None`."
                )
            converted = self.image_converter(images)
            pixel_values = convert_to_numpy(converted["pixel_values"])
            pixel_values_local = convert_to_numpy(
                converted["pixel_values_local"]
            )
            patches_grid = convert_to_numpy(converted["patches_grid"])
        else:
            pixel_values, pixel_values_local, patches_grid = (
                self._empty_image_inputs()
            )

        tokenized = [
            convert_to_numpy(self.tokenizer(prompt)).tolist()
            for prompt in prompts
        ]
        placeholders = sum(
            ids.count(self.tokenizer.image_token_id) for ids in tokenized
        )
        if placeholders != len(images):
            raise ValueError(
                "The number of `<image>` tokens in the prompts must equal the "
                f"number of images. Received {placeholders} `<image>` "
                f"token(s) and {len(images)} image(s)."
            )
        num_tokens = [
            self.image_converter.get_num_image_tokens(columns, rows)
            for columns, rows in patches_grid.tolist()
        ]
        # Each prompt takes the next images in order.
        expanded = []
        image_counts = []
        offset = 0
        for ids in tokenized:
            count = ids.count(self.tokenizer.image_token_id)
            counts = num_tokens[offset : offset + count]
            expanded.append(self._expand_image_tokens(ids, counts))
            image_counts.append(count)
            offset += count
        pixel_values, pixel_values_local, patches_grid = (
            self._group_images_by_prompt(
                pixel_values, pixel_values_local, patches_grid, image_counts
            )
        )

        if add_labels:
            # Pad one extra token to account for the truncation below.
            token_ids, padding_mask = self.packer(
                expanded,
                sequence_length=sequence_length + 1,
                add_start_value=self.add_start_token,
                add_end_value=self.add_end_token,
            )
        else:
            token_ids, padding_mask = self.packer(
                expanded,
                sequence_length=sequence_length,
                add_start_value=self.add_start_token,
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
        image_token_id = self.tokenizer.image_token_id
        expected = sum(ids.count(image_token_id) for ids in expanded)
        found = int(np.sum(outputs[0] == self.tokenizer.image_token_id))
        if found != expected:
            raise ValueError(
                "The packed sequence cuts the image tokens. Expected "
                f"{expected} image token(s), but only {found} fit. Increase "
                "`sequence_length` or use a smaller image."
            )
        return outputs + [pixel_values, pixel_values_local, patches_grid]

    def _flatten_images_tf(self, images):
        """Flattens `images` to an `(n, height, width, 3)` float tensor."""
        images = tf.cast(images, "float32")
        if images.shape.rank == 3:
            return images[None]
        if images.shape.rank in (4, 5):
            return tf.reshape(
                images, tf.concat([[-1], tf.shape(images)[-3:]], axis=0)
            )
        raise ValueError(
            "`images` must have 3, 4 or 5 dimensions when it is an "
            f"array. Received shape: {images.shape}"
        )

    def _static_num_images(self, images):
        """Returns the number of images when the static shape gives it.

        The method returns `None` for an unsupported rank.
        `_flatten_images_tf` then raises the error.
        """
        if images is None:
            return 0
        images = tf.convert_to_tensor(images)
        if images.shape.rank not in (3, 4, 5):
            return None
        leading = {3: [], 4: [images.shape[0]], 5: images.shape[:2]}
        leading = leading[images.shape.rank]
        if None in leading:
            return None
        return int(np.prod(leading))

    def _process_tf(self, prompts, images, sequence_length, add_labels):
        """Tokenizes, expands and packs the inputs with TensorFlow ops.

        The tokenizer is the only step that calls Python. It uses the
        `tokenizers` library. The method returns the same outputs as
        `_process_python`. Errors are `InvalidArgumentError` at run time.
        """
        if not self.built:
            self.build(None)
        if not isinstance(prompts, tf.Tensor):
            prompts = tf.constant(prompts, dtype=tf.string)
        if images is not None:
            images = tf.convert_to_tensor(images)
            # An empty list becomes a rank 1 tensor of shape `(0,)`.
            if images.shape.rank == 1 and images.shape[0] == 0:
                images = None
        static_images = self._static_num_images(images)
        if images is None:
            pixel_values, pixel_values_local, patches_grid = (
                tf.constant(array) for array in self._empty_image_inputs()
            )
        else:
            if self.image_converter is None:
                raise ValueError(
                    "`images` need an `image_converter`. Received images, "
                    "but the preprocessor has `image_converter=None`."
                )
            converted = self.image_converter(self._flatten_images_tf(images))
            pixel_values = converted["pixel_values"]
            pixel_values_local = converted["pixel_values_local"]
            patches_grid = converted["patches_grid"]

        tokens = tf.cast(self.tokenizer(prompts), "int32")
        image_token_id = self.tokenizer.image_token_id
        values = tokens.flat_values
        is_image = tf.equal(values, image_token_id)
        num_images = tf.shape(patches_grid)[0]
        check = tf.debugging.assert_equal(
            tf.reduce_sum(tf.cast(is_image, "int32")),
            num_images,
            message=(
                "The number of `<image>` tokens in the prompts must equal "
                "the number of images."
            ),
        )
        # Each image token repeats once per feature of its image. Each
        # prompt takes the next images in order. The extra `0` keeps the
        # index in range when a check fails.
        num_tokens = self.image_converter._get_num_image_tokens_tf(
            patches_grid[:, 0], patches_grid[:, 1]
        )
        num_tokens = tf.concat([num_tokens, [0]], axis=0)
        image_index = tf.cumsum(tf.cast(is_image, "int32")) - 1
        image_index = tf.clip_by_value(image_index, 0, num_images)
        with tf.control_dependencies([check]):
            counts = tf.where(is_image, tf.gather(num_tokens, image_index), 1)
        row_counts = tf.math.unsorted_segment_sum(
            counts, tokens.value_rowids(), tokens.nrows()
        )
        # The grouping fails with a shape error when the counts differ. The
        # check runs first, so the user gets the clear message.
        with tf.control_dependencies([check]):
            image_counts = tf.math.unsorted_segment_sum(
                tf.cast(is_image, "int32"),
                tokens.value_rowids(),
                tokens.nrows(),
            )
            pixel_values, pixel_values_local, patches_grid = (
                self._group_images_by_prompt_tf(
                    pixel_values, pixel_values_local, patches_grid, image_counts
                )
            )
        expanded = tf.RaggedTensor.from_row_lengths(
            tf.repeat(values, counts), tf.cast(row_counts, "int64")
        )

        if add_labels:
            # Pad one extra token to account for the truncation below.
            token_ids, padding_mask = self.packer(
                expanded,
                sequence_length=sequence_length + 1,
                add_start_value=self.add_start_token,
                add_end_value=self.add_end_token,
            )
        else:
            token_ids, padding_mask = self.packer(
                expanded,
                sequence_length=sequence_length,
                add_start_value=self.add_start_token,
                add_end_value=False,
            )
        token_ids = tf.cast(token_ids, "int32")
        padding_mask = tf.cast(padding_mask, "bool")
        if add_labels:
            outputs = [
                token_ids[..., :-1],
                padding_mask[..., :-1],
                token_ids[..., 1:],
                padding_mask[..., 1:],
            ]
        else:
            outputs = [token_ids, padding_mask]
        check = tf.debugging.assert_equal(
            tf.reduce_sum(tf.cast(outputs[0] == image_token_id, "int32")),
            tf.reduce_sum(tf.where(is_image, counts, 0)),
            message=(
                "The packed sequence cuts the image tokens. Increase "
                "`sequence_length` or use a smaller image."
            ),
        )
        with tf.control_dependencies([check]):
            outputs[0] = tf.identity(outputs[0])
        for output in outputs:
            output.set_shape([None, sequence_length])
        # Set the static shapes that the inputs give. One prompt owns all
        # images. Without tiling, there is no tile. This keeps the shapes of
        # an unbatched `tf.data` element static.
        converter = self.image_converter
        batch = prompts.shape[0]
        max_images = static_images if batch == 1 else None
        max_tiles = 0 if static_images == 0 else None
        if not converter.crop_to_patches:
            max_tiles = 0
        pixel_values.set_shape(
            [batch, max_images, converter.global_size, converter.global_size, 3]
        )
        pixel_values_local.set_shape(
            [batch, max_tiles, converter.tile_size, converter.tile_size, 3]
        )
        patches_grid.set_shape([batch, max_images, 2])
        return outputs + [pixel_values, pixel_values_local, patches_grid]

    def _split_inputs(self, x):
        """Normalizes `x` into `(prompts, images, batched)`."""
        images = None
        if isinstance(x, dict):
            images = x.get("images")
            x = x["prompts"]
        batched = True
        if isinstance(x, str):
            batched = False
            x = [x]
        elif tf is not None and isinstance(x, tf.Tensor):
            if x.shape.rank == 0:
                batched = False
                x = tf.expand_dims(x, 0)
        else:
            x = list(x)
        return x, images, batched

    def _squeeze_batch(self, tensors):
        return [
            tf.squeeze(t, axis=0)
            if tf is not None and isinstance(t, tf.Tensor)
            else np.squeeze(t, axis=0)
            for t in tensors
        ]

    def _build_outputs(self, outputs, add_labels, batched):
        if add_labels:
            token_ids, padding_mask, y, sample_weight = outputs[:4]
        else:
            token_ids, padding_mask = outputs[:2]
        pixel_values, pixel_values_local, patches_grid = outputs[-3:]
        if not batched:
            (
                token_ids,
                padding_mask,
                pixel_values,
                pixel_values_local,
                (patches_grid),
            ) = self._squeeze_batch(
                [
                    token_ids,
                    padding_mask,
                    pixel_values,
                    pixel_values_local,
                    patches_grid,
                ]
            )
            if add_labels:
                y, sample_weight = self._squeeze_batch([y, sample_weight])
        x = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "pixel_values": pixel_values,
            "pixel_values_local": pixel_values_local,
            "patches_grid": patches_grid,
        }
        if add_labels:
            return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
        return x

    def _call_python(self, x, y=None, sample_weight=None, sequence_length=None):
        sequence_length = int(sequence_length or self.sequence_length)
        prompts, images, batched = self._split_inputs(x)
        outputs = self._process_python(
            prompts, self._flatten_images(images), sequence_length, True
        )
        outputs = self._build_outputs(outputs, True, batched)
        return convert_preprocessing_outputs_python(outputs)

    @preprocessing_function
    def _call_tf(self, x, y=None, sample_weight=None, sequence_length=None):
        sequence_length = sequence_length or self.sequence_length
        prompts, images, batched = self._split_inputs(x)
        outputs = self._process_tf(prompts, images, sequence_length, True)
        return self._build_outputs(outputs, True, batched)

    def _generate_preprocess_python(self, x, sequence_length=None):
        sequence_length = int(sequence_length or self.sequence_length)
        prompts, images, batched = self._split_inputs(x)
        outputs = self._process_python(
            prompts, self._flatten_images(images), sequence_length, False
        )
        outputs = self._build_outputs(outputs, False, batched)
        return convert_preprocessing_outputs_python(outputs)

    @preprocessing_function
    def _generate_preprocess_tf(self, x, sequence_length=None):
        if not self.built:
            self.build(None)
        sequence_length = sequence_length or self.sequence_length
        prompts, images, batched = self._split_inputs(x)
        outputs = self._process_tf(prompts, images, sequence_length, False)
        return self._build_outputs(outputs, False, batched)
