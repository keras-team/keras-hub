import keras
import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_image_converter import (
    MuseGlimmerImageConverter,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_tokenizer import (
    MuseGlimmerTokenizer,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_video_converter import (
    MuseGlimmerVideoConverter,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_video_converter import (
    _sample_frame_indices,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_video_converter import (
    _sample_frame_indices_tf,
)
from keras_hub.src.utils.tensor_utils import canonicalize_python_string_inputs
from keras_hub.src.utils.tensor_utils import (
    convert_preprocessing_outputs_python,
)
from keras_hub.src.utils.tensor_utils import convert_to_numpy
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tf


def _rank(x):
    if hasattr(x, "shape"):
        return len(x.shape)
    return np.ndim(x)


def _split_media(media, item_rank, media_name):
    """Split raw media input into a list of single items.

    `media` is one item of rank `item_rank`, a stack of items with one or
    two leading axes (`(num_items, ...)` or `(batch, num_items, ...)`), or
    a nested list of these. Items can have different sizes. The order is
    row-major, so the i-th item goes to the i-th placeholder.
    """
    if isinstance(media, (list, tuple)):
        items = []
        for child in media:
            items.extend(_split_media(child, item_rank, media_name))
        return items
    rank = _rank(media)
    if rank == item_rank:
        return [media]
    if rank in (item_rank + 1, item_rank + 2):
        media = convert_to_numpy(media)
        return list(media.reshape((-1,) + media.shape[-item_rank:]))
    raise ValueError(
        f"`{media_name}` must be one {media_name[:-1]} of rank {item_rank}, "
        f"a stack of rank {item_rank + 1} or {item_rank + 2}, or a list of "
        f"{media_name}. Received rank {rank}."
    )


def _media_per_row(media, item_rank, batched, media_name):
    """Add the missing batch and item axes to same-size media.

    The result has shape `(batch_size, num_items) + item_shape`.
    """
    rank = _rank(media)
    if not batched:
        media = media[None]
    if _rank(media) == item_rank + 1:
        media = media[:, None]
    if _rank(media) != item_rank + 2:
        raise ValueError(
            f"Each prompt takes one {media_name[:-1]} of rank {item_rank}, "
            f"or a stack of {media_name} of rank {item_rank + 1}. Received "
            f"`{media_name}` of rank {rank}."
        )
    return media


def _same_size_media_python(media, item_rank, media_name):
    """Return `media` as one NumPy array, or raise if item sizes differ."""
    leaves = []

    def visit(entry):
        if isinstance(entry, (list, tuple)):
            for child in entry:
                visit(child)
        else:
            leaves.append(convert_to_numpy(entry))

    visit(media)
    shapes = sorted({leaf.shape[-item_rank:] for leaf in leaves})
    if len(shapes) > 1:
        raise ValueError(
            f"All `{media_name}` in one call need the same size when the "
            f"preprocessor builds training inputs. Received item shapes "
            f"{shapes}. Use same-size {media_name}, or use "
            "`generate_preprocess` for mixed sizes."
        )
    return convert_to_numpy(media)


def _fps_per_item_python(fps, batched, batch_size, num_items):
    """Return a flat NumPy array with one frame rate per video.

    `fps` is a scalar, one value per row, or one value per item. The
    order matches the flattened `(batch_size, num_items)` videos. A flat
    list of videos has `batched=False` and `batch_size=1`.
    """
    fps = np.asarray(convert_to_numpy(fps), dtype="float64")
    if fps.ndim > 0:
        fps = _media_per_row(fps, 0, batched, "video_fps")
    try:
        fps = np.broadcast_to(fps, (batch_size, num_items))
    except ValueError:
        raise ValueError(
            f"`video_fps` must be a scalar, one value per prompt, or one "
            f"value per video. Received shape {fps.shape} for "
            f"{batch_size} prompts with {num_items} videos each."
        )
    return fps.reshape(-1)


def _fps_per_item_tf(fps, batched, batch_size, num_items):
    """TensorFlow version of `_fps_per_item_python`."""
    fps = tf.cast(fps, "float64")
    if fps.shape.rank > 0:
        fps = _media_per_row(fps, 0, batched, "video_fps")
    return tf.reshape(tf.broadcast_to(fps, (batch_size, num_items)), (-1,))


@keras_hub_export("keras_hub.models.MuseGlimmerCausalLMPreprocessor")
class MuseGlimmerCausalLMPreprocessor(CausalLMPreprocessor):
    """MuseGlimmer causal LM preprocessor with optional image/video inputs.

    For text-only usage this behaves identically to the base
    `CausalLMPreprocessor`. When an `image_converter`/`video_converter` is
    provided and the input contains `"images"`/`"videos"`, each occurrence
    of the tokenizer's `image_token_id`/`video_token_id` placeholder in the
    prompt expands to the layout of the Hugging Face processor, and
    `vision_indices` are computed for scattering the vision encoder's
    output into the text embedding sequence. MuseGlimmer uses standard
    sequential text positions throughout — no M-RoPE is needed, since
    spatial awareness lives entirely inside the vision tower.

    The layout of an image is the image start token, one placeholder per
    merged vision token, and the image end token. The layout of a video is
    the video start token, then one block per temporal group, and the video
    end token. A block holds the text `Time: {seconds:.1f}s` and one
    placeholder per merged vision token of the group. The video separator
    token sits between two blocks. The seconds value is the index of the
    first frame of the group, divided by the source frame rate. A prompt
    needs one placeholder per item, also for a video.

    A placeholder that already sits between its own start and end tokens
    takes only the inner part of the layout. The delimiter tokens then do
    not appear twice.

    For training, each prompt needs the same number of same-size media
    items. `sample_weight` is zero where the label is a vision placeholder.

    The input dict can have a `"video_fps"` entry: one frame rate for all
    videos, or one per prompt or per video. If the number of prompts equals
    the number of videos, the values apply per video. Without it, the video
    converter uses its `source_fps`. A training batch needs the same
    frame count for all videos.

    Args:
        tokenizer: A `MuseGlimmerTokenizer` instance.
        image_converter: A `MuseGlimmerImageConverter` instance, or `None`.
        video_converter: A `MuseGlimmerVideoConverter` instance, or `None`.
        sequence_length: int. The length of the packed inputs.
        add_start_token: bool. Whether to prepend the tokenizer's start
            token. Defaults to `True`.
        add_end_token: bool. Whether to append the tokenizer's end token.
            Defaults to `True`.

    Example:

    Preprocess prompts for training and for generation.
    ```python
    preprocessor = keras_hub.models.MuseGlimmerCausalLMPreprocessor.from_preset(
        "muse_glimmer_30b", sequence_length=512
    )
    # Call the preprocessor for training. It returns `(x, y, sample_weight)`.
    x, y, sample_weight = preprocessor(["What is a plane?"])
    x, y, sample_weight = preprocessor({
        "prompts": ["Describe <|image|>"],
        "images": np.ones((1, 224, 224, 3), dtype="float32"),
    })
    # Call `generate_preprocess` for generation. It returns the model inputs.
    preprocessor.generate_preprocess(["What is a plane?"])
    preprocessor.generate_preprocess({
        "prompts": ["Describe <|image|>"],
        "images": np.ones((1, 224, 224, 3), dtype="float32"),
    })
    ```
    """

    backbone_cls = MuseGlimmerBackbone
    tokenizer_cls = MuseGlimmerTokenizer
    image_converter_cls = MuseGlimmerImageConverter
    video_converter_cls = MuseGlimmerVideoConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        video_converter=None,
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
        self.video_converter = video_converter

    def _split_inputs(self, x):
        """Split `x` into prompts and a list of media entries.

        Each media entry is `(media_name, media, converter, item_rank,
        token_id, fps)`. `fps` is the `"video_fps"` input for videos and
        `None` for images. Images come before videos. The vision tokens
        use the same order.
        """
        if not isinstance(x, dict):
            return x, []
        media_entries = []
        for media_name, converter, item_rank, token_id, fps in (
            (
                "images",
                self.image_converter,
                3,
                self.tokenizer.image_token_id,
                None,
            ),
            (
                "videos",
                self.video_converter,
                4,
                self.tokenizer.video_token_id,
                x.get("video_fps", None),
            ),
        ):
            media = x.get(media_name, None)
            if media is None:
                continue
            if converter is None:
                raise ValueError(
                    f"`{media_name}` were given, but this preprocessor has "
                    f"no `{media_name[:-1]}_converter`."
                )
            media_entries.append(
                (media_name, media, converter, item_rank, token_id, fps)
            )
        return x["prompts"], media_entries

    def _drop_empty_media(self, media_entries):
        """Drop the media entries with no items, like missing entries."""
        return [
            entry
            for entry in media_entries
            if _split_media(entry[1], entry[3], entry[0])
        ]

    # === Python path ===

    def _compute_vision_indices(self, token_ids):
        token_ids = convert_to_numpy(token_ids).reshape(-1)
        img_indices = np.flatnonzero(token_ids == self.tokenizer.image_token_id)
        vid_indices = np.flatnonzero(token_ids == self.tokenizer.video_token_id)
        return np.concatenate([img_indices, vid_indices]).astype("int32")

    def _image_layout(self, num_tokens):
        """Return `(prefix, body, suffix)` ids for one image."""
        tokenizer = self.tokenizer
        return (
            [tokenizer.image_start_token_id],
            [tokenizer.image_token_id] * num_tokens,
            [tokenizer.image_end_token_id],
        )

    def _time_ids(self, seconds):
        text = f"Time: {seconds:.1f}s"
        return convert_to_numpy(self.tokenizer(text)).tolist()

    def _video_layout(self, timestamps, tokens_per_group):
        """Return `(prefix, body, suffix)` ids for one video.

        `timestamps` has one value in seconds for each temporal group.
        """
        tokenizer = self.tokenizer
        body = []
        for group, seconds in enumerate(timestamps):
            body += self._time_ids(seconds)
            body += [tokenizer.video_token_id] * tokens_per_group
            if group < len(timestamps) - 1:
                body.append(tokenizer.video_separator_token_id)
        return (
            [tokenizer.video_start_token_id],
            body,
            [tokenizer.video_end_token_id],
        )

    def _media_layouts(self, media_name, converter, grids, timestamps):
        """Return the layout of each item in a list of media."""
        if media_name == "images":
            return [
                self._image_layout(
                    int(t)
                    * (int(h) // converter.merge_size)
                    * (int(w) // converter.merge_size)
                )
                for t, h, w in grids
            ]
        return [
            self._video_layout(
                item_timestamps,
                (int(h) // converter.merge_size)
                * (int(w) // converter.merge_size),
            )
            for (_, h, w), item_timestamps in zip(grids, timestamps)
        ]

    def _expand_vision_placeholders(
        self, sequences, image_layouts, video_layouts
    ):
        """Expand placeholders in a batch of token id lists.

        Media items map to placeholders in order across the whole batch.
        The i-th image placeholder takes `image_layouts[i]`. A layout is
        `(prefix, body, suffix)`. A placeholder between its own start and
        end tokens takes only `body`.
        """
        tokenizer = self.tokenizer
        kinds = {
            tokenizer.image_token_id: (
                "image",
                image_layouts,
                tokenizer.image_start_token_id,
                tokenizer.image_end_token_id,
            ),
            tokenizer.video_token_id: (
                "video",
                video_layouts,
                tokenizer.video_start_token_id,
                tokenizer.video_end_token_id,
            ),
        }
        next_index = {token_id: 0 for token_id in kinds}
        expanded_sequences = []
        for ids in sequences:
            expanded = []
            for position, token_id in enumerate(ids):
                if token_id not in kinds:
                    expanded.append(token_id)
                    continue
                media_name, layouts, start_id, end_id = kinds[token_id]
                i = next_index[token_id]
                if i >= len(layouts):
                    raise ValueError(
                        f"The prompts contain more {media_name} "
                        f"placeholders than the {len(layouts)} "
                        f"{media_name}(s) given."
                    )
                prefix, body, suffix = layouts[i]
                is_delimited = (
                    position > 0
                    and position + 1 < len(ids)
                    and ids[position - 1] == start_id
                    and ids[position + 1] == end_id
                )
                if not is_delimited:
                    expanded.extend(prefix)
                expanded.extend(body)
                if not is_delimited:
                    expanded.extend(suffix)
                next_index[token_id] = i + 1
            expanded_sequences.append(expanded)
        for token_id, (media_name, layouts, _, _) in kinds.items():
            if next_index[token_id] != len(layouts):
                raise ValueError(
                    f"{len(layouts)} {media_name}(s) were given, but "
                    f"the prompts contain {next_index[token_id]} "
                    f"{media_name} placeholder(s). Add one placeholder per "
                    f"{media_name}."
                )
        return expanded_sequences

    def _video_timestamps(self, converter, num_frames, source_fps, num_groups):
        """Return the start time in seconds of each temporal group.

        The time is the index of the first frame of the group, divided by
        the source frame rate. The computation uses `float32`, like HF.
        """
        indices = _sample_frame_indices(
            num_frames,
            converter.num_frames,
            converter.patch_temporal,
            converter.fps,
            source_fps,
        )
        if indices is None:
            indices = np.arange(num_frames)
        positions = np.minimum(
            np.arange(num_groups) * converter.patch_temporal, len(indices) - 1
        )
        seconds = indices[positions].astype("float32") / np.float32(source_fps)
        return seconds.tolist()

    def _convert_media(
        self,
        media,
        converter,
        item_rank,
        media_name,
        fps=None,
        items_per_prompt=None,
    ):
        """Run `converter` on each media item separately.

        `fps` is `None`, one value, one frame rate per item, or one per
        prompt. A per-prompt value applies to all items of that prompt,
        as given by `items_per_prompt`. The converter gets the frame rate
        as `source_fps`.

        Returns the per-item patches, the per-item `grid_thw`, the per-item
        count of merged vision tokens, and for videos the start time in
        seconds of each temporal group of each item.
        """
        patches, grids, timestamps = [], [], []
        items = _split_media(media, item_rank, media_name)
        item_fps = None
        if fps is not None:
            fps = np.asarray(convert_to_numpy(fps), dtype="float64")
            if (
                fps.ndim == 1
                and items_per_prompt is not None
                and fps.size != len(items)
                and fps.size == len(items_per_prompt)
            ):
                fps = np.repeat(fps, items_per_prompt)
            item_fps = _fps_per_item_python(fps, False, 1, len(items))
        for i, item in enumerate(items):
            item = keras.ops.convert_to_tensor(convert_to_numpy(item))
            if item_fps is None:
                result = converter(item)
            else:
                result = converter(item, source_fps=float(item_fps[i]))
            patches.append(convert_to_numpy(result["patches"]))
            grids.append(convert_to_numpy(result["grid_thw"]).astype("int32"))
            if media_name == "videos":
                source_fps = float(
                    converter.source_fps if item_fps is None else item_fps[i]
                )
                timestamps.append(
                    self._video_timestamps(
                        converter,
                        int(item.shape[0]),
                        source_fps,
                        int(grids[-1][0]),
                    )
                )
        merge_size = converter.merge_size
        num_tokens = [
            int(t) * (int(h) // merge_size) * (int(w) // merge_size)
            for t, h, w in grids
        ]
        return patches, grids, num_tokens, timestamps

    def _pack_with_media_python(self, x, sequence_length, add_end_value):
        """Tokenize and pack prompts, and convert same-size media per row.

        Returns `token_ids` and `padding_mask` of shape `(batch_size,
        sequence_length)`, a list of `(media_name, token_id, patches,
        grid_thw, tokens_per_row)` per media kind, and `batched`. `patches`
        has shape `(batch_size, num_patches, patch_dim)`. `grid_thw` has
        shape `(batch_size, num_items, 3)`.
        """
        prompts, media_entries = self._split_inputs(x)
        media_entries = self._drop_empty_media(media_entries)
        prompts, batched, _ = canonicalize_python_string_inputs(prompts)
        batch_size = len(prompts)

        media_outputs = []
        layouts = {"images": [], "videos": []}
        for (
            media_name,
            media,
            converter,
            item_rank,
            token_id,
            fps,
        ) in media_entries:
            media = _media_per_row(
                _same_size_media_python(media, item_rank, media_name),
                item_rank,
                batched,
                media_name,
            )
            if media.shape[0] != batch_size:
                raise ValueError(
                    f"`{media_name}` must have one row per prompt. Received "
                    f"{batch_size} prompts and {media.shape[0]} rows."
                )
            num_items = media.shape[1]
            if fps is not None:
                fps = _fps_per_item_python(fps, batched, batch_size, num_items)
            patches, grids, num_tokens, timestamps = self._convert_media(
                media.reshape((-1,) + media.shape[2:]),
                converter,
                item_rank,
                media_name,
                fps,
            )
            patch_shapes = sorted({patch.shape for patch in patches})
            if len(patch_shapes) > 1:
                raise ValueError(
                    f"All `{media_name}` in one call need the same output "
                    f"size when the preprocessor builds training inputs. "
                    f"The converter gave patch shapes {patch_shapes}. "
                    "Different `video_fps` values can give different frame "
                    "counts. Use `video_fps` values that give the same "
                    "frame count, or use `generate_preprocess`."
                )
            patch_dim = patches[0].shape[-1]
            media_outputs.append(
                (
                    media_name,
                    token_id,
                    np.stack(patches).reshape((batch_size, -1, patch_dim)),
                    np.stack(grids).reshape((batch_size, num_items, 3)),
                    num_items * num_tokens[0],
                )
            )
            layouts[media_name] = self._media_layouts(
                media_name, converter, grids, timestamps
            )

        tokenized = [
            convert_to_numpy(ids).tolist() for ids in self.tokenizer(prompts)
        ]
        expanded_sequences = self._expand_vision_placeholders(
            tokenized, layouts["images"], layouts["videos"]
        )
        token_ids, padding_mask = self.packer(
            expanded_sequences,
            sequence_length=sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=add_end_value,
        )
        return (
            convert_to_numpy(token_ids),
            convert_to_numpy(padding_mask),
            media_outputs,
            batched,
        )

    def _vision_indices_python(self, token_ids, media_outputs, sequence_length):
        """Find the per-row positions of the tokens of each media kind."""
        indices = []
        for media_name, token_id, _, _, tokens_per_row in media_outputs:
            rows = [np.flatnonzero(row == token_id) for row in token_ids]
            if any(len(row) != tokens_per_row for row in rows):
                raise ValueError(
                    f"Each prompt must have one placeholder per item in "
                    f"`{media_name}`, and all expanded tokens must fit in "
                    f"`sequence_length={sequence_length}`. Fix the "
                    "placeholders or increase `sequence_length`."
                )
            indices.append(np.stack(rows).astype("int32"))
        return indices

    def _patch_dim(self):
        """Return the flat patch length of the media converter, or `None`."""
        converter = (
            self.image_converter
            if self.image_converter is not None
            else self.video_converter
        )
        if converter is None:
            return None
        return converter.patch_temporal * 3 * converter.patch_size**2

    def _add_empty_media_python(self, outputs):
        """Add empty vision inputs to text-only training outputs.

        A vision-enabled backbone needs these inputs, also without media.
        """
        patch_dim = self._patch_dim()
        if patch_dim is None:
            return outputs
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(outputs)
        batch_shape = tuple(x["token_ids"].shape[:-1])
        x = {
            **x,
            "pixel_values": np.zeros(batch_shape + (0, patch_dim), "float32"),
            "image_grid_thw": np.zeros(batch_shape + (0, 3), "int32"),
            "vision_indices": np.zeros(batch_shape + (0,), "int32"),
        }
        return convert_preprocessing_outputs_python(
            keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
        )

    def _add_empty_media_tf(self, outputs):
        """TensorFlow version of `_add_empty_media_python`."""
        patch_dim = self._patch_dim()
        if patch_dim is None:
            return outputs
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(outputs)
        batch_shape = tf.shape(x["token_ids"])[:-1]

        def empty(trailing_shape, dtype):
            shape = tf.concat([batch_shape, trailing_shape], axis=0)
            return tf.zeros(shape, dtype)

        x = {
            **x,
            "pixel_values": empty([0, patch_dim], "float32"),
            "image_grid_thw": empty([0, 3], "int32"),
            "vision_indices": empty([0], "int32"),
        }
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def _call_python(self, x, y=None, sample_weight=None, sequence_length=None):
        prompts, media_entries = self._split_inputs(x)
        media_entries = self._drop_empty_media(media_entries)
        if not media_entries:
            return self._add_empty_media_python(
                super()._call_python(
                    prompts,
                    y=y,
                    sample_weight=sample_weight,
                    sequence_length=sequence_length,
                )
            )
        sequence_length = sequence_length or self.sequence_length
        # Pad with one extra token to account for the truncation below.
        token_ids, padding_mask, media_outputs, batched = (
            self._pack_with_media_python(
                x, sequence_length + 1, self.add_end_token
            )
        )
        vision_indices = self._vision_indices_python(
            token_ids[:, :-1], media_outputs, sequence_length
        )
        y = token_ids[:, 1:]
        is_vision_label = np.isin(y, [out[1] for out in media_outputs])
        x = {
            "token_ids": token_ids[:, :-1],
            "padding_mask": padding_mask[:, :-1],
            "pixel_values": np.concatenate(
                [out[2] for out in media_outputs], axis=1
            ),
            "image_grid_thw": np.concatenate(
                [out[3] for out in media_outputs], axis=1
            ),
            "vision_indices": np.concatenate(vision_indices, axis=1),
        }
        sample_weight = padding_mask[:, 1:] & ~is_vision_label
        outputs = (x, y, sample_weight)
        if not batched:
            outputs = keras.tree.map_structure(lambda t: t[0], outputs)
        return convert_preprocessing_outputs_python(
            keras.utils.pack_x_y_sample_weight(*outputs)
        )

    def _generate_preprocess_python(self, x, sequence_length=None):
        """Convert prompts and media to generation inputs with NumPy.

        Unlike the training call, items can have different sizes, and
        prompts can have different item counts. The outputs are flat:
        `pixel_values` has shape `(num_patches, patch_dim)`,
        `image_grid_thw` has shape `(num_items, 3)`, and `vision_indices`
        index into the flattened `token_ids`.
        """
        prompts, media_entries = self._split_inputs(x)
        # An entry with no items counts as a missing entry.
        split_entries = []
        for (
            media_name,
            media,
            converter,
            item_rank,
            token_id,
            fps,
        ) in media_entries:
            items = _split_media(media, item_rank, media_name)
            if items:
                split_entries.append(
                    (media_name, items, converter, item_rank, token_id, fps)
                )
        media_entries = split_entries
        if not media_entries:
            return super()._generate_preprocess_python(
                prompts, sequence_length=sequence_length
            )
        if not self.built:
            self.build(None)

        sequence_length = sequence_length or self.sequence_length
        prompts, batched, _ = canonicalize_python_string_inputs(prompts)

        tokenized = [
            convert_to_numpy(self.tokenizer(prompt)).tolist()
            for prompt in prompts
        ]
        pixel_values, grid_thw = [], []
        num_tokens_per_item = {"images": [], "videos": []}
        layouts = {"images": [], "videos": []}
        for (
            media_name,
            media,
            converter,
            item_rank,
            token_id,
            fps,
        ) in media_entries:
            items_per_prompt = [ids.count(token_id) for ids in tokenized]
            patches, grids, num_tokens, timestamps = self._convert_media(
                media, converter, item_rank, media_name, fps, items_per_prompt
            )
            pixel_values.extend(patches)
            grid_thw.extend(grids)
            num_tokens_per_item[media_name] = num_tokens
            layouts[media_name] = self._media_layouts(
                media_name, converter, grids, timestamps
            )

        expanded_sequences = self._expand_vision_placeholders(
            tokenized, layouts["images"], layouts["videos"]
        )
        token_ids, padding_mask = self.packer(
            expanded_sequences if batched else expanded_sequences[0],
            sequence_length=sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=False,
        )

        vision_indices = self._compute_vision_indices(token_ids)
        num_vision_tokens = sum(num_tokens_per_item["images"]) + sum(
            num_tokens_per_item["videos"]
        )
        if vision_indices.shape[0] != num_vision_tokens:
            raise ValueError(
                f"`sequence_length={sequence_length}` truncates the "
                f"expanded image/video tokens. The media need "
                f"{num_vision_tokens} tokens, but only "
                f"{vision_indices.shape[0]} fit. Increase "
                "`sequence_length`."
            )

        return convert_preprocessing_outputs_python(
            {
                "token_ids": token_ids,
                "padding_mask": padding_mask,
                "pixel_values": np.concatenate(pixel_values, axis=0),
                "image_grid_thw": np.stack(grid_thw, axis=0),
                "vision_indices": vision_indices,
            }
        )

    # === TensorFlow path ===

    def _convert_media_tf(
        self, media, converter, item_rank, batched, name, fps=None
    ):
        """Run `converter` on each same-size media item with `tf.map_fn`.

        `fps` is `None`, a scalar, or one frame rate per row or item. The
        converter gets the frame rate as `source_fps`. The items need the
        same output size. Different frame rates can break this.

        Returns per-row patches of shape `(batch_size, num_patches,
        patch_dim)`, `grid_thw` of shape `(batch_size, num_items, 3)`, the
        item count per row and the merged token count per item.
        """
        media = _media_per_row(media, item_rank, batched, name)
        batch_size = tf.shape(media)[0]
        num_items = media.shape[1]
        if num_items is None:
            num_items = tf.shape(media)[1]
        items = tf.reshape(
            media, tf.concat([[-1], tf.shape(media)[2:]], axis=0)
        )
        # Trace the converter once to get its output signature. With a
        # static item size, the signature and the outputs are static.
        item_spec = tf.TensorSpec(items.shape[1:], items.dtype)
        if fps is None:
            convert, elems, signature = converter, items, item_spec
        else:
            fps = _fps_per_item_tf(fps, batched, batch_size, num_items)
            elems = (items, fps)
            signature = (item_spec, tf.TensorSpec([], "float64"))

            def convert(args):
                return converter(args[0], source_fps=args[1])

        item_outputs = (
            tf.function(convert)
            .get_concrete_function(signature)
            .structured_outputs
        )
        outputs = tf.map_fn(
            convert,
            elems,
            fn_output_signature=tf.nest.map_structure(
                lambda t: tf.TensorSpec(t.shape, t.dtype), item_outputs
            ),
        )
        patches = outputs["patches"]
        # The static sizes keep the output shapes static.
        patches_per_item, patch_dim = patches.shape[1], patches.shape[2]
        if patches_per_item is None:
            patches_per_item = tf.shape(patches)[1]
        if patch_dim is None:
            patch_dim = tf.shape(patches)[2]
        patches = tf.reshape(
            patches, (batch_size, num_items * patches_per_item, patch_dim)
        )
        grid_thw = tf.reshape(outputs["grid_thw"], (batch_size, num_items, 3))
        # Each merged token takes `merge_size**2` patches.
        num_tokens = patches_per_item // (converter.merge_size**2)
        timestamps = None
        if item_rank == 4:
            timestamps = self._video_timestamps_tf(
                media, converter, fps, grid_thw, tf.shape(items)[0]
            )
        return patches, grid_thw, num_items, num_tokens, timestamps

    def _video_timestamps_tf(self, media, converter, fps, grid_thw, num_rows):
        """TensorFlow version of `_video_timestamps`.

        `fps` is `None` or one frame rate per item. Returns the start times
        in seconds as a tensor of shape `(num_rows, num_groups)`.
        """
        total_frames = tf.shape(media)[2]
        num_groups = tf.cast(tf.reduce_max(grid_thw[..., 0]), "int32")
        num_groups = tf.maximum(num_groups, 0)
        if fps is None:
            fps = tf.fill([num_rows], tf.cast(converter.source_fps, "float64"))

        def item_timestamps(source_fps):
            indices = _sample_frame_indices_tf(
                total_frames,
                converter.num_frames,
                converter.patch_temporal,
                converter.fps,
                source_fps,
            )
            positions = tf.minimum(
                tf.range(num_groups) * converter.patch_temporal,
                tf.shape(indices)[0] - 1,
            )
            seconds = tf.cast(tf.gather(indices, positions), "float32")
            return seconds / tf.cast(source_fps, "float32")

        return tf.map_fn(
            item_timestamps,
            fps,
            fn_output_signature=tf.TensorSpec([None], "float32"),
        )

    def _layouts_tf(self, media_name, num_rows, num_tokens, timestamps, dtype):
        """Build the ids of each item layout as ragged tensors.

        Returns `(full, inner)`. Both have shape `(num_rows, None)`. `full`
        has the start and end tokens. `inner` has none of them.
        """
        tokenizer = self.tokenizer
        num_rows = tf.cast(num_rows, "int32")

        def column(token_id):
            shape = tf.stack([num_rows, 1])
            return tf.RaggedTensor.from_tensor(
                tf.fill(shape, tf.cast(token_id, dtype))
            )

        def run(token_id, rows, length):
            shape = tf.stack([rows, tf.cast(length, "int32")])
            return tf.RaggedTensor.from_tensor(
                tf.fill(shape, tf.cast(token_id, dtype))
            )

        if media_name == "images":
            body = run(tokenizer.image_token_id, num_rows, num_tokens)
            start = column(tokenizer.image_start_token_id)
            end = column(tokenizer.image_end_token_id)
        else:
            num_groups = tf.shape(timestamps)[1]
            group_rows = num_rows * num_groups
            tokens_per_group = tf.cast(num_tokens, "int32") // tf.maximum(
                num_groups, 1
            )
            seconds = tf.strings.as_string(
                tf.reshape(timestamps, [-1]), precision=1
            )
            time_ids = tf.cast(
                tokenizer(tf.strings.join(["Time: ", seconds, "s"])), dtype
            )
            group_index = tf.tile(tf.range(num_groups), [num_rows])
            has_separator = group_index < num_groups - 1
            separators = tf.RaggedTensor.from_row_lengths(
                tf.fill(
                    [tf.reduce_sum(tf.cast(has_separator, "int32"))],
                    tf.cast(tokenizer.video_separator_token_id, dtype),
                ),
                tf.cast(has_separator, "int64"),
            )
            groups = tf.concat(
                [
                    time_ids,
                    run(tokenizer.video_token_id, group_rows, tokens_per_group),
                    separators,
                ],
                axis=1,
            )
            body = tf.RaggedTensor.from_uniform_row_length(
                groups,
                tf.cast(num_groups, "int64"),
                nrows=tf.cast(num_rows, "int64"),
            ).merge_dims(1, 2)
            start = column(tokenizer.video_start_token_id)
            end = column(tokenizer.video_end_token_id)
        return tf.concat([start, body, end], axis=1), body

    def _expand_placeholders_tf(
        self, token_ids, token_id, start_id, end_id, full, inner
    ):
        """TensorFlow version of `_expand_vision_placeholders`.

        The i-th placeholder in the whole batch takes row i of `full`. A
        placeholder between `start_id` and `end_id` takes row i of `inner`.
        """
        flat_ids = token_ids.flat_values
        is_placeholder = tf.equal(flat_ids, token_id)
        num_items = full.nrows()
        item_index = (
            tf.cast(tf.cumsum(tf.cast(is_placeholder, "int32")), "int64") - 1
        )
        item_index = tf.clip_by_value(
            item_index, 0, tf.maximum(num_items - 1, 0)
        )
        # A column of -1 marks the row borders in `previous` and `following`.
        border = tf.RaggedTensor.from_tensor(
            tf.fill(
                tf.stack([tf.cast(token_ids.nrows(), "int32"), 1]),
                tf.cast(-1, flat_ids.dtype),
            )
        )
        previous = tf.concat([border, token_ids], axis=1)[:, :-1]
        following = tf.concat([token_ids, border], axis=1)[:, 1:]
        is_delimited = (
            is_placeholder
            & tf.equal(previous.flat_values, start_id)
            & tf.equal(following.flat_values, end_id)
        )
        empty = tf.RaggedTensor.from_row_lengths(
            tf.zeros([0], flat_ids.dtype), tf.zeros([1], "int64")
        )
        replacements = tf.concat([full, inner, empty], axis=0)
        index = tf.where(is_delimited, item_index + num_items, item_index)
        index = tf.where(is_placeholder, index, 2 * num_items)
        replaced = tf.gather(replacements, index)
        kept = tf.RaggedTensor.from_row_lengths(
            tf.boolean_mask(flat_ids, tf.logical_not(is_placeholder)),
            tf.cast(tf.logical_not(is_placeholder), "int64"),
        )
        pieces = tf.concat([kept, replaced], axis=1)
        return token_ids.with_flat_values(pieces).merge_dims(1, 2)

    def _pack_with_media_tf(self, x, sequence_length, add_end_value):
        """TensorFlow version of `_pack_with_media_python`."""
        prompts, media_entries = self._split_inputs(x)
        batched = prompts.shape.rank == 1
        if not batched:
            prompts = prompts[tf.newaxis]
        token_ids = self.tokenizer(prompts)

        media_outputs = []
        for (
            media_name,
            media,
            converter,
            item_rank,
            token_id,
            fps,
        ) in media_entries:
            patches, grids, num_items, num_tokens, timestamps = (
                self._convert_media_tf(
                    media, converter, item_rank, batched, media_name, fps
                )
            )
            media_outputs.append(
                (media_name, token_id, patches, grids, num_items * num_tokens)
            )
            full, inner = self._layouts_tf(
                media_name,
                tf.shape(tf.reshape(grids, (-1, 3)))[0],
                num_tokens,
                timestamps,
                token_ids.dtype,
            )
            if media_name == "images":
                start_id = self.tokenizer.image_start_token_id
                end_id = self.tokenizer.image_end_token_id
            else:
                start_id = self.tokenizer.video_start_token_id
                end_id = self.tokenizer.video_end_token_id
            token_ids = self._expand_placeholders_tf(
                token_ids, token_id, start_id, end_id, full, inner
            )

        token_ids, padding_mask = self.packer(
            token_ids,
            sequence_length=sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=add_end_value,
        )
        return token_ids, padding_mask, media_outputs, batched

    def _vision_indices_tf(self, token_ids, media_outputs, sequence_length):
        """TensorFlow version of `_vision_indices_python`."""
        batch_size = tf.shape(token_ids)[0]
        indices = []
        for media_name, token_id, _, _, tokens_per_row in media_outputs:
            is_vision = token_ids == token_id
            check = tf.debugging.assert_equal(
                tf.reduce_sum(tf.cast(is_vision, "int32"), axis=1),
                tokens_per_row,
                message=(
                    f"Each prompt must have one placeholder per item in "
                    f"`{media_name}`, and all expanded tokens must fit in "
                    f"`sequence_length={sequence_length}`."
                ),
            )
            with tf.control_dependencies([check]):
                positions = tf.cast(tf.where(is_vision)[:, 1], "int32")
            indices.append(tf.reshape(positions, (batch_size, tokens_per_row)))
        return indices

    @preprocessing_function
    def _call_tf(self, x, y=None, sample_weight=None, sequence_length=None):
        prompts, media_entries = self._split_inputs(x)
        if not media_entries:
            return self._add_empty_media_tf(
                super()._call_python(
                    prompts,
                    y=y,
                    sample_weight=sample_weight,
                    sequence_length=sequence_length,
                )
            )
        sequence_length = sequence_length or self.sequence_length
        # Pad with one extra token to account for the truncation below.
        token_ids, padding_mask, media_outputs, batched = (
            self._pack_with_media_tf(x, sequence_length + 1, self.add_end_token)
        )
        vision_indices = self._vision_indices_tf(
            token_ids[:, :-1], media_outputs, sequence_length
        )
        y = token_ids[:, 1:]
        is_vision_label = tf.zeros_like(y, dtype="bool")
        for _, token_id, _, _, _ in media_outputs:
            is_vision_label |= y == token_id
        x = {
            "token_ids": token_ids[:, :-1],
            "padding_mask": padding_mask[:, :-1],
            "pixel_values": tf.concat(
                [out[2] for out in media_outputs], axis=1
            ),
            "image_grid_thw": tf.concat(
                [out[3] for out in media_outputs], axis=1
            ),
            "vision_indices": tf.concat(vision_indices, axis=1),
        }
        sample_weight = padding_mask[:, 1:] & ~is_vision_label
        outputs = (x, y, sample_weight)
        if not batched:
            outputs = keras.tree.map_structure(lambda t: t[0], outputs)
        return keras.utils.pack_x_y_sample_weight(*outputs)

    @preprocessing_function
    def _generate_preprocess_tf(self, x, sequence_length=None):
        """Convert prompts and media to generation inputs with TensorFlow.

        The outputs use the flat layout of `_generate_preprocess_python`.
        Unlike the Python path, each prompt needs the same number of
        same-size items.
        """
        prompts, media_entries = self._split_inputs(x)
        if not media_entries:
            return super()._generate_preprocess_python(
                prompts, sequence_length=sequence_length
            )
        if not self.built:
            self.build(None)

        sequence_length = sequence_length or self.sequence_length
        token_ids, padding_mask, media_outputs, batched = (
            self._pack_with_media_tf(x, sequence_length, False)
        )
        vision_indices = self._vision_indices_tf(
            token_ids, media_outputs, sequence_length
        )
        # Flatten each media kind over the batch. Images come before
        # videos, so `pixel_values` and `vision_indices` stay aligned.
        row_offsets = tf.range(tf.shape(token_ids)[0]) * sequence_length
        pixel_values, grid_thw, flat_indices = [], [], []
        for (_, _, patches, grids, _), indices in zip(
            media_outputs, vision_indices
        ):
            pixel_values.append(
                tf.reshape(patches, (-1, tf.shape(patches)[-1]))
            )
            grid_thw.append(tf.reshape(grids, (-1, 3)))
            flat_indices.append(
                tf.reshape(indices + row_offsets[:, tf.newaxis], (-1,))
            )
        if not batched:
            token_ids, padding_mask = token_ids[0], padding_mask[0]
        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "pixel_values": tf.concat(pixel_values, axis=0),
            "image_grid_thw": tf.concat(grid_thw, axis=0),
            "vision_indices": tf.concat(flat_indices, axis=0),
        }
