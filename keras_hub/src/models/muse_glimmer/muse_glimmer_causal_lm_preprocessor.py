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

    `media` is one item of rank `item_rank`, a batch of rank
    `item_rank + 1`, or a list of items. Items in a list can have
    different sizes.
    """
    if isinstance(media, (list, tuple)):
        if not media:
            return []
        if all(_rank(item) == item_rank for item in media):
            return list(media)
    rank = _rank(media)
    if rank == item_rank:
        return [media]
    if rank == item_rank + 1:
        return [media[i] for i in range(len(media))]
    raise ValueError(
        f"`{media_name}` must be one {media_name[:-1]} of rank {item_rank}, "
        f"a batch of rank {item_rank + 1}, or a list of "
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


@keras_hub_export("keras_hub.models.MuseGlimmerCausalLMPreprocessor")
class MuseGlimmerCausalLMPreprocessor(CausalLMPreprocessor):
    """MuseGlimmer causal LM preprocessor with optional image/video inputs.

    For text-only usage this behaves identically to the base
    `CausalLMPreprocessor`. When an `image_converter`/`video_converter` is
    provided and the input contains `"images"`/`"videos"`, each occurrence
    of the tokenizer's `image_token_id`/`video_token_id` placeholder in the
    prompt is expanded to the number of merged vision tokens for that
    image/frame, and `vision_indices` are computed for scattering the
    vision encoder's output into the text embedding sequence. MuseGlimmer
    uses standard sequential text positions throughout — no M-RoPE is
    needed, since spatial awareness lives entirely inside the vision tower.

    For training, each prompt needs the same number of same-size media
    items. `sample_weight` is zero where the label is a vision placeholder.

    Args:
        tokenizer: A `MuseGlimmerTokenizer` instance.
        image_converter: A `MuseGlimmerImageConverter` instance, or `None`.
        video_converter: A `MuseGlimmerVideoConverter` instance, or `None`.
        sequence_length: int. The length of the packed inputs.
        add_start_token: bool. Whether to prepend the tokenizer's start
            token. Defaults to `True`.
        add_end_token: bool. Whether to append the tokenizer's end token.
            Defaults to `True`.
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
        token_id)`. Images come before videos. The vision tokens use the
        same order.
        """
        if not isinstance(x, dict):
            return x, []
        media_entries = []
        for media_name, converter, item_rank, token_id in (
            ("images", self.image_converter, 3, self.tokenizer.image_token_id),
            ("videos", self.video_converter, 4, self.tokenizer.video_token_id),
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
                (media_name, media, converter, item_rank, token_id)
            )
        return x["prompts"], media_entries

    # === Python path ===

    def _compute_vision_indices(self, token_ids):
        token_ids = convert_to_numpy(token_ids).reshape(-1)
        img_indices = np.flatnonzero(token_ids == self.tokenizer.image_token_id)
        vid_indices = np.flatnonzero(token_ids == self.tokenizer.video_token_id)
        return np.concatenate([img_indices, vid_indices]).astype("int32")

    def _expand_vision_placeholders(
        self, sequences, num_image_tokens, num_video_tokens
    ):
        """Expand placeholders in a batch of token id lists.

        Media items map to placeholders in order across the whole batch.
        The i-th image placeholder takes `num_image_tokens[i]` tokens.
        """
        image_token_id = self.tokenizer.image_token_id
        video_token_id = self.tokenizer.video_token_id
        counts = {
            image_token_id: ("image", num_image_tokens),
            video_token_id: ("video", num_video_tokens),
        }
        next_index = {image_token_id: 0, video_token_id: 0}
        expanded_sequences = []
        for ids in sequences:
            expanded = []
            for tok in ids:
                if tok not in counts:
                    expanded.append(tok)
                    continue
                media_name, media_counts = counts[tok]
                i = next_index[tok]
                if i >= len(media_counts):
                    raise ValueError(
                        f"The prompts contain more {media_name} "
                        f"placeholders than the {len(media_counts)} "
                        f"{media_name}(s) given."
                    )
                expanded.extend([tok] * media_counts[i])
                next_index[tok] = i + 1
            expanded_sequences.append(expanded)
        for tok, (media_name, media_counts) in counts.items():
            if next_index[tok] != len(media_counts):
                raise ValueError(
                    f"{len(media_counts)} {media_name}(s) were given, but "
                    f"the prompts contain {next_index[tok]} {media_name} "
                    "placeholder(s). Add one placeholder per "
                    f"{media_name}."
                )
        return expanded_sequences

    def _num_merged_tokens(self, grid_thw, merge_size):
        counts = []
        for t, h, w in grid_thw:
            counts.append(t * (h // merge_size) * (w // merge_size))
        return counts

    def _convert_media(self, media, converter, item_rank, media_name):
        """Run `converter` on each media item separately.

        Returns the per-item patches, the per-item `grid_thw` and the
        per-item count of merged vision tokens.
        """
        patches, grids = [], []
        for item in _split_media(media, item_rank, media_name):
            item = keras.ops.convert_to_tensor(convert_to_numpy(item))
            result = converter(item)
            patches.append(convert_to_numpy(result["patches"]))
            grids.append(convert_to_numpy(result["grid_thw"]).astype("int32"))
        num_tokens = self._num_merged_tokens(
            [grid.tolist() for grid in grids], converter.merge_size
        )
        return patches, grids, num_tokens

    def _pack_with_media_python(self, x, sequence_length, add_end_value):
        """Tokenize and pack prompts, and convert same-size media per row.

        Returns `token_ids` and `padding_mask` of shape `(batch_size,
        sequence_length)`, a list of `(media_name, token_id, patches,
        grid_thw, tokens_per_row)` per media kind, and `batched`. `patches`
        has shape `(batch_size, num_patches, patch_dim)`. `grid_thw` has
        shape `(batch_size, num_items, 3)`.
        """
        prompts, media_entries = self._split_inputs(x)
        prompts, batched, _ = canonicalize_python_string_inputs(prompts)
        batch_size = len(prompts)

        media_outputs = []
        num_tokens_per_item = {"images": [], "videos": []}
        for media_name, media, converter, item_rank, token_id in media_entries:
            media = _media_per_row(
                convert_to_numpy(media), item_rank, batched, media_name
            )
            if media.shape[0] != batch_size:
                raise ValueError(
                    f"`{media_name}` must have one row per prompt. Received "
                    f"{batch_size} prompts and {media.shape[0]} rows."
                )
            num_items = media.shape[1]
            patches, grids, num_tokens = self._convert_media(
                media.reshape((-1,) + media.shape[2:]),
                converter,
                item_rank,
                media_name,
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
            num_tokens_per_item[media_name] = num_tokens

        tokenized = [
            convert_to_numpy(ids).tolist() for ids in self.tokenizer(prompts)
        ]
        expanded_sequences = self._expand_vision_placeholders(
            tokenized,
            num_tokens_per_item["images"],
            num_tokens_per_item["videos"],
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

    def _call_python(self, x, y=None, sample_weight=None, sequence_length=None):
        prompts, media_entries = self._split_inputs(x)
        if not media_entries:
            return super()._call_python(
                prompts,
                y=y,
                sample_weight=sample_weight,
                sequence_length=sequence_length,
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
        if not media_entries:
            return super()._generate_preprocess_python(
                prompts, sequence_length=sequence_length
            )
        if not self.built:
            self.build(None)

        sequence_length = sequence_length or self.sequence_length
        prompts, batched, _ = canonicalize_python_string_inputs(prompts)

        pixel_values, grid_thw = [], []
        num_tokens_per_item = {"images": [], "videos": []}
        for media_name, media, converter, item_rank, _ in media_entries:
            patches, grids, num_tokens = self._convert_media(
                media, converter, item_rank, media_name
            )
            pixel_values.extend(patches)
            grid_thw.extend(grids)
            num_tokens_per_item[media_name] = num_tokens

        tokenized = [
            convert_to_numpy(self.tokenizer(prompt)).tolist()
            for prompt in prompts
        ]
        expanded_sequences = self._expand_vision_placeholders(
            tokenized,
            num_tokens_per_item["images"],
            num_tokens_per_item["videos"],
        )
        token_ids, padding_mask = self.packer(
            expanded_sequences if batched else expanded_sequences[0],
            sequence_length=sequence_length,
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

    def _convert_media_tf(self, media, converter, item_rank, batched, name):
        """Run `converter` on each same-size media item with `tf.map_fn`.

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
        item_outputs = (
            tf.function(converter)
            .get_concrete_function(item_spec)
            .structured_outputs
        )
        outputs = tf.map_fn(
            converter,
            items,
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
        return patches, grid_thw, num_items, num_tokens

    def _pack_with_media_tf(self, x, sequence_length, add_end_value):
        """TensorFlow version of `_pack_with_media_python`."""
        prompts, media_entries = self._split_inputs(x)
        batched = prompts.shape.rank == 1
        if not batched:
            prompts = prompts[tf.newaxis]
        token_ids = self.tokenizer(prompts)

        media_outputs = []
        for media_name, media, converter, item_rank, token_id in media_entries:
            patches, grids, num_items, num_tokens = self._convert_media_tf(
                media, converter, item_rank, batched, media_name
            )
            media_outputs.append(
                (media_name, token_id, patches, grids, num_items * num_tokens)
            )
            # Repeat each placeholder `num_tokens` times.
            flat_ids = token_ids.flat_values
            repeats = tf.where(flat_ids == token_id, num_tokens, 1)
            row_lengths = tf.reduce_sum(
                token_ids.with_flat_values(repeats), axis=1
            )
            token_ids = tf.RaggedTensor.from_row_lengths(
                tf.repeat(flat_ids, repeats), row_lengths
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
            return super()._call_python(
                prompts,
                y=y,
                sample_weight=sample_weight,
                sequence_length=sequence_length,
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

    def get_config(self):
        config = super().get_config()
        if self.image_converter is not None:
            config["image_converter"] = keras.layers.serialize(
                self.image_converter
            )
        if self.video_converter is not None:
            config["video_converter"] = keras.layers.serialize(
                self.video_converter
            )
        return config
