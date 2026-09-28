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
from keras_hub.src.utils.tensor_utils import in_tf_function


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


@keras_hub_export("keras_hub.models.MuseGlimmerCausalLMPreprocessor")
class MuseGlimmerCausalLMPreprocessor(CausalLMPreprocessor):
    """MuseGlimmer causal LM preprocessor with optional image/video inputs.

    For text-only usage this behaves identically to the base
    `CausalLMPreprocessor`. When an `image_converter`/`video_converter` is
    provided and the input contains `"images"`/`"videos"`, each occurrence
    of the tokenizer's `image_token_id`/`video_token_id` placeholder in the
    prompt is expanded to the number of merged vision tokens for that
    image/frame, and flat `vision_indices` are computed for scattering the
    vision encoder's output into the text embedding sequence. MuseGlimmer
    uses standard sequential text positions throughout — no M-RoPE is
    needed, since spatial awareness lives entirely inside the vision tower.

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
        if converter is None:
            raise ValueError(
                f"`{media_name}` were given, but this preprocessor has no "
                f"`{media_name[:-1]}_converter`."
            )
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

    def generate_preprocess(self, x, sequence_length=None):
        """Convert prompts and optional media to generation inputs.

        `x` is a prompt, a list of prompts, or a dict with a `"prompts"`
        key and optional `"images"` and `"videos"` keys. Each media value
        is one item, a batch of same-size items, or a list of items. Items
        in a list can have different sizes. The media path runs eagerly
        with NumPy and `keras.ops`, so it does not need TensorFlow.
        """
        images, videos = None, None
        if isinstance(x, dict):
            images = x.get("images", None)
            videos = x.get("videos", None)
            if images is None and videos is None:
                x = x["prompts"]

        if images is None and videos is None:
            return super().generate_preprocess(
                x, sequence_length=sequence_length
            )

        if in_tf_function():
            raise ValueError(
                "Image and video inputs need eager execution. Do not call "
                "`generate_preprocess()` with media inside `tf.data` or "
                "`tf.function`."
            )
        if not self.built:
            self.build(None)

        sequence_length = sequence_length or self.sequence_length
        prompts, batched, outer_shape = canonicalize_python_string_inputs(
            x["prompts"]
        )
        if outer_shape is not None:
            raise ValueError(
                "`prompts` must be a string or a list of strings. "
                f"Received: {x['prompts']}"
            )

        pixel_values, grid_thw = [], []
        num_image_tokens, num_video_tokens = [], []
        if images is not None:
            patches, grids, num_image_tokens = self._convert_media(
                images, self.image_converter, 3, "images"
            )
            pixel_values.extend(patches)
            grid_thw.extend(grids)
        if videos is not None:
            patches, grids, num_video_tokens = self._convert_media(
                videos, self.video_converter, 4, "videos"
            )
            pixel_values.extend(patches)
            grid_thw.extend(grids)

        tokenized = [
            convert_to_numpy(self.tokenizer(prompt)).tolist()
            for prompt in prompts
        ]
        expanded_sequences = self._expand_vision_placeholders(
            tokenized, num_image_tokens, num_video_tokens
        )
        token_ids, padding_mask = self.packer(
            expanded_sequences if batched else expanded_sequences[0],
            sequence_length=sequence_length,
            add_end_value=False,
        )

        vision_indices = self._compute_vision_indices(token_ids)
        num_vision_tokens = sum(num_image_tokens) + sum(num_video_tokens)
        if vision_indices.shape[0] != num_vision_tokens:
            raise ValueError(
                f"`sequence_length={sequence_length}` truncates the "
                f"expanded image/video tokens. The media need "
                f"{num_vision_tokens} tokens, but only "
                f"{vision_indices.shape[0]} fit. Increase "
                "`sequence_length`."
            )
        if pixel_values:
            pixel_values = np.concatenate(pixel_values, axis=0)
            grid_thw = np.stack(grid_thw, axis=0)
        else:
            pixel_values = np.zeros((0, 0), dtype="float32")
            grid_thw = np.zeros((0, 3), dtype="int32")

        return convert_preprocessing_outputs_python(
            {
                "token_ids": token_ids,
                "padding_mask": padding_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": grid_thw,
                "vision_indices": vision_indices,
            }
        )

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
