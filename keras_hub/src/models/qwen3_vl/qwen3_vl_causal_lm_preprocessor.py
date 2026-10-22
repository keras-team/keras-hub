import itertools
import math
import re

import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.qwen3_vl.qwen3_vl_backbone import Qwen3_VLBackbone
from keras_hub.src.models.qwen3_vl.qwen3_vl_image_converter import (
    Qwen3_VLImageConverter,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_tokenizer import Qwen3_VLTokenizer
from keras_hub.src.models.qwen3_vl.qwen3_vl_video_converter import (
    Qwen3_VLVideoConverter,
)
from keras_hub.src.utils.tensor_utils import (
    convert_preprocessing_outputs_python,
)
from keras_hub.src.utils.tensor_utils import convert_to_list
from keras_hub.src.utils.tensor_utils import convert_to_numpy
from keras_hub.src.utils.tensor_utils import tensor_to_list
from keras_hub.src.utils.tensor_utils import tf

# Values of `mm_token_type_ids` in the HF implementation.
TEXT_TOKEN, IMAGE_TOKEN, VIDEO_TOKEN = 0, 1, 2


def _text_position_ids(padding_mask):
    """1D positions repeated on the (T, H, W) channels, `(..., seq_len, 3)`.

    Padding positions are set to 0, as in HF `get_rope_index`. The result has
    the same tensor type as `padding_mask`, or a NumPy array in a Grain
    pipeline.
    """
    if tf is not None and isinstance(padding_mask, tf.Tensor):
        mask = tf.cast(padding_mask, "int32")
        positions = (tf.cumsum(mask, axis=-1) - 1) * mask
        return tf.repeat(positions[..., None], 3, axis=-1)
    mask = ops.cast(padding_mask, "int32")
    positions = (ops.cumsum(mask, axis=-1) - 1) * mask
    return convert_preprocessing_outputs_python(
        ops.repeat(positions[..., None], 3, axis=-1)
    )


def _mrope_positions(token_types, image_grids, video_grids, merge_size):
    """Port of HF `get_rope_index` for one unpadded sequence.

    Args:
        token_types: list of `seq_len` ints with 0 for text, 1 for image and
            2 for video placeholder tokens.
        image_grids: list of `(t, h, w)` grids, one per image run.
        video_grids: list of `(t, h, w)` grids, one per video run. Videos are
            already split into one grid per temporal patch.
        merge_size: int. Spatial merge size of the vision encoder.

    Returns:
        A list with the `[t, h, w]` positions of every token.
    """
    grids = {IMAGE_TOKEN: iter(image_grids), VIDEO_TOKEN: iter(video_grids)}
    positions = []
    current = 0
    for kind, run in itertools.groupby(token_types):
        if kind == TEXT_TOKEN:
            length = len(list(run))
            positions += [[current + i] * 3 for i in range(length)]
            current += length
            continue
        _, height, width = next(grids[kind])
        for row in range(height // merge_size):
            for col in range(width // merge_size):
                positions.append([current, current + row, current + col])
        current += max(height, width) // merge_size
    return positions


def _media_list(item, ndim):
    """Normalize the media of one sample to a list of numpy arrays."""
    if item is None:
        return []
    if isinstance(item, (list, tuple)):
        return [convert_to_numpy(m) for m in item]
    array = convert_to_numpy(item)
    return [array] if array.ndim == ndim else list(array)


def _fps_list(item, count, default):
    """Normalize the source frame rates of one sample to one per video."""
    if item is None:
        return [default] * count
    item = convert_to_list(item)
    if isinstance(item, (list, tuple)):
        return [float(f) for f in item]
    return [float(item)] * count


@keras_hub_export("keras_hub.models.Qwen3_VLCausalLMPreprocessor")
class Qwen3_VLCausalLMPreprocessor(CausalLMPreprocessor):
    """Qwen3-VL Causal LM preprocessor.

    This preprocessor turns text, images and videos into the inputs of
    `Qwen3_VLBackbone`. With plain strings it behaves like the base
    `CausalLMPreprocessor` and also returns text `position_ids`, which works
    inside `tf.data`. With a dictionary that holds `"images"` and/or
    `"videos"`, the media is processed on the host in Python, so that path
    runs eagerly and cannot be used inside a `tf.data` map.

    A prompt marks where each image goes with `<|image_pad|>` and where each
    video goes with `<|video_pad|>`. Media are consumed in the order of the
    markers. Like the HF processor, each image marker is expanded to one pad
    token per merged patch, and a video marker is replaced by the blocks
    below while the `<|vision_start|>` and `<|vision_end|>` that the chat
    template puts around it are kept. Each video is cut into chunks of
    `temporal_patch_size` frames, and each chunk gets a `<t seconds>` text
    prefix, where `t` is the mean timestamp of the frames in the chunk,
    followed by its own vision start, pad tokens and vision end.

    Args:
        tokenizer: A `keras_hub.models.Qwen3_VLTokenizer` instance.
        image_converter: A `keras_hub.layers.Qwen3_VLImageConverter` instance
            or `None`. Required to pass images.
        video_converter: A `keras_hub.layers.Qwen3_VLVideoConverter` instance
            or `None`. Required to pass videos.
        sequence_length: int. The length of the packed inputs. Defaults to
            `1024`.
        add_start_token: bool. Whether to prepend the tokenizer start token.
            Defaults to `False`.
        add_end_token: bool. Whether to append the tokenizer end token when
            calling the layer for training. Defaults to `True`.
        default_video_fps: float. Frame rate assumed for a video that comes
            without `"video_fps"`. Defaults to `24.0`, the HF fallback.

    Call arguments:
        x: A string, a list of strings, or a dict with these keys:
            `"prompts"`: a string or a list of strings.
            `"images"`: optional. For a string prompt, an image or a list of
            images, as `(height, width, 3)` arrays. For a list of prompts, one
            such entry per prompt.
            `"videos"`: optional. Same layout as `"images"`, with
            `(frames, height, width, 3)` arrays holding every decoded frame.
            Frames are chosen with `video_converter.sample_frame_indices`.
            `"video_fps"`: optional. Frame rate of each source video, laid
            out like `"videos"`.
        y: Label data. Should always be `None`, as the layer generates labels.
        sample_weight: Label weights. Should always be `None`.
        sequence_length: Pass to override the configured sequence length.

    Returns:
        A dict with `"token_ids"` and `"padding_mask"` of shape `(batch,
        sequence_length)` and `"position_ids"` of shape `(batch,
        sequence_length, 3)`. When the input has media, the dict also holds
        `"pixel_values"` `(batch, num_patches, temporal_patch_size,
        patch_size, patch_size, 3)`, `"image_grid_thw"` `(batch, num_items,
        3)` with images and videos in prompt order, and `"vision_indices"`
        `(batch, num_vision_tokens)`. The vision indices are positions in the
        flattened `(batch * sequence_length)` token axis. Samples with fewer
        vision tokens than the longest one are padded with zero patches and
        zero grids, and their extra indices point at padding positions. The
        batch axis is dropped for a single string prompt. Calling the layer
        returns `(x, y, sample_weight)` as in the base class.

    Examples:
    ```python
    preprocessor = keras_hub.models.Qwen3_VLCausalLMPreprocessor.from_preset(
        "hf://Qwen/Qwen3-VL-2B-Instruct"
    )
    prompt = preprocessor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ],
        add_generation_prompt=True,
    )
    inputs = preprocessor.generate_preprocess(
        {"prompts": prompt, "images": [np.zeros((224, 224, 3))]}
    )
    ```
    """

    backbone_cls = Qwen3_VLBackbone
    tokenizer_cls = Qwen3_VLTokenizer
    image_converter_cls = Qwen3_VLImageConverter
    video_converter_cls = Qwen3_VLVideoConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        video_converter=None,
        sequence_length=1024,
        add_start_token=False,
        add_end_token=True,
        default_video_fps=24.0,
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
        self.default_video_fps = default_video_fps

    @staticmethod
    def apply_chat_template(
        messages, add_generation_prompt=False, add_vision_id=False
    ):
        """Format a conversation with the Qwen3-VL chat template.

        Returns the prompt string for `generate`, with one placeholder per
        image or video. Pass the media itself as `"images"` and `"videos"`,
        in the same order. The output equals the Hugging Face chat template for
        the conversations accepted here, which are those without tools. Input
        that the Hugging Face template would silently drop raises an error
        instead: a `tool` role, a system message that is not first, and an
        image or video in a system or assistant message.

        Args:
            messages: list of dicts with a `"role"` of `"system"`, `"user"` or
                `"assistant"` and a `"content"` that is a string or a list of
                parts. A part is a string, `{"type": "text", "text": ...}`,
                `{"type": "image"}` or `{"type": "video"}`; the Hugging Face
                forms `{"image": ...}`, `{"image_url": ...}` and
                `{"video": ...}` also work. Only user messages may hold
                images and videos, and only the first message may be a
                system message.
            add_generation_prompt: bool. Whether to end with the opening of an
                assistant turn, which is what generation continues from.
            add_vision_id: bool. Whether to label each image and video with
                `Picture N:` or `Video N:`.

        Example:
        ```python
        prompt = Qwen3_VLCausalLMPreprocessor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe the image."},
                    ],
                }
            ],
            add_generation_prompt=True,
        )
        ```
        """
        if isinstance(messages, dict):
            messages = [messages]
        if not messages:
            raise ValueError("`messages` must not be empty.")
        counts = {"image": 0, "video": 0}
        prompt = ""
        for i, message in enumerate(messages):
            role, content = message["role"], message["content"]
            if role not in ("system", "user", "assistant"):
                raise ValueError(
                    "`role` must be 'system', 'user' or 'assistant'. "
                    f"Received: {role}"
                )
            if role == "system" and i > 0:
                raise ValueError("Only the first message may be a system one.")
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            elif not isinstance(content, (list, tuple)):
                raise ValueError(
                    "`content` must be a string or a list of parts. "
                    f"Received: {content!r}"
                )
            text = ""
            for part in content:
                if isinstance(part, str):
                    part = {"type": "text", "text": part}
                if part.get("type") == "image" or any(
                    k in part for k in ("image", "image_url")
                ):
                    kind = "image"
                elif part.get("type") == "video" or "video" in part:
                    kind = "video"
                elif "text" in part:
                    text += part["text"]
                    continue
                else:
                    raise ValueError(f"Unsupported content part: {part!r}")
                if role != "user":
                    raise ValueError(f"A {role} message cannot hold a {kind}.")
                counts[kind] += 1
                if add_vision_id:
                    label = "Picture" if kind == "image" else "Video"
                    text += f"{label} {counts[kind]}: "
                text += f"<|vision_start|><|{kind}_pad|><|vision_end|>"
            prompt += f"<|im_start|>{role}\n{text}<|im_end|>\n"
        if add_generation_prompt:
            prompt += "<|im_start|>assistant\n"
        return prompt

    @property
    def _merge_size(self):
        converter = self.image_converter or self.video_converter
        return converter.spatial_merge_size

    @staticmethod
    def _has_media(x):
        return isinstance(x, dict) and (
            x.get("images") is not None or x.get("videos") is not None
        )

    def _empty_vision_inputs(self, padding_mask):
        """Zero-size vision inputs, so text-only data fits a vision model."""
        converter = self.image_converter or self.video_converter
        if converter is None:
            return {}
        patch = (
            converter.temporal_patch_size,
            converter.patch_size,
            converter.patch_size,
            3,
        )
        if tf is not None and isinstance(padding_mask, tf.Tensor):
            lead = tf.shape(padding_mask)[:-1]

            def zeros(tail, dtype):
                return tf.zeros(tf.concat([lead, tail], 0), dtype)

        else:
            lead = tuple(padding_mask.shape[:-1])

            def zeros(tail, dtype):
                return convert_preprocessing_outputs_python(
                    ops.zeros(lead + tuple(tail), dtype)
                )

        return {
            "pixel_values": zeros((0,) + patch, "float32"),
            "image_grid_thw": zeros((0, 3), "int32"),
            "vision_indices": zeros((0,), "int32"),
        }

    def _generate_preprocess_python(self, x, sequence_length=None):
        if isinstance(x, dict):
            x = x["prompts"]
        output = super()._generate_preprocess_python(
            x, sequence_length=sequence_length
        )
        output["position_ids"] = _text_position_ids(output["padding_mask"])
        output.update(self._empty_vision_inputs(output["padding_mask"]))
        return output

    def _call_python(self, x, y=None, sample_weight=None, sequence_length=None):
        if isinstance(x, dict):
            x = x["prompts"]
        output = super()._call_python(
            x, y=y, sample_weight=sample_weight, sequence_length=sequence_length
        )
        x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(output)
        x["position_ids"] = _text_position_ids(x["padding_mask"])
        x.update(self._empty_vision_inputs(x["padding_mask"]))
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def generate_preprocess(self, x, sequence_length=None):
        """Convert prompts, images and videos to model inputs for generation.

        Unlike calling the layer for training, this does not compute labels
        and never appends the end token.
        """
        if not self._has_media(x):
            return super().generate_preprocess(
                x, sequence_length=sequence_length
            )
        outputs = self._preprocess_media(
            x, sequence_length or self.sequence_length, training=False
        )
        return convert_preprocessing_outputs_python(outputs)

    def call(self, x, y=None, sample_weight=None, sequence_length=None):
        if not self._has_media(x):
            return super().call(
                x,
                y=y,
                sample_weight=sample_weight,
                sequence_length=sequence_length,
            )
        outputs = self._preprocess_media(
            x, sequence_length or self.sequence_length, training=True
        )
        return convert_preprocessing_outputs_python(outputs)

    def _split_media(self, x):
        """Return `(prompts, images, videos, fps, batched)` per sample."""
        prompts = x["prompts"]
        if tf is not None and isinstance(prompts, tf.Tensor):
            prompts = tensor_to_list(prompts)
        else:
            prompts = convert_to_list(prompts)
        batched = not isinstance(prompts, str)
        if not batched:
            prompts = [prompts]

        def per_sample(key):
            value = x.get(key)
            if value is None:
                return [None] * len(prompts)
            value = value if batched else [value]
            if len(value) != len(prompts):
                raise ValueError(
                    f"`{key}` must have one entry per prompt. Received "
                    f"{len(value)} entries for {len(prompts)} prompts."
                )
            return value

        images = [_media_list(m, 3) for m in per_sample("images")]
        videos = [_media_list(m, 4) for m in per_sample("videos")]
        fps = [
            _fps_list(f, len(v), self.default_video_fps)
            for f, v in zip(per_sample("video_fps"), videos)
        ]
        return prompts, images, videos, fps, batched

    def _timestamps(self, indices, source_fps):
        """Mean timestamp in seconds of every temporal patch of frames."""
        size = self.video_converter.temporal_patch_size
        indices = list(indices)
        indices += [indices[-1]] * (-len(indices) % size)
        seconds = [index / source_fps for index in indices]
        return [
            sum(seconds[i : i + size]) / size
            for i in range(0, len(seconds), size)
        ]

    def _expand_sample(self, prompt, images, videos, fps):
        """Expand the markers of one prompt and process its media.

        Returns:
            The expanded prompt and a dict with the patches, the grids in
            prompt order, and the image and per-chunk video grids.
        """
        tok = self.tokenizer
        pattern = re.compile(
            f"(?P<video>{re.escape(tok.video_token)})"
            f"|(?P<image>{re.escape(tok.image_token)})"
        )
        merge = self._merge_size
        sample = {"patches": [], "grids": [], "images": [], "videos": []}
        images, videos, fps = list(images), list(videos), list(fps)

        def expand(match):
            if match.lastgroup == "image":
                if not images or self.image_converter is None:
                    raise ValueError(
                        "The prompt has more image markers than images, or "
                        "images were passed without an `image_converter`."
                    )
                out = self.image_converter(images.pop(0))
                grid = convert_to_list(out["grid_thw"])
                sample["images"].append(grid)
                text = tok.image_token * (math.prod(grid) // merge**2)
            else:
                if not videos or self.video_converter is None:
                    raise ValueError(
                        "The prompt has more video markers than videos, or "
                        "videos were passed without a `video_converter`."
                    )
                video, video_fps = videos.pop(0), fps.pop(0)
                indices = convert_to_list(
                    self.video_converter.sample_frame_indices(
                        len(video), video_fps
                    )
                )
                out = self.video_converter(video[indices])
                grid = convert_to_list(out["grid_thw"])
                count = grid[1] * grid[2] // merge**2
                text = ""
                for seconds in self._timestamps(indices, video_fps):
                    text += f"<{seconds:.1f} seconds>" + tok.vision_start_token
                    text += tok.video_token * count + tok.vision_end_token
                    sample["videos"].append([1, grid[1], grid[2]])
            sample["patches"].append(out["patches"])
            sample["grids"].append(grid)
            return text

        expanded = pattern.sub(expand, prompt)
        if images or videos:
            raise ValueError(
                "The prompt has fewer image or video markers than the media "
                "that was passed."
            )
        return expanded, sample

    def _preprocess_media(self, x, sequence_length, training):
        prompts, images, videos, fps, batched = self._split_media(x)
        if not prompts:
            raise ValueError("`prompts` must not be empty.")
        expanded, samples = zip(
            *[
                self._expand_sample(*args)
                for args in zip(prompts, images, videos, fps)
            ]
        )
        if not self.built:
            self.build(None)
        token_ids, padding_mask = self.packer(
            self.tokenizer(list(expanded)),
            sequence_length=sequence_length + int(training),
            add_start_value=self.add_start_token,
            add_end_value=training and self.add_end_token,
        )
        token_ids = convert_to_list(token_ids)
        padding_mask = convert_to_list(padding_mask)
        batch_size, total = len(samples), len(token_ids[0])

        # The model sees the first `sequence_length` tokens. Training labels
        # are the same tokens shifted by one.
        image_id = self.tokenizer.image_token_id
        video_id = self.tokenizer.video_token_id
        types = [
            [
                IMAGE_TOKEN
                if token == image_id
                else VIDEO_TOKEN
                if token == video_id
                else TEXT_TOKEN
                for token in row
            ]
            for row in token_ids
        ]
        vision_indices = self._vision_indices(
            [row[:sequence_length] for row in types],
            [row[:sequence_length] for row in padding_mask],
            samples,
        )

        position_ids = []
        for b, sample in enumerate(samples):
            length = sum(padding_mask[b])
            positions = _mrope_positions(
                types[b][:length],
                sample["images"],
                sample["videos"],
                self._merge_size,
            )
            position_ids.append(positions + [[0, 0, 0]] * (total - length))

        converter = self.image_converter or self.video_converter
        patch_shape = (
            converter.temporal_patch_size,
            converter.patch_size,
            converter.patch_size,
            3,
        )
        num_patches = max(sum(len(p) for p in s["patches"]) for s in samples)
        max_items = max(len(s["grids"]) for s in samples)
        pixel_values, grids = [], []
        for sample in samples:
            if sample["patches"]:
                patches = ops.concatenate(sample["patches"])
            else:
                patches = ops.zeros((0,) + patch_shape)
            padding = [(0, num_patches - len(patches))] + [(0, 0)] * 4
            pixel_values.append(ops.pad(ops.cast(patches, "float32"), padding))
            grids.append(
                sample["grids"]
                + [[0, 0, 0]] * (max_items - len(sample["grids"]))
            )

        def to_tensor(rows, dtype):
            return ops.convert_to_tensor(rows, dtype=dtype)

        x = {
            "token_ids": to_tensor(
                [r[:sequence_length] for r in token_ids], "int32"
            ),
            "padding_mask": to_tensor(
                [r[:sequence_length] for r in padding_mask], "bool"
            ),
            "position_ids": to_tensor(
                [r[:sequence_length] for r in position_ids], "int32"
            ),
            "pixel_values": ops.stack(pixel_values),
            "image_grid_thw": ops.reshape(
                to_tensor(grids, "int32"), (batch_size, max_items, 3)
            ),
            "vision_indices": to_tensor(vision_indices, "int32"),
        }
        if not training:
            return x if batched else {k: v[0] for k, v in x.items()}
        y = to_tensor([r[1:] for r in token_ids], "int32")
        sample_weight = to_tensor([r[1:] for r in padding_mask], "bool")
        if not batched:
            x = {k: v[0] for k, v in x.items()}
            y, sample_weight = y[0], sample_weight[0]
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def _vision_indices(self, types, padding_mask, samples):
        """Flat indices of the vision tokens, padded to a common width."""
        length = len(types[0])
        indices = []
        for b, sample in enumerate(samples):
            expected = sum(len(p) for p in sample["patches"])
            found = [i for i, kind in enumerate(types[b]) if kind]
            if len(found) * self._merge_size**2 != expected:
                raise ValueError(
                    "The vision tokens do not fit in `sequence_length`. "
                    f"Sample {b} needs {expected // self._merge_size**2} "
                    f"vision tokens but only {len(found)} are left after "
                    "truncation."
                )
            indices.append([i + b * length for i in found])
        width = max(len(i) for i in indices)
        for b in range(len(samples)):
            missing = width - len(indices[b])
            if missing:
                # Padded features are written to padding positions of the
                # sample, so they never overwrite a real token.
                free = [i for i, keep in enumerate(padding_mask[b]) if not keep]
                if not free:
                    raise ValueError(
                        f"Sample {b} has no padding position left for the "
                        "padded vision features. Increase `sequence_length`."
                    )
                indices[b] += [
                    free[i % len(free)] + b * length for i in range(missing)
                ]
        return indices

    def get_config(self):
        config = super().get_config()
        config["default_video_fps"] = self.default_video_fps
        return config
