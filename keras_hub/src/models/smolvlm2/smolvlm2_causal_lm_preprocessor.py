"""CausalLM preprocessor for SmolVLM2 models."""

import re

import keras
import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.smolvlm2.smolvlm2_backbone import SmolVLM2Backbone
from keras_hub.src.models.smolvlm2.smolvlm2_image_converter import (
    SmolVLM2ImageConverter,
)
from keras_hub.src.models.smolvlm2.smolvlm2_tokenizer import SmolVLM2Tokenizer
from keras_hub.src.models.smolvlm2.smolvlm2_video_converter import (
    SmolVLM2VideoConverter,
)
from keras_hub.src.utils.tensor_utils import convert_to_numpy
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tf

# HF-compatible video prompt templates.
DEFAULT_VIDEO_INTRO = (
    "You are provided the following series of {frame_count} frames "
    "from a {video_duration} [H:MM:SS] video.\n"
)
DEFAULT_MEDIA_OUTTRO = "\n\n"
FRAME_TIMESTAMP_MESSAGE = "\nFrame from {timestamp}:"
# `<row_R_col_C>` tags marking each crop of a split image.
ROW_COL_PATTERN = re.compile(r"<row_\d+_col_\d+>")


def _get_image_prompt_string(
    image_seq_len,
    image_rows,
    image_cols,
    fake_token_around_image,
    image_token,
    global_image_token,
):
    """Build the expanded image prompt string.

    Replicates HF's ``_prompt_split_image`` / ``_prompt_single_image``.

    When ``image_rows == 0 and image_cols == 0``, the image was not split
    and a single global-image block is produced.  Otherwise, each
    ``(row, col)`` patch gets its own block plus a trailing global view.

    Args:
        image_seq_len: int. Number of ``<image>`` tokens per sub-image.
        image_rows: int. Number of row splits (0 if unsplit).
        image_cols: int. Number of column splits (0 if unsplit).
        fake_token_around_image: str.
        image_token: str.
        global_image_token: str.
    Returns:
        str. The expanded prompt fragment.
    """
    if image_rows == 0 and image_cols == 0:
        # Single unsplit image.
        return (
            f"{fake_token_around_image}"
            f"{global_image_token}"
            f"{image_token * image_seq_len}"
            f"{fake_token_around_image}"
        )

    # Split image: row×col patches + 1 global.
    text = ""
    for r in range(image_rows):
        for c in range(image_cols):
            text += (
                f"{fake_token_around_image}"
                f"<row_{r + 1}_col_{c + 1}>"
                f"{image_token * image_seq_len}"
            )
        text += "\n"
    text += (
        f"\n{fake_token_around_image}"
        f"{global_image_token}"
        f"{image_token * image_seq_len}"
        f"{fake_token_around_image}"
    )
    return text


@keras_hub_export("keras_hub.models.SmolVLM2CausalLMPreprocessor")
class SmolVLM2CausalLMPreprocessor(CausalLMPreprocessor):
    """Preprocessor for SmolVLM2 causal language models.

    This preprocessing layer handles tokenization and image processing
    for the SmolVLM2 model. It tokenizes text inputs, processes images
    via the image converter, and packs them into the format expected by
    ``SmolVLM2CausalLM``.

    For multimodal inputs, prompts should contain a single ``<image>``
    placeholder per image. The preprocessor expands each ``<image>``
    into the full HF-style token sequence:

        ``<fake_token_around_image><row_R_col_C><image>×N ... ``
        ``<global-img><image>×N<fake_token_around_image>``

    The expansion count ``N`` (``image_seq_len``) is computed as::

        N = (max_image_size / patch_size)² / scale_factor²

    For text-only inputs, no ``pixel_values`` or ``vision_indices``
    are returned; `SmolVLM2Backbone` fills in empty placeholders.

    During training (``fit()``), image inputs are supported only when the
    image converter is created with ``do_image_splitting=False``: the number
    of crops for a split image depends on its original size, which cannot be
    resolved inside the ``tf.data`` graph. Video inputs are inference-only.

    Args:
        tokenizer: A ``SmolVLM2Tokenizer`` instance.
        image_converter: A ``SmolVLM2ImageConverter`` instance or ``None``.
        video_converter: A ``SmolVLM2VideoConverter`` instance or ``None``.
        sequence_length: int. Maximum sequence length. Default ``1024``.
        add_start_token: bool. Whether to prepend BOS. Default ``False``
            (matching HF's ``add_bos_token``).
        add_end_token: bool. Whether to append EOS. Default ``True``.
        image_seq_len: int. Number of ``<image>`` tokens per sub-image
            crop.  Default ``64`` for the 256M preset
            (``(512/16)² / 4² = 64``).
    """

    backbone_cls = SmolVLM2Backbone
    tokenizer_cls = SmolVLM2Tokenizer
    image_converter_cls = SmolVLM2ImageConverter
    video_converter_cls = SmolVLM2VideoConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        video_converter=None,
        sequence_length=1024,
        add_start_token=False,
        add_end_token=True,
        image_seq_len=64,
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
        self.image_seq_len = image_seq_len
        # Per-video metadata: list of dicts with "fps" and optionally
        # "duration" and "frames_indices". Set before calling
        # generate_preprocess for accurate timestamps.
        self.video_metadata = None
        # Lazily filled `{"<row_R_col_C>": id}` cache, so the vocabulary is
        # only materialized once per preprocessor.
        self._row_col_token_ids = {}

    def build(self, input_shape):
        # Let parent create self.packer = StartEndPacker (used by
        # the inherited generate_preprocess for text-only inputs).
        super().build(input_shape)
        # MultiSegmentPacker for the training call() path which
        # needs prompt/response segment packing.
        self._training_packer = MultiSegmentPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sep_value=[],
            sequence_length=self.sequence_length,
        )

    # ------------------------------------------------------------------
    # Special-token-aware tokenization
    # ------------------------------------------------------------------
    def _build_special_token_map(self):
        """Return {token_string: token_id} for all registered specials."""
        special_map = {}
        for attr in [
            "start_token",
            "end_token",
            "image_token",
            "end_of_utterance_token",
            "fake_image_token",
            "global_image_token",
        ]:
            tok_str = getattr(self.tokenizer, attr, None)
            tok_id = getattr(self.tokenizer, f"{attr}_id", None)
            if tok_str is not None and tok_id is not None:
                special_map[tok_str] = tok_id
        return special_map

    def _row_col_token_id(self, token):
        """Return the vocabulary id of a `<row_R_col_C>` tag, or `None`.

        `get_vocabulary()` materializes the whole vocabulary, so the
        lookups are cached on the layer instead of being redone for every
        prompt that contains a split image.
        """
        if token in self._row_col_token_ids:
            return self._row_col_token_ids[token]
        vocab = self.tokenizer.get_vocabulary()
        if not isinstance(vocab, dict):
            vocab = {t: i for i, t in enumerate(vocab)}
        # Cache every row/col tag in one pass over the vocabulary.
        for vocab_token, token_id in vocab.items():
            if ROW_COL_PATTERN.fullmatch(vocab_token):
                self._row_col_token_ids[vocab_token] = token_id
        self._row_col_token_ids.setdefault(token, None)
        return self._row_col_token_ids[token]

    def _tokenize_with_special_tokens(self, text, special_map=None):
        """Tokenize text while preserving special tokens as single IDs.

        The KerasHub BPE tokenizer may break added special tokens into
        sub-word pieces. This method splits the input by known specials,
        tokenizes only the text segments, and manually inserts the
        correct token IDs.

        Additionally, ``<row_R_col_C>`` positional tokens are recognised
        and mapped to their vocabulary IDs. Tags that are not in the
        vocabulary are tokenized as plain text, matching how HuggingFace
        treats them when they are not registered as added tokens.

        Args:
            text: str. The fully-expanded prompt string.
            special_map: dict or None. ``{token_str: token_id}``.
        Returns:
            list[int]. The complete token ID sequence.
        """
        if special_map is None:
            special_map = self._build_special_token_map()

        for match in ROW_COL_PATTERN.finditer(text):
            token = match.group()
            if token in special_map:
                continue
            token_id = self._row_col_token_id(token)
            if token_id is not None:
                special_map[token] = token_id

        # Build regex for splitting.
        escaped = [re.escape(t) for t in special_map]
        # Also match <row_R_col_C> generically.
        pattern = re.compile(
            "(" + "|".join(escaped) + "|" + ROW_COL_PATTERN.pattern + ")"
        )

        parts = pattern.split(text)
        all_ids = []
        for part in parts:
            if part in special_map:
                all_ids.append(special_map[part])
            elif part:
                tokenized = self.tokenizer(part)
                if hasattr(tokenized, "numpy"):
                    all_ids.extend(tokenized.numpy().tolist())
                else:
                    all_ids.extend(list(tokenized))
        return all_ids

    # ------------------------------------------------------------------
    # Prompt extraction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_python_string(value):
        """Convert a single scalar prompt of any type to a Python str."""
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if hasattr(value, "numpy"):
            value = value.numpy()
        if isinstance(value, np.ndarray):
            value = value.item() if value.ndim == 0 else value.tolist()
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    @classmethod
    def _extract_prompt_strings(cls, prompts):
        """Return `(list[str], input_is_scalar)` for any prompt input."""
        if isinstance(prompts, (str, bytes)):
            return [cls._to_python_string(prompts)], True
        if isinstance(prompts, (list, tuple)):
            return [cls._to_python_string(p) for p in prompts], False
        # tf.Tensor, np.ndarray, or np scalar.
        value = prompts
        if hasattr(value, "numpy"):
            value = value.numpy()
        if isinstance(value, np.ndarray) and value.ndim > 0:
            return [cls._to_python_string(p) for p in value], False
        return [cls._to_python_string(value)], True

    def _image_token_strings(self):
        """Return the `(image, fake_image, global_image)` token strings."""
        return (
            getattr(self.tokenizer, "image_token", "<image>"),
            getattr(
                self.tokenizer,
                "fake_image_token",
                "<fake_token_around_image>",
            ),
            getattr(self.tokenizer, "global_image_token", "<global-img>"),
        )

    # ------------------------------------------------------------------
    # Vision indices
    # ------------------------------------------------------------------
    def _compute_vision_indices(self, token_ids):
        """Return flat indices where token_ids == image_token_id.

        This runs on the eager `generate_preprocess()` path only, so the
        indices are computed with NumPy. `convert_to_numpy` handles tensors
        from any backend, including torch tensors on a non-CPU device, where
        a bare `.numpy()` call would raise.

        Args:
            token_ids: `(batch, seq_len)` int tensor of packed token ids.
        Returns:
            `np.ndarray` of int32 flat indices into the flattened
            `(batch * seq_len,)` sequence.
        """
        image_token_id = getattr(self.tokenizer, "image_token_id", None)
        if image_token_id is None:
            return np.zeros((0,), dtype="int32")

        token_ids = convert_to_numpy(token_ids)
        mask = token_ids.reshape(-1) == image_token_id
        return np.where(mask)[0].astype("int32")

    def _graph_vision_indices(self, token_ids, num_images):
        """Graph-mode version of `_compute_vision_indices`.

        `SmolVLM2InterleaveEmbeddings` scatters into the *flattened*
        `(batch * seq_len, hidden_dim)` text tensor, so the indices here are
        flat offsets `batch_index * seq_len + position`.

        Args:
            token_ids: `(batch, seq_len)` int tensor of packed token ids.
            num_images: int tensor. Total number of (sub-)images in the batch.
        Returns:
            `(batch, image_seq_len)` int32 tensor of flat indices.
        """
        image_token_id = self.tokenizer.image_token_id
        mask = tf.equal(token_ids, image_token_id)
        counts = tf.reduce_sum(tf.cast(mask, "int32"), axis=-1)
        # Every `<image>` token must survive packing, otherwise the image
        # embeddings and the slots they are scattered into no longer line up.
        tf.debugging.assert_equal(
            tf.reduce_sum(counts),
            num_images * self.image_seq_len,
            message=(
                "The packed sequence does not contain "
                "`num_images * image_seq_len` `<image>` tokens. This usually "
                "means `sequence_length` is too short and truncated the "
                "expanded image tokens."
            ),
        )
        flat_mask = tf.reshape(mask, [-1])
        indices = tf.cast(tf.where(flat_mask)[:, 0], "int32")
        return tf.reshape(indices, (tf.shape(token_ids)[0], -1))

    # ------------------------------------------------------------------
    # call (training)
    # ------------------------------------------------------------------
    @preprocessing_function
    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        sequence_length = sequence_length or self.sequence_length

        # Handle both dict and string inputs.
        if isinstance(x, dict):
            images = x.get("images", None)
            videos = x.get("videos", None)
            prompts = x["prompts"]
            responses = x["responses"]
        else:
            images = None
            videos = None
            prompts = x
            responses = x

        if videos is not None:
            raise ValueError(
                "Training on video inputs is not supported. Pass `videos` to "
                "`generate_preprocess()` for inference instead."
            )
        if images is not None and self.image_converter is None:
            raise ValueError(
                "Received `images` in the input, but this preprocessor was "
                "created with `image_converter=None`."
            )

        pixel_values = None
        if images is not None:
            if self.image_converter.do_image_splitting:
                raise ValueError(
                    "Training with `do_image_splitting=True` is not "
                    "supported: the number of crops (and therefore the "
                    "number of `<image>` tokens) depends on each image's "
                    "original size, which cannot be resolved inside the "
                    "`tf.data` graph that `fit()` runs on. Set "
                    "`do_image_splitting=False` on the image converter."
                )
            pixel_values = self.image_converter(images)
            if isinstance(pixel_values, dict):
                pixel_values = pixel_values["pixel_values"]
            # With splitting disabled, every image expands to the same
            # constant token run, so the expansion can be done with graph
            # mode string ops.
            image_token, fake_token, global_token = self._image_token_strings()
            prompts = tf.strings.regex_replace(
                prompts,
                image_token,
                _get_image_prompt_string(
                    image_seq_len=self.image_seq_len,
                    image_rows=0,
                    image_cols=0,
                    fake_token_around_image=fake_token,
                    image_token=image_token,
                    global_image_token=global_token,
                ),
            )

        prompts = self.tokenizer(prompts)
        responses = self.tokenizer(responses)

        # Pad with one extra token for truncation below.
        token_ids, segment_ids = self._training_packer(
            (prompts, responses),
            sequence_length=sequence_length + 1,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        padding_mask = token_ids != self.tokenizer.pad_token_id
        response_mask = segment_ids == 1

        out = {
            "token_ids": token_ids[..., :-1],
            "padding_mask": padding_mask[..., :-1],
        }

        # Text-only batches omit the vision keys entirely;
        # `SmolVLM2Backbone.__call__` fills in empty placeholders, which keeps
        # the vision encoder from running on dummy pixels.
        if pixel_values is not None:
            out["pixel_values"] = pixel_values
            # `pixel_values` is a backend tensor, and `tf.shape()` would
            # convert it through `__array__`, which fails for torch tensors
            # on a non-CPU device. The static shape is known whenever the
            # converter ran eagerly; only a graph-mode batch needs `tf`.
            num_images = pixel_values.shape[0]
            if num_images is None:
                num_images = tf.shape(pixel_values)[0]
            out["vision_indices"] = self._graph_vision_indices(
                out["token_ids"], num_images
            )
        elif self.image_converter is not None:
            # Text-only batch for a vision model. Pass a zero-length image
            # batch (the Gemma3 pattern) so the vision encoder does no work,
            # while the output structure still matches the model inputs.
            crop_size = self.image_converter.max_image_size
            out["pixel_values"] = tf.zeros(
                (0, crop_size, crop_size, 3), dtype="float32"
            )
            out["vision_indices"] = tf.zeros(
                (tf.shape(token_ids)[0], 0), dtype="int32"
            )
        # With no image converter at all there is no way to know the image
        # size, so the vision keys are omitted entirely and
        # `SmolVLM2Backbone.__call__` fills in the placeholders.

        y = token_ids[..., 1:]
        sample_weight = response_mask[..., 1:]
        return keras.utils.pack_x_y_sample_weight(out, y, sample_weight)

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------
    @staticmethod
    def _split_per_sample(media, batch_size, sample_rank, name):
        """Split a batched media input into one entry per prompt.

        Args:
            media: A list of samples, a batched array of rank
                `sample_rank + 1`, or a single sample of rank `sample_rank`.
            batch_size: int. The number of prompts.
            sample_rank: int. Rank of a single sample (3 for images,
                4 for videos).
            name: str. Input name, used in error messages.
        Returns:
            list. One entry per prompt.
        """
        if isinstance(media, (list, tuple)):
            items = list(media)
        elif hasattr(media, "shape") and len(media.shape) == sample_rank + 1:
            items = [media[i] for i in range(media.shape[0])]
        else:
            items = [media]
        if len(items) != batch_size:
            raise ValueError(
                f"Received {len(items)} `{name}` for {batch_size} prompts. "
                f"Pass exactly one entry in `{name}` per prompt."
            )
        return items

    def _preprocess_image(self, image):
        """Process a single image through the converter.

        Args:
            image: A single 2-D (grayscale) or 3-D image.
        Returns:
            dict with ``pixel_values`` (N, H, W, 3), ``rows``, ``cols``.
            ``N`` is the number of crops, which is 1 unless image splitting
            is enabled.
        """
        if isinstance(image, np.ndarray) and image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        if self.image_converter is not None:
            result = self.image_converter(image)
            if isinstance(result, dict):
                pixel_values = result["pixel_values"]
                rows = result.get("rows", 0)
                cols = result.get("cols", 0)
            else:
                pixel_values = result
                rows = 0
                cols = 0
        else:
            pixel_values = np.array(image, dtype="float32")
            if pixel_values.ndim == 3:
                pixel_values = np.expand_dims(pixel_values, 0)
            rows = 0
            cols = 0

        # Ensure numpy array for downstream use.
        if not isinstance(pixel_values, np.ndarray):
            pixel_values = ops.convert_to_numpy(pixel_values)

        return {
            "pixel_values": pixel_values,
            "rows": int(rows),
            "cols": int(cols),
        }

    # ------------------------------------------------------------------
    # Video preprocessing
    # ------------------------------------------------------------------
    def _preprocess_video(self, video):
        """Process a single video through the video converter.

        Args:
            video: A single 4-D `(T, H, W, 3)` video.
        Returns:
            dict with ``pixel_values`` (num_frames, ms, ms, 3) and
            ``num_frames`` int.
        """
        if self.video_converter is not None:
            result = self.video_converter(video)
            pixel_values = result["pixel_values"]
            num_frames = int(result["num_frames"])
        else:
            # Fallback: treat each frame as an unsplit image.
            pixel_values = video
            num_frames = int(ops.shape(video)[0])

        if not isinstance(pixel_values, np.ndarray):
            pixel_values = ops.convert_to_numpy(pixel_values)

        return {"pixel_values": pixel_values, "num_frames": num_frames}

    def _get_video_prompt_string(
        self,
        num_frames,
        metadata=None,
    ):
        """Build the expanded video prompt string.

        Replicates HF's ``expand_text_with_video_tokens``.

        Each frame is wrapped with a timestamp and uses the same
        single-image prompt as an unsplit image.

        Args:
            num_frames: int. Number of frames.
            metadata: dict or None. If provided, should contain
                ``"fps"`` and optionally ``"duration"``.
        Returns:
            str. The expanded prompt fragment.
        """
        from datetime import timedelta

        image_token_str = getattr(self.tokenizer, "image_token", "<image>")
        fake_image_str = getattr(
            self.tokenizer,
            "fake_image_token",
            "<fake_token_around_image>",
        )
        global_image_str = getattr(
            self.tokenizer, "global_image_token", "<global-img>"
        )

        # Determine per-frame timestamps.
        fps = 1.0
        if metadata is not None and "fps" in metadata:
            fps = metadata["fps"]

        if metadata is not None and "frames_indices" in metadata and fps > 0:
            timestamps_secs = [idx / fps for idx in metadata["frames_indices"]]
        else:
            # Default: sequential frames at given fps.
            timestamps_secs = [i / fps for i in range(num_frames)]

        # Duration.
        if metadata is not None and "duration" in metadata:
            duration_secs = int(metadata["duration"])
        elif timestamps_secs:
            duration_secs = int(timestamps_secs[-1])
        else:
            duration_secs = 0

        duration_td = timedelta(seconds=duration_secs)

        # Build prompt.
        prompt = DEFAULT_VIDEO_INTRO.format(
            frame_count=str(num_frames),
            video_duration=str(duration_td),
        )

        for ts in timestamps_secs:
            minutes = int(ts) // 60
            seconds = int(ts) % 60
            timestamp_str = f"{minutes:02d}:{seconds:02d}"
            frame_prompt = _get_image_prompt_string(
                image_seq_len=self.image_seq_len,
                image_rows=0,
                image_cols=0,
                fake_token_around_image=fake_image_str,
                image_token=image_token_str,
                global_image_token=global_image_str,
            )
            prompt += (
                FRAME_TIMESTAMP_MESSAGE.format(timestamp=timestamp_str)
                + frame_prompt
            )

        prompt += DEFAULT_MEDIA_OUTTRO
        return prompt

    # ------------------------------------------------------------------
    # generate_preprocess
    # ------------------------------------------------------------------
    def generate_preprocess(
        self,
        x,
        sequence_length=None,
    ):
        """Convert inputs to integer token IDs for generation.

        For text-only inputs, returns only ``token_ids`` and
        ``padding_mask``.  For multimodal inputs, processes images
        through the image converter, expands ``<image>`` tokens using
        HF's prompt format, and returns ``pixel_values`` and
        ``vision_indices``.

        Multimodal prompts must contain ``<image>`` placeholder tokens.
        Each ``<image>`` is expanded to the full sub-image token
        sequence matching HuggingFace's ``SmolVLMProcessor``.

        Args:
            x: A string, a batch of strings, or a dict with a
                ``"prompts"`` key plus one of ``"images"``/``"videos"``
                (one entry per prompt) and an optional
                ``"video_metadata"``.
            sequence_length: int or None. Overrides `self.sequence_length`.
        """
        if not self.built:
            self.build(None)
        sequence_length = sequence_length or self.sequence_length

        # Handle both dict and string inputs.
        video_metadata = self.video_metadata
        if isinstance(x, dict):
            images = x.get("images", None)
            videos = x.get("videos", None)
            video_metadata = x.get("video_metadata", video_metadata)
            prompts = x["prompts"]
        else:
            images = None
            videos = None
            prompts = x

        # ------ Text-only path ------
        if images is None and videos is None:
            return super().generate_preprocess(
                prompts, sequence_length=sequence_length
            )
        if images is not None and videos is not None:
            raise ValueError(
                "Received both `images` and `videos`. Pass only one of them."
            )

        prompt_strings, _ = self._extract_prompt_strings(prompts)
        batch_size = len(prompt_strings)
        image_token_str, _, _ = self._image_token_strings()

        expanded_prompts = []
        pixel_values_per_sample = []

        # ------ Image path ------
        if images is not None:
            for prompt, image in zip(
                prompt_strings,
                self._split_per_sample(images, batch_size, 3, "images"),
            ):
                image_output = self._preprocess_image(image)
                pixel_values_per_sample.append(image_output["pixel_values"])
                image_prompt = self._expand_image_prompt(
                    image_output["rows"], image_output["cols"]
                )
                expanded_prompts.append(
                    prompt.replace(image_token_str, image_prompt, 1)
                )
        # ------ Video path ------
        else:
            # `<video>` is a chat-template marker, not a vocab token.
            video_token_str = "<video>"
            metadata_per_sample = self._split_video_metadata(
                video_metadata, batch_size
            )
            for prompt, video, metadata in zip(
                prompt_strings,
                self._split_per_sample(videos, batch_size, 4, "videos"),
                metadata_per_sample,
            ):
                video_output = self._preprocess_video(video)
                pixel_values_per_sample.append(video_output["pixel_values"])
                video_prompt = self._get_video_prompt_string(
                    num_frames=video_output["num_frames"],
                    metadata=metadata,
                )
                expanded_prompts.append(
                    prompt.replace(video_token_str, video_prompt, 1)
                )

        pixel_values = np.concatenate(pixel_values_per_sample, axis=0)

        # Tokenize with special token handling.
        special_map = self._build_special_token_map()
        token_id_lists = [
            self._tokenize_with_special_tokens(prompt, special_map)
            for prompt in expanded_prompts
        ]

        # Pack to fixed length using the training packer which
        # supports the tuple input + segment_ids return format.
        token_ids_tensor = tf.ragged.constant(token_id_lists, dtype="int32")
        token_ids, _ = self._training_packer(
            (token_ids_tensor,),
            sequence_length=sequence_length,
            add_start_value=self.add_start_token,
            add_end_value=False,
        )
        padding_mask = token_ids != self.tokenizer.pad_token_id

        # Compute vision_indices as flat `batch_index * seq_len + position`
        # offsets, which is what `SmolVLM2InterleaveEmbeddings` scatters on.
        vision_indices = self._compute_vision_indices(token_ids)
        expected = pixel_values.shape[0] * self.image_seq_len
        if vision_indices.size != expected:
            raise ValueError(
                f"Expected {expected} `<image>` tokens in the packed prompts "
                f"({pixel_values.shape[0]} sub-images x image_seq_len="
                f"{self.image_seq_len}), but found "
                f"{vision_indices.size}. Either the prompts are "
                "missing an `<image>` placeholder, or `sequence_length="
                f"{sequence_length}` is too short and truncated them."
            )
        # `vision_indices` is a dense `(batch, n)` tensor, so every prompt
        # has to expand to the same number of `<image>` slots. With image
        # splitting on, that means every image in the batch has to produce
        # the same number of crops.
        per_sample = vision_indices.size // batch_size
        if batch_size > 1:
            sample_ids = vision_indices // int(token_ids.shape[1])
            counts = np.bincount(sample_ids, minlength=batch_size)
            if not np.all(counts == per_sample):
                raise ValueError(
                    "Every prompt in a batch must expand to the same number "
                    "of `<image>` tokens, but the batch produced "
                    f"{counts.tolist()} tokens per prompt. This happens when "
                    "images of different sizes are split into different "
                    "numbers of crops. Either call `generate()` once per "
                    "image, or create the image converter with "
                    "`do_image_splitting=False`."
                )
        vision_indices = vision_indices.reshape((batch_size, per_sample))

        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
            "pixel_values": pixel_values,
            "vision_indices": vision_indices,
        }

    @staticmethod
    def _split_video_metadata(video_metadata, batch_size):
        """Return one metadata dict (or None) per prompt."""
        if video_metadata is None:
            return [None] * batch_size
        if isinstance(video_metadata, dict):
            return [video_metadata] * batch_size
        metadata = list(video_metadata)
        if len(metadata) != batch_size:
            raise ValueError(
                f"Received {len(metadata)} `video_metadata` entries for "
                f"{batch_size} prompts. Pass exactly one entry per prompt."
            )
        return metadata

    def _expand_image_prompt(self, rows, cols):
        """Expand a single `<image>` placeholder for a `rows x cols` split."""
        image_token, fake_token, global_token = self._image_token_strings()
        return _get_image_prompt_string(
            image_seq_len=self.image_seq_len,
            image_rows=rows,
            image_cols=cols,
            fake_token_around_image=fake_token,
            image_token=image_token,
            global_image_token=global_token,
        )

    def get_config(self):
        config = super().get_config()
        # `image_seq_len` is preset specific and is not derivable from the
        # tokenizer or the converters, so it has to be serialized here.
        config.update({"image_seq_len": self.image_seq_len})
        return config
