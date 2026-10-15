import re

import keras
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.embedding_gemma2.embedding_gemma2_audio_converter import (  # noqa: E501
    EmbeddingGemma2AudioConverter,
)
from keras_hub.src.models.embedding_gemma2.embedding_gemma2_backbone import (
    EmbeddingGemma2Backbone,
)
from keras_hub.src.models.embedding_gemma2.embedding_gemma2_image_converter import (  # noqa: E501
    EmbeddingGemma2ImageConverter,
)
from keras_hub.src.models.embedding_gemma2.embedding_gemma2_tokenizer import (
    EmbeddingGemma2Tokenizer,
)
from keras_hub.src.models.embedding_gemma2.embedding_gemma2_video_converter import (  # noqa: E501
    EmbeddingGemma2VideoConverter,
)
from keras_hub.src.models.text_embedder_preprocessor import (
    TextEmbedderPreprocessor,
)
from keras_hub.src.utils.tensor_utils import canonicalize_python_string_inputs
from keras_hub.src.utils.tensor_utils import (
    convert_preprocessing_outputs_python,
)
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.models.EmbeddingGemma2TextEmbedderPreprocessor")
class EmbeddingGemma2TextEmbedderPreprocessor(TextEmbedderPreprocessor):
    """EmbeddingGemma2 text embedder preprocessor.

    Turns raw text, images, videos and audio into the inputs of
    `keras_hub.models.EmbeddingGemma2Backbone`. Text is tokenized and packed
    to `sequence_length`; each image, video or audio clip is converted by the
    matching converter and its placeholder token in the text is expanded to
    the right number of soft tokens, with the positions of those tokens
    returned as `vision_indices` / `audio_indices`.

    Inputs are a `dict` with any of the keys `"texts"`, `"images"`,
    `"videos"`, `"audio"`, or a bare string / list of strings. A modality
    given without text gets a prompt made of its placeholder alone.
    Unbatched inputs (a single string, a rank-3 image, a rank-4 video, a
    rank-1 waveform) return unbatched outputs.

    Two limits apply: one image per prompt (a rank-3 image, or a rank-4
    batch with one image per example), and `"images"` and `"videos"` cannot
    be combined in a single call — pass them in separate calls. Both raise a
    `ValueError`.

    Python lists and NumPy arrays run through a pure-Python path that does
    not need TensorFlow; inside `tf.data` the TensorFlow path is used.

    Args:
        tokenizer: An `EmbeddingGemma2Tokenizer` instance.
        image_converter: An `EmbeddingGemma2ImageConverter`, or `None` to
            disable image inputs.
        audio_converter: An `EmbeddingGemma2AudioConverter`, or `None` to
            disable audio inputs.
        video_converter: An `EmbeddingGemma2VideoConverter`, or `None` to
            disable video inputs.
        sequence_length: int. Length every packed sequence is padded or
            truncated to. Image, video and audio soft tokens count against
            it (up to `max_soft_tokens` per image or video frame, and up to
            750 per 30 s audio clip), and a `ValueError` is raised when
            packing would truncate any of them. Defaults to `512`.
        truncate: string. The truncation algorithm of
            `keras_hub.layers.MultiSegmentPacker`. Defaults to
            `"round_robin"`.

    Examples:
    ```python
    preprocessor = (
        keras_hub.models.EmbeddingGemma2TextEmbedderPreprocessor.from_preset(
            "embedding_gemma2"
        )
    )
    preprocessor("What is the capital of France?")
    preprocessor({"texts": ["<|image|> describe this"], "images": [image]})
    ```
    """

    backbone_cls = EmbeddingGemma2Backbone
    tokenizer_cls = EmbeddingGemma2Tokenizer
    image_converter_cls = EmbeddingGemma2ImageConverter
    audio_converter_cls = EmbeddingGemma2AudioConverter
    video_converter_cls = EmbeddingGemma2VideoConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        audio_converter=None,
        video_converter=None,
        sequence_length=512,
        truncate="round_robin",
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            truncate=truncate,
            **kwargs,
        )
        self.image_converter = image_converter
        self.audio_converter = audio_converter
        self.video_converter = video_converter

        # Special tokens
        self.image_placeholder = "<|image|>"
        self.audio_placeholder = "<|audio|>"
        self.video_placeholder = "<|video|>"

        self.start_of_image_token = "<|image>"
        self.end_of_image_token = "<image|>"
        self.start_of_audio_token = "<|audio>"
        self.end_of_audio_token = "<audio|>"

    def _check_media_inputs(self, images, videos):
        """Reject input combinations this preprocessor does not support.

        Shared by the TF and Python paths so both raise the same error.
        """
        if images is not None and videos is not None:
            raise ValueError(
                "EmbeddingGemma2TextEmbedderPreprocessor does not support "
                "`images` and `videos` in the same call. Pass them in "
                "separate calls."
            )

    @property
    def _placeholders(self):
        """`(modality, placeholder)` pairs, in HF prompt order."""
        return (
            ("image", self.image_placeholder),
            ("video", self.video_placeholder),
            ("audio", self.audio_placeholder),
        )

    def _expected_placeholder_counts(self, texts, is_tf):
        """Occurrences of each placeholder in every (un-expanded) text.

        Returns a dict modality -> int32 vector of shape `(batch,)`.
        """
        counts = {}
        for mod, ph in self._placeholders:
            if is_tf:
                counts[mod] = tf.cast(
                    tf.strings.split(texts, sep=ph).row_lengths() - 1,
                    tf.int32,
                )
            else:
                counts[mod] = np.array(
                    [t.count(ph) for t in texts], dtype="int32"
                )
        return counts

    def _check_media_counts(self, expected, media_lens, is_tf):
        """HF-style `validate_inputs`, for one media item per prompt.

        For every modality that is passed, there must be one item per text
        and exactly one placeholder in each text; a placeholder for a
        modality that is not passed is an error. Python raises `ValueError`;
        the TF path asserts with the same messages.
        """
        for mod, _ in self._placeholders:
            exp = expected[mod]
            m_len = media_lens.get(mod)
            if m_len is not None:
                one_each = (
                    f"Each text must contain exactly 1 {mod} placeholder."
                )
                same_batch = f"Batch size of texts and {mod} must match."
                if is_tf:
                    tf.debugging.assert_equal(
                        exp, tf.ones_like(exp), message=one_each
                    )
                    tf.debugging.assert_equal(
                        tf.shape(exp)[0], m_len, message=same_batch
                    )
                else:
                    if np.any(exp != 1):
                        raise ValueError(one_each)
                    if len(exp) != m_len:
                        raise ValueError(same_batch)
            else:
                msg = f"Found {mod} placeholder but no media was provided."
                if is_tf:
                    tf.debugging.assert_equal(
                        exp, tf.zeros_like(exp), message=msg
                    )
                elif np.any(exp > 0):
                    raise ValueError(msg)

    def _build_truncation_error_msg(self, modality, seq_len):
        return (
            f"The {modality} soft tokens do not fit in "
            f"`sequence_length={seq_len}`: media soft tokens count against "
            "`sequence_length`, and packing truncated some of them. Increase "
            "`sequence_length` or shorten the text / media."
        )

    def _check_truncation(self, expected, token_ids, sequence_length, is_tf):
        """Every soft token of the expanded prompt must survive packing.

        `expected` is `_expected_placeholder_counts` of the expanded texts,
        i.e. the soft-token count per modality; fewer of them in the packed
        `token_ids` means the packer truncated media tokens.
        """
        for mod, ph in self._placeholders:
            exp = expected[mod]
            msg = self._build_truncation_error_msg(mod, sequence_length)
            placeholder_id = self.tokenizer.token_to_id(ph)
            if is_tf:
                kept = tf.math.reduce_sum(
                    tf.cast(token_ids == placeholder_id, tf.int32), axis=1
                )
                tf.debugging.assert_greater_equal(kept, exp, message=msg)
            else:
                kept = (token_ids == placeholder_id).sum(axis=1)
                if np.any(kept < exp):
                    raise ValueError(msg)

    def _build_default_texts(self, has_images, has_videos, has_audio):
        """Prompt used when `texts` is `None`.

        Matches the Hugging Face processor: one placeholder per present
        modality, in the order images, videos, audio, joined by a space.
        """
        prompts = []
        if has_images:
            prompts.append(self.image_placeholder)
        if has_videos:
            prompts.append(self.video_placeholder)
        if has_audio:
            prompts.append(self.audio_placeholder)
        return " ".join(prompts)

    # Rank of one un-batched item per modality: an image, a clip, a waveform.
    _SINGLE_ITEM_RANK = {"image": 3, "video": 4, "audio": 1}

    def _media_batch_sizes(self, media, is_tf):
        """Number of items per modality, and whether any is a bare item.

        Args:
            media: dict modality -> list / array / tensor or `None`.
            is_tf: whether the values are TF tensors.

        Returns:
            `(media_lens, any_single)`. `media_lens` maps each present
            modality to its item count (a Python int, or a scalar tensor on
            the TF path when the batch axis is symbolic). `any_single` is
            `True` when a modality was given as one bare item (rank-3 image,
            rank-4 video, rank-1 waveform), which the caller wraps into a
            one-item batch.
        """
        media_lens = {}
        any_single = False
        for mod, value in media.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                media_lens[mod] = len(value)
                continue
            if not is_tf:
                value = np.asarray(value)
            # Works for numpy arrays, `tf.Tensor` and `tf.RaggedTensor`.
            rank = len(value.shape)
            if rank == self._SINGLE_ITEM_RANK[mod]:
                media_lens[mod] = 1
                any_single = True
            elif is_tf:
                media_lens[mod] = tf.shape(value)[0]
            else:
                media_lens[mod] = value.shape[0]
        return media_lens, any_single

    def _check_same_batch_size(self, media_lens, is_tf):
        """All present modalities must have the same number of items."""
        sizes = list(media_lens.values())
        msg = (
            "Received inconsistently sized modality batches when `texts` is "
            f"None: {sorted(media_lens)} must all have the same length."
        )
        for other in sizes[1:]:
            if is_tf and not (
                isinstance(sizes[0], int) and isinstance(other, int)
            ):
                tf.debugging.assert_equal(sizes[0], other, message=msg)
            elif sizes[0] != other:
                raise ValueError(msg)
        return sizes[0]

    def _frame_layout(self, frame_counts_per_row):
        """The one layout rule for `vision_indices` / `audio_indices`.

        Every row gets `num_frames_max * width` columns: the most frames any
        row has, times the most soft tokens any single frame has. Frame `f`
        of a row occupies columns `[f * width, (f + 1) * width)`, real
        positions first, then `0`. `Gemma4InterleaveEmbeddings` keeps the
        first `width` tower slots of every frame and pairs them with these
        columns in order, so the pairing is per frame and never depends on
        the other rows in the batch. `_python_indices` applies this rule
        directly; `_get_placeholder_indices` reproduces it with ragged tensors.

        Args:
            frame_counts_per_row: list (rows) of lists (frames) of soft-token
                counts.

        Returns:
            `(num_frames_max, width)`.
        """
        num_frames_max = max(
            (len(row) for row in frame_counts_per_row), default=0
        )
        width = max(
            (max(row) if row else 0 for row in frame_counts_per_row), default=0
        )
        return num_frames_max, width

    def _check_image_rank(self, rank, is_list_item=False):
        """Enforce one image per prompt.

        A rank-3 array is one image; a rank-4 array is a batch with one image
        per example. Anything else (e.g. `(batch, n_images, h, w, c)`) is not
        supported.
        """
        allowed = (3,) if is_list_item else (3, 4)
        if rank not in allowed:
            raise ValueError(
                "EmbeddingGemma2TextEmbedderPreprocessor accepts one image "
                "per prompt: a rank-3 image or a rank-4 batch of images. "
                f"Received an image input of rank {rank}."
            )

    def _get_placeholder_indices(self, mask):
        """Frame-major placeholder positions, TF twin of `_python_indices`.

        Used for images, video frames and audio clips alike: a contiguous
        run of placeholders is one frame (one image, one video frame or one
        audio clip).

        Each row is laid out as `num_frames_max` blocks of `width` columns,
        where `width` is the largest soft-token count of any frame in the
        batch (see `_frame_layout`). Block `f` of a row holds that frame's
        placeholder positions first and `0` after, so that slot `i` of frame
        `f` from the vision tower always pairs with column `f * width + i`
        in `Gemma4InterleaveEmbeddings`, whatever the other rows contain.
        Column `0` is harmless: the interleave layer restores position 0.

        Frames are separated by the `<image|><|image>` wrapper tokens, so a
        gap larger than 1 between consecutive placeholder positions marks a
        frame boundary. A row boundary cannot produce a gap of exactly 1,
        because position 0 of every row is the start token.
        """
        batch_size = tf.shape(mask)[0]
        sequence_length = tf.shape(mask)[1]

        mask_flattened = tf.reshape(mask, [-1])
        indices = tf.where(mask_flattened)[..., 0]
        indices = tf.cast(indices, dtype=tf.int32)

        def _compute():
            diffs = indices[1:] - indices[:-1]
            is_new_frame = tf.concat([[True], diffs > 1], axis=0)
            frame_ids = tf.cumsum(tf.cast(is_new_frame, tf.int32)) - 1
            frame_lengths = tf.math.segment_sum(
                tf.ones_like(indices), frame_ids
            )

            row_indices = indices // sequence_length
            frame_starts = tf.where(is_new_frame)[..., 0]
            frame_rows = tf.gather(row_indices, frame_starts)
            frames_per_row = tf.math.bincount(
                frame_rows, minlength=batch_size, maxlength=batch_size
            )
            local_indices = indices - row_indices * sequence_length

            # (batch, frames, width) with zeros where a frame is narrower
            # than `width` or a row has fewer frames than `num_frames_max`.
            frames = tf.RaggedTensor.from_row_lengths(
                values=local_indices, row_lengths=frame_lengths
            )
            rows = tf.RaggedTensor.from_row_lengths(
                values=frames, row_lengths=frames_per_row
            )
            padded = rows.to_tensor(default_value=0)
            num_frames_max = tf.shape(padded)[1]
            width = tf.shape(padded)[2]
            return tf.reshape(padded, [batch_size, num_frames_max * width])

        return tf.cond(
            tf.math.equal(tf.shape(indices)[0], 0),
            lambda: tf.zeros([batch_size, 0], dtype=tf.int32),
            _compute,
        )

    @preprocessing_function
    def _call_tf(self, x, y=None, sample_weight=None, sequence_length=None):
        sequence_length = sequence_length or self.sequence_length

        def _to_dense_tensor(t):
            if isinstance(t, tf.RaggedTensor):
                t = t.to_tensor()
            return tf.convert_to_tensor(t)

        if isinstance(x, dict):
            texts = x.get("texts", None)
            images = x.get("images", None)
            videos = x.get("videos", None)
            audio = x.get("audio", None)
        else:
            texts = x
            images = videos = audio = None
        self._check_media_inputs(images, videos)

        batched = True
        if isinstance(texts, str):
            batched = False
            texts = [texts]
        elif isinstance(texts, tf.Tensor) and len(texts.shape) == 0:
            batched = False
            texts = tf.expand_dims(texts, axis=0)

        media_lens, any_single = self._media_batch_sizes(
            {"image": images, "video": videos, "audio": audio}, is_tf=True
        )
        if texts is None:
            # Media only: one placeholder prompt per example, built from
            # every present modality. A single rank-3 image, rank-4 video or
            # rank-1 waveform is one example, exactly as in `_call_python`.
            batched = not any_single
            if media_lens:
                b = self._check_same_batch_size(media_lens, is_tf=True)
                default_prompt = self._build_default_texts(
                    images is not None, videos is not None, audio is not None
                )
                texts = tf.repeat(tf.constant(default_prompt), b)
            else:
                texts = tf.constant([""])
            expected_counts = self._expected_placeholder_counts(texts, True)
        else:
            expected_counts = self._expected_placeholder_counts(texts, True)
            self._check_media_counts(expected_counts, media_lens, True)

        # Vision handling
        pixel_values = None
        pixel_position_ids = None

        if self.image_converter is not None and images is not None:
            pooling = self.image_converter.pooling_kernel_size**2

            def _convert_image(img):
                # One image -> (patches, patch_dim), (patches, 2), soft tokens.
                img = tf.expand_dims(_to_dense_tensor(img), 0)
                image_outputs = self.image_converter(img)
                pv = image_outputs["pixel_values"][0]
                ppi = image_outputs["pixel_position_ids"][0]
                valid_patches = tf.cast(ppi[:, 0] != -1, tf.int32)
                num_soft_tokens = tf.reduce_sum(valid_patches) // pooling
                return pv, ppi, num_soft_tokens

            if isinstance(images, (list, tuple)):
                # A Python list only exists in eager mode; a plain loop is
                # fine and keeps per-image shapes free.
                for img in images:
                    self._check_image_rank(
                        tf.convert_to_tensor(img).shape.rank
                        if not isinstance(img, tf.RaggedTensor)
                        else img.shape.rank,
                        is_list_item=True,
                    )
                converted = [_convert_image(img) for img in images]
                pixel_values = tf.stack([c[0] for c in converted], axis=0)
                pixel_position_ids = tf.stack([c[1] for c in converted], axis=0)
                num_tokens_tensor = tf.stack([c[2] for c in converted])
            else:
                if not isinstance(images, tf.RaggedTensor):
                    images = tf.convert_to_tensor(images)
                self._check_image_rank(images.shape.rank)
                if images.shape.rank == 3:
                    images = tf.expand_dims(images, 0)
                # `tf.map_fn` iterates the batch axis symbolically, so this
                # traces inside `tf.data.Dataset.map` with an unknown batch.
                pixel_values, pixel_position_ids, num_tokens_tensor = tf.map_fn(
                    _convert_image,
                    images,
                    fn_output_signature=(
                        tf.TensorSpec([None, None], tf.float32),
                        tf.TensorSpec([None, 2], tf.int32),
                        tf.TensorSpec([], tf.int32),
                    ),
                )

            pixel_values = tf.expand_dims(pixel_values, 1)
            pixel_position_ids = tf.expand_dims(pixel_position_ids, 1)

            def _replace_image(args):
                text, num_tokens = args
                replacement = tf.strings.join(
                    [
                        self.start_of_image_token,
                        tf.strings.reduce_join(
                            tf.repeat(self.image_placeholder, num_tokens),
                            axis=0,
                        ),
                        self.end_of_image_token,
                    ]
                )
                return tf.strings.regex_replace(
                    text, re.escape(self.image_placeholder), replacement
                )

            texts = tf.map_fn(
                _replace_image,
                (texts, num_tokens_tensor),
                fn_output_signature=tf.string,
            )

        if self.video_converter is not None and videos is not None:
            pooling = (
                self.video_converter.image_converter.pooling_kernel_size**2
            )
            patch_dim = self.video_converter.patch_size**2 * 3

            def _convert_video(vid):
                # One clip -> (frames, patches, patch_dim), (frames, patches,
                # 2) and the per-frame soft-token counts, all ragged in
                # `frames`.
                vid = tf.expand_dims(_to_dense_tensor(vid), 0)
                video_outputs = self.video_converter(vid)
                pv = video_outputs["pixel_values"][0]
                ppi = video_outputs["pixel_position_ids"][0]
                valid_patches = tf.cast(ppi[:, :, 0] != -1, tf.int32)
                num_soft_tokens = (
                    tf.reduce_sum(valid_patches, axis=1) // pooling
                )
                return (
                    tf.RaggedTensor.from_tensor(pv),
                    tf.RaggedTensor.from_tensor(ppi),
                    num_soft_tokens,
                )

            ragged_spec = lambda inner, dtype: tf.RaggedTensorSpec(  # noqa: E731
                shape=[None] + inner, dtype=dtype, ragged_rank=1
            )
            if isinstance(videos, (list, tuple)):
                converted = [_convert_video(vid) for vid in videos]
                pv_ragged = tf.ragged.stack([c[0] for c in converted])
                ppi_ragged = tf.ragged.stack([c[1] for c in converted])
                ragged_tokens = tf.ragged.stack([c[2] for c in converted])
            else:
                if not isinstance(videos, tf.RaggedTensor):
                    videos = tf.convert_to_tensor(videos)
                if videos.shape.rank == 4:
                    videos = tf.expand_dims(videos, 0)
                pv_ragged, ppi_ragged, ragged_tokens = tf.map_fn(
                    _convert_video,
                    videos,
                    fn_output_signature=(
                        ragged_spec([None, patch_dim], tf.float32),
                        ragged_spec([None, 2], tf.int32),
                        tf.RaggedTensorSpec(
                            shape=[None], dtype=tf.int32, ragged_rank=0
                        ),
                    ),
                )

            # Pad clips to the longest one in the batch: 0.0 for pixels, -1
            # for position ids (a padding patch), exactly as before.
            pixel_values = pv_ragged.to_tensor(default_value=0.0)
            pixel_position_ids = ppi_ragged.to_tensor(default_value=-1)

            def _replace_video(args):
                text, tokens = args

                def make_frame_block(t):
                    return tf.strings.join(
                        [
                            self.start_of_image_token,
                            tf.strings.reduce_join(
                                tf.repeat(self.video_placeholder, t), axis=0
                            ),
                            self.end_of_image_token,
                        ]
                    )

                blocks = tf.map_fn(
                    make_frame_block, tokens, fn_output_signature=tf.string
                )
                replacement = tf.strings.reduce_join(blocks, axis=0)
                return tf.strings.regex_replace(
                    text, re.escape(self.video_placeholder), replacement
                )

            texts = tf.map_fn(
                _replace_video,
                (texts, ragged_tokens),
                fn_output_signature=tf.string,
            )

        audio_mel = None
        audio_mel_mask = None

        if self.audio_converter is not None and audio is not None:
            stride = self.audio_converter.stride
            num_mels = self.audio_converter.num_mels

            def _convert_audio(aud):
                # One clip -> mel frames (ragged), its frame count and its
                # soft-token count. Each clip is converted on its own, never
                # zero-padded to the batch, so the mel frames match HF.
                aud = _to_dense_tensor(aud)
                len_i = tf.minimum(
                    tf.shape(aud)[0], self.audio_converter.num_samples
                )
                mel = tf.convert_to_tensor(
                    self.audio_converter(tf.expand_dims(aud, 0))
                )[0]
                # HF derivation: left-pad 160, unfold 321, hop 160
                # -> (len_i - 1) // hop frames.
                output_length = tf.maximum((len_i - 1) // stride, 0)
                output_length = tf.minimum(output_length, tf.shape(mel)[0])
                num_audio_tokens = tf.maximum((output_length + 3) // 4, 1)
                return (
                    tf.RaggedTensor.from_tensor(mel[:output_length]),
                    output_length,
                    num_audio_tokens,
                )

            if isinstance(audio, (list, tuple)):
                converted = [_convert_audio(aud) for aud in audio]
                mel_ragged = tf.ragged.stack([c[0] for c in converted])
                output_lengths = tf.stack([c[1] for c in converted])
                num_tokens_tensor = tf.stack([c[2] for c in converted])
            else:
                if not isinstance(audio, tf.RaggedTensor):
                    audio = tf.convert_to_tensor(audio)
                if audio.shape.rank == 1:
                    audio = tf.expand_dims(audio, 0)
                mel_ragged, output_lengths, num_tokens_tensor = tf.map_fn(
                    _convert_audio,
                    audio,
                    fn_output_signature=(
                        tf.RaggedTensorSpec(
                            shape=[None, num_mels],
                            dtype=tf.float32,
                            ragged_rank=1,
                        ),
                        tf.TensorSpec([], tf.int32),
                        tf.TensorSpec([], tf.int32),
                    ),
                )

            # Pad clips to the longest one in the batch (zeros), with a mask
            # of ones over the real frames, exactly as before.
            audio_mel = mel_ragged.to_tensor(default_value=0.0)
            audio_mel_mask = tf.sequence_mask(
                output_lengths, maxlen=tf.shape(audio_mel)[1], dtype=tf.int32
            )
            audio_mel = tf.expand_dims(audio_mel, 1)
            audio_mel_mask = tf.expand_dims(audio_mel_mask, 1)

            def _replace_audio(args):
                text, num_tokens = args
                replacement = tf.strings.join(
                    [
                        self.start_of_audio_token,
                        tf.strings.reduce_join(
                            tf.repeat(self.audio_placeholder, num_tokens),
                            axis=0,
                        ),
                        self.end_of_audio_token,
                    ]
                )
                return tf.strings.regex_replace(
                    text, re.escape(self.audio_placeholder), replacement
                )

            texts = tf.map_fn(
                _replace_audio,
                (texts, num_tokens_tensor),
                fn_output_signature=tf.string,
            )

        # Soft-token counts of the expanded prompts; packing must keep all.
        expanded_counts = self._expected_placeholder_counts(texts, True)
        token_ids = self.tokenizer(texts)
        token_ids, segment_ids = self.packer(
            token_ids,
            sequence_length=sequence_length,
            add_start_value=True,
            add_end_value=True,
        )
        padding_mask = token_ids != self.tokenizer.pad_token_id
        self._check_truncation(
            expanded_counts, token_ids, sequence_length, True
        )

        batch_size = tf.shape(token_ids)[0]

        vision_indices = None

        if pixel_values is not None:
            vision_mask = token_ids == self.tokenizer.token_to_id(
                self.image_placeholder
            )
            if self.video_converter is not None and videos is not None:
                vision_mask = tf.logical_or(
                    vision_mask,
                    token_ids
                    == self.tokenizer.token_to_id(self.video_placeholder),
                )
            vision_indices = self._get_placeholder_indices(vision_mask)
            vision_indices = tf.cast(vision_indices, "int32")
        elif (
            self.image_converter is not None or self.video_converter is not None
        ):
            if self.image_converter is not None:
                patch_dim = self.image_converter.patch_size**2 * 3
            else:
                patch_dim = self.video_converter.patch_size**2 * 3
            pixel_values = tf.ones(
                [batch_size, 0, 0, patch_dim], dtype="float32"
            )
            pixel_position_ids = tf.zeros([batch_size, 0, 0, 2], dtype="int32")
            vision_indices = tf.zeros([batch_size, 0], dtype="int32")

        audio_indices = None

        if audio_mel is not None:
            audio_mask = token_ids == self.tokenizer.token_to_id(
                self.audio_placeholder
            )
            audio_indices = self._get_placeholder_indices(audio_mask)
            audio_indices = tf.cast(audio_indices, "int32")
        elif self.audio_converter is not None:
            input_feat_size = self.audio_converter.num_mels
            audio_mel = tf.zeros(
                [batch_size, 0, 1, input_feat_size], dtype="float32"
            )
            audio_mel_mask = tf.zeros([batch_size, 0, 1], dtype="int32")
            audio_indices = tf.zeros([batch_size, 0], dtype="int32")

        output = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

        if self.image_converter is not None or self.video_converter is not None:
            output["pixel_values"] = pixel_values
            output["pixel_position_ids"] = pixel_position_ids
            output["vision_indices"] = vision_indices

        if self.audio_converter is not None:
            output["audio_mel"] = audio_mel
            output["audio_mel_mask"] = tf.cast(audio_mel_mask, "int32")
            output["audio_indices"] = audio_indices

        if not batched:
            output = keras.tree.map_structure(
                lambda x: tf.squeeze(x, axis=0), output
            )

        return keras.utils.pack_x_y_sample_weight(output, y, sample_weight)

    def _python_indices(self, token_ids, placeholder_ids):
        """Numpy twin of `_get_placeholder_indices`.

        Args:
            token_ids: int numpy array of shape `(batch, seq)`.
            placeholder_ids: the token ids that mark a soft-token slot.

        Returns:
            int32 `(batch, num_frames_max * width)` positions laid out
            frame-major: block `f` of each row holds frame `f`'s placeholder
            positions first and `0` after, with `(num_frames_max, width)`
            from `_frame_layout`. Contiguous placeholder runs are frames (the
            `<image|><|image>` wrappers separate them), so a single image or
            audio clip is one frame. Identical to the TF path.
        """
        mask = np.isin(token_ids, list(placeholder_ids))
        batch_size = mask.shape[0]

        # Per row: the length of every contiguous run of placeholders.
        frame_counts = []
        for row in mask:
            padded = np.concatenate([[False], row, [False]])
            edges = np.flatnonzero(padded[1:] != padded[:-1])
            frame_counts.append((edges[1::2] - edges[::2]).tolist())

        num_frames_max, width = self._frame_layout(frame_counts)
        indices = np.zeros((batch_size, num_frames_max * width), dtype="int32")
        for r, row in enumerate(mask):
            positions = np.flatnonzero(row)
            start = 0
            for f, count in enumerate(frame_counts[r]):
                indices[r, f * width : f * width + count] = positions[
                    start : start + count
                ]
                start += count
        return indices

    def _call_python(self, x, y=None, sample_weight=None, sequence_length=None):
        sequence_length = sequence_length or self.sequence_length

        if isinstance(x, dict):
            texts = x.get("texts", None)
            images = x.get("images", None)
            videos = x.get("videos", None)
            audio = x.get("audio", None)
        else:
            texts = x
            images = videos = audio = None
        self._check_media_inputs(images, videos)

        def _to_host(value):
            # Everything below is host-side numpy. A backend tensor handed in
            # by the caller (on the torch backend with a GPU, a CUDA tensor
            # that `np.asarray` cannot read) is copied to host once here.
            if tf is not None and isinstance(value, tf.RaggedTensor):
                return [r.numpy() for r in value]
            if isinstance(value, (list, tuple)):
                return [_to_host(v) for v in value]
            if keras.ops.is_tensor(value):
                return keras.ops.convert_to_numpy(value)
            return value

        images = _to_host(images)
        videos = _to_host(videos)
        audio = _to_host(audio)

        media_lens, any_single = self._media_batch_sizes(
            {"image": images, "video": videos, "audio": audio}, is_tf=False
        )
        if texts is not None:
            texts, is_batched, outer_shape = canonicalize_python_string_inputs(
                texts
            )
            if outer_shape is not None:
                raise ValueError(
                    "EmbeddingGemma2TextEmbedderPreprocessor does not "
                    "support rank>=2 text inputs."
                )
            expected_counts = self._expected_placeholder_counts(texts, False)
            self._check_media_counts(expected_counts, media_lens, False)
        else:
            # Media only: one placeholder prompt per example, built from
            # every present modality. A single rank-3 image, rank-4 video or
            # rank-1 waveform is one example.
            is_batched = not any_single
            if media_lens:
                b = self._check_same_batch_size(media_lens, is_tf=False)
                default_prompt = self._build_default_texts(
                    images is not None, videos is not None, audio is not None
                )
                texts = [default_prompt] * b
            else:
                texts = [""]
            expected_counts = self._expected_placeholder_counts(texts, False)

        # Vision handling
        pixel_values = None
        pixel_position_ids = None

        if self.image_converter is not None and images is not None:
            if isinstance(images, (list, tuple)):
                images_list = [np.asarray(e) for e in images]
                for img in images_list:
                    self._check_image_rank(img.ndim, is_list_item=True)
            else:
                images = np.asarray(images)
                self._check_image_rank(images.ndim)
                if images.ndim == 3:
                    images_list = [images]
                else:
                    images_list = [images[i] for i in range(images.shape[0])]

            pv_list, ppi_list, tokens_list = [], [], []

            for img in images_list:
                img_arr = np.array(img)
                img_arr = np.expand_dims(img_arr, axis=0)
                image_outputs = self.image_converter(img_arr)
                pv = keras.ops.convert_to_numpy(image_outputs["pixel_values"])
                ppi = keras.ops.convert_to_numpy(
                    image_outputs["pixel_position_ids"]
                )
                valid_patches = ppi[0, :, 0] != -1
                num_patches = np.sum(valid_patches)
                num_soft_tokens = num_patches // (
                    self.image_converter.pooling_kernel_size**2
                )

                pv_list.append(pv[0])
                ppi_list.append(ppi[0])
                tokens_list.append(num_soft_tokens)

            pixel_values = np.stack(pv_list, axis=0)
            pixel_position_ids = np.stack(ppi_list, axis=0)
            pixel_values = np.expand_dims(pixel_values, axis=1)
            pixel_position_ids = np.expand_dims(pixel_position_ids, axis=1)

            for i in range(len(texts)):
                replacement = (
                    self.start_of_image_token
                    + self.image_placeholder * int(tokens_list[i])
                    + self.end_of_image_token
                )
                texts[i] = texts[i].replace(self.image_placeholder, replacement)

        if self.video_converter is not None and videos is not None:
            if isinstance(videos, (list, tuple)):
                videos_list = [np.asarray(e) for e in videos]
            elif isinstance(videos, np.ndarray):
                if len(videos.shape) == 4:
                    videos_list = [videos]
                else:
                    videos_list = [videos[i] for i in range(videos.shape[0])]
            else:
                videos_list = [np.asarray(videos)]

            pv_list, ppi_list, tokens_list = [], [], []

            for vid in videos_list:
                vid_arr = np.array(vid)
                vid_arr = np.expand_dims(vid_arr, axis=0)
                video_outputs = self.video_converter(vid_arr)
                pv = keras.ops.convert_to_numpy(
                    video_outputs["pixel_values"][0]
                )
                ppi = keras.ops.convert_to_numpy(
                    video_outputs["pixel_position_ids"][0]
                )

                valid_patches = ppi[:, :, 0] != -1
                num_patches = np.sum(valid_patches, axis=1)
                num_soft_tokens = num_patches // (
                    self.video_converter.image_converter.pooling_kernel_size**2
                )

                pv_list.append(pv)
                ppi_list.append(ppi)
                tokens_list.append(num_soft_tokens.tolist())

            max_frames = (
                max([ppi.shape[0] for ppi in ppi_list]) if ppi_list else 0
            )
            max_patches = (
                max([ppi.shape[1] for ppi in ppi_list]) if ppi_list else 0
            )

            padded_pv_list = []
            padded_ppi_list = []

            for pv, ppi in zip(pv_list, ppi_list):
                pad_frames = max_frames - pv.shape[0]
                pad_patches = max_patches - pv.shape[1]
                padded_pv = np.pad(
                    pv,
                    ((0, pad_frames), (0, pad_patches), (0, 0)),
                    constant_values=0.0,
                )
                padded_ppi = np.pad(
                    ppi,
                    ((0, pad_frames), (0, pad_patches), (0, 0)),
                    constant_values=-1,
                )
                padded_pv_list.append(padded_pv)
                padded_ppi_list.append(padded_ppi)

            pixel_values = np.stack(padded_pv_list, axis=0)
            pixel_position_ids = np.stack(padded_ppi_list, axis=0)

            for i in range(len(texts)):
                tokens = tokens_list[i]
                blocks = []
                for t in tokens:
                    blocks.append(
                        self.start_of_image_token
                        + self.video_placeholder * int(t)
                        + self.end_of_image_token
                    )
                replacement = "".join(blocks)
                texts[i] = texts[i].replace(self.video_placeholder, replacement)

        audio_mel = None
        audio_mel_mask = None

        if self.audio_converter is not None and audio is not None:
            if isinstance(audio, (list, tuple)):
                audio_list = [np.asarray(e) for e in audio]
            elif isinstance(audio, np.ndarray):
                if len(audio.shape) == 1:
                    audio_list = [audio]
                else:
                    audio_list = [audio[i] for i in range(audio.shape[0])]
            else:
                audio_list = [np.asarray(audio)]

            mel_list = []
            tokens_list = []
            output_lengths_list = []
            max_len = 0

            for aud in audio_list:
                aud_arr = np.array(aud)
                len_i = min(aud_arr.shape[0], self.audio_converter.num_samples)
                aud_arr = np.expand_dims(aud_arr, axis=0)
                mel = self.audio_converter(aud_arr)
                mel = keras.ops.convert_to_numpy(mel)[0]

                output_lengths_i = max(
                    (len_i - 1) // self.audio_converter.stride, 0
                )
                output_lengths_i = min(output_lengths_i, mel.shape[0])
                num_audio_tokens = max((output_lengths_i + 3) // 4, 1)

                mel_list.append(mel[:output_lengths_i])
                output_lengths_list.append(output_lengths_i)
                tokens_list.append(num_audio_tokens)
                max_len = max(max_len, output_lengths_i)

            padded_mel_list = []
            mask_list = []
            for mel, out_len in zip(mel_list, output_lengths_list):
                pad_l = max_len - out_len
                padded_mel = np.pad(
                    mel, ((0, pad_l), (0, 0)), constant_values=0.0
                )
                padded_mel_list.append(padded_mel)
                mask_arr = np.concatenate(
                    [
                        np.ones(out_len, dtype="int32"),
                        np.zeros(pad_l, dtype="int32"),
                    ],
                    axis=0,
                )
                mask_list.append(mask_arr)

            audio_mel = np.stack(padded_mel_list, axis=0)
            audio_mel = np.expand_dims(audio_mel, axis=1)
            audio_mel_mask = np.stack(mask_list, axis=0)
            audio_mel_mask = np.expand_dims(audio_mel_mask, axis=1)

            for i in range(len(texts)):
                replacement = (
                    self.start_of_audio_token
                    + self.audio_placeholder * int(tokens_list[i])
                    + self.end_of_audio_token
                )
                texts[i] = texts[i].replace(self.audio_placeholder, replacement)

        # Soft-token counts of the expanded prompts; packing must keep all.
        expanded_counts = self._expected_placeholder_counts(texts, False)
        token_ids = self.tokenizer(texts)
        token_ids, _ = self.packer(
            token_ids,
            sequence_length=sequence_length,
            add_start_value=True,
            add_end_value=True,
        )
        # The packer returns a backend tensor (on a GPU box, a device
        # tensor); everything below is host-side numpy.
        token_ids = keras.ops.convert_to_numpy(token_ids)
        padding_mask = token_ids != self.tokenizer.pad_token_id
        self._check_truncation(
            expanded_counts, token_ids, sequence_length, False
        )

        batch_size = token_ids.shape[0]

        vision_indices = None

        if pixel_values is not None:
            img_tok = self.tokenizer.token_to_id(self.image_placeholder)
            vid_tok = self.tokenizer.token_to_id(self.video_placeholder)
            vision_indices = self._python_indices(token_ids, [img_tok, vid_tok])
        elif (
            self.image_converter is not None or self.video_converter is not None
        ):
            if self.image_converter is not None:
                patch_dim = self.image_converter.patch_size**2 * 3
            else:
                patch_dim = self.video_converter.patch_size**2 * 3
            pixel_values = np.zeros(
                (batch_size, 0, 0, patch_dim), dtype="float32"
            )
            pixel_position_ids = np.zeros((batch_size, 0, 0, 2), dtype="int32")
            vision_indices = np.zeros((batch_size, 0), dtype="int32")

        audio_indices = None

        if audio_mel is not None:
            aud_tok = self.tokenizer.token_to_id(self.audio_placeholder)
            audio_indices = self._python_indices(token_ids, [aud_tok])
        elif self.audio_converter is not None:
            input_feat_size = self.audio_converter.num_mels
            audio_mel = np.zeros(
                (batch_size, 0, 1, input_feat_size), dtype="float32"
            )
            audio_mel_mask = np.zeros((batch_size, 0, 1), dtype="int32")
            audio_indices = np.zeros((batch_size, 0), dtype="int32")

        output = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

        if self.image_converter is not None or self.video_converter is not None:
            output["pixel_values"] = pixel_values
            output["pixel_position_ids"] = pixel_position_ids
            output["vision_indices"] = vision_indices

        if self.audio_converter is not None:
            output["audio_mel"] = audio_mel
            output["audio_mel_mask"] = np.array(audio_mel_mask, dtype="int32")
            output["audio_indices"] = audio_indices

        output = convert_preprocessing_outputs_python(output)
        if not is_batched:
            output = keras.tree.map_structure(lambda t: t[0], output)

        return keras.utils.pack_x_y_sample_weight(output, y, sample_weight)

    def call(self, x, y=None, sample_weight=None, sequence_length=None):
        if self._use_tf_workflow():
            return self._call_tf(
                x,
                y=y,
                sample_weight=sample_weight,
                sequence_length=sequence_length,
            )
        else:
            return self._call_python(
                x,
                y=y,
                sample_weight=sample_weight,
                sequence_length=sequence_length,
            )
