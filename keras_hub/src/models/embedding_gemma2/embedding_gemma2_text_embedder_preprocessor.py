import re

import keras
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
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
            truncated to. Defaults to `512`.
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

    def build(self, input_shape):
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            truncate=self.truncate,
            sequence_length=self.sequence_length,
        )
        self.built = True

    def _get_vision_indices(self, vision_mask, max_tokens=None):
        batch_size, sequence_length = vision_mask.shape

        vision_mask_flattened = tf.reshape(vision_mask, [-1])
        vision_indices = tf.where(vision_mask_flattened)[..., 0]
        vision_indices = tf.cast(vision_indices, dtype=tf.int32)

        row_lengths = tf.math.reduce_sum(
            tf.cast(vision_mask, dtype=vision_indices.dtype), axis=1
        )

        batched_vision_indices = tf.RaggedTensor.from_row_lengths(
            values=vision_indices,
            row_lengths=row_lengths,
        )

        to_subtract = tf.math.scalar_mul(
            scalar=tf.cast(sequence_length, dtype=tf.int32),
            x=tf.range(
                start=0,
                limit=tf.cast(batch_size, dtype=tf.int32),
                dtype=tf.int32,
            ),
        )
        to_subtract = tf.expand_dims(to_subtract, axis=-1)
        batched_vision_indices = tf.math.subtract(
            batched_vision_indices, to_subtract
        )

        batched_vision_indices = batched_vision_indices.to_tensor(
            default_value=0
        )

        if max_tokens is not None:
            # Pad or truncate to max_tokens.
            current_tokens = tf.shape(batched_vision_indices)[1]
            pad_amount = tf.maximum(0, max_tokens - current_tokens)
            batched_vision_indices = tf.pad(
                batched_vision_indices, [[0, 0], [0, pad_amount]]
            )
            batched_vision_indices = batched_vision_indices[:, :max_tokens]

        return batched_vision_indices

    def _get_audio_indices(self, audio_mask):
        batch_size, sequence_length = audio_mask.shape

        audio_mask_flattened = tf.reshape(audio_mask, [-1])
        audio_indices = tf.where(audio_mask_flattened)[..., 0]
        audio_indices = tf.cast(audio_indices, dtype=tf.int32)

        row_lengths = tf.math.reduce_sum(
            tf.cast(audio_mask, dtype=audio_indices.dtype), axis=1
        )

        batched_audio_indices = tf.RaggedTensor.from_row_lengths(
            values=audio_indices,
            row_lengths=row_lengths,
        )

        to_subtract = tf.math.scalar_mul(
            scalar=tf.cast(sequence_length, dtype=tf.int32),
            x=tf.range(
                start=0,
                limit=tf.cast(batch_size, dtype=tf.int32),
                dtype=tf.int32,
            ),
        )
        to_subtract = tf.expand_dims(to_subtract, axis=-1)
        batched_audio_indices = tf.math.subtract(
            batched_audio_indices, to_subtract
        )

        batched_audio_indices = batched_audio_indices.to_tensor(default_value=0)
        return batched_audio_indices

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

        batched = True
        if isinstance(texts, str):
            batched = False
            texts = [texts]
        elif isinstance(texts, tf.Tensor) and len(texts.shape) == 0:
            batched = False
            texts = tf.expand_dims(texts, axis=0)

        if texts is None:
            # Media only: one placeholder prompt per example. A single
            # rank-3 image, rank-4 video or rank-1 waveform is one example,
            # exactly as in `_call_python`.
            batched = True
            if images is not None:
                if isinstance(images, list):
                    b = len(images)
                elif len(images.shape) == 3:
                    batched = False
                    b = 1
                else:
                    b = tf.shape(images)[0]
                texts = tf.repeat(tf.constant(self.image_placeholder), b)
            elif videos is not None:
                if isinstance(videos, list):
                    b = len(videos)
                elif len(videos.shape) == 4:
                    batched = False
                    b = 1
                else:
                    b = tf.shape(videos)[0]
                texts = tf.repeat(tf.constant(self.video_placeholder), b)
            elif audio is not None:
                if isinstance(audio, list):
                    b = len(audio)
                elif len(audio.shape) == 1:
                    batched = False
                    b = 1
                else:
                    b = tf.shape(audio)[0]
                texts = tf.repeat(tf.constant(self.audio_placeholder), b)
            else:
                texts = tf.constant([""])

        # Vision handling
        pixel_values = None
        pixel_position_ids = None

        if self.image_converter is not None and images is not None:
            if isinstance(images, tf.RaggedTensor):
                images_list = [images[i] for i in range(images.shape[0])]
            elif isinstance(images, list):
                images_list = images
            else:
                if len(tf.shape(images)) == 3 and not batched:
                    images_list = [images]
                else:
                    images_list = [
                        images[i] for i in range(tf.shape(images)[0])
                    ]

            pv_list, ppi_list, tokens_list = [], [], []
            for img in images_list:
                img = _to_dense_tensor(img)
                img = tf.expand_dims(img, 0)
                image_outputs = self.image_converter(img)
                pv = image_outputs["pixel_values"]
                ppi = image_outputs["pixel_position_ids"]
                # Count non padded
                valid_patches = tf.cast(ppi[0, :, 0] != -1, tf.int32)
                num_patches = tf.reduce_sum(valid_patches)
                num_soft_tokens = num_patches // (
                    self.image_converter.pooling_kernel_size**2
                )

                pv_list.append(pv[0])
                ppi_list.append(ppi[0])
                tokens_list.append(num_soft_tokens)

            pixel_values = tf.stack(pv_list, axis=0)
            pixel_position_ids = tf.stack(ppi_list, axis=0)
            pixel_values = tf.expand_dims(pixel_values, 1)
            pixel_position_ids = tf.expand_dims(pixel_position_ids, 1)

            num_tokens_tensor = tf.stack(tokens_list)

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
            if isinstance(videos, tf.RaggedTensor):
                videos_list = [videos[i] for i in range(videos.shape[0])]
            elif isinstance(videos, list):
                videos_list = videos
            else:
                if len(tf.shape(videos)) == 4 and not batched:
                    videos_list = [videos]
                else:
                    videos_list = [
                        videos[i] for i in range(tf.shape(videos)[0])
                    ]

            pv_list, ppi_list, tokens_list = [], [], []

            for vid in videos_list:
                vid = _to_dense_tensor(vid)
                vid = tf.expand_dims(vid, 0)
                video_outputs = self.video_converter(vid)
                pv = video_outputs["pixel_values"][0]
                ppi = video_outputs["pixel_position_ids"][0]

                valid_patches = tf.cast(ppi[:, :, 0] != -1, tf.int32)
                num_patches = tf.reduce_sum(valid_patches, axis=1)
                num_soft_tokens = num_patches // (
                    self.video_converter.image_converter.pooling_kernel_size**2
                )

                pv_list.append(pv)
                ppi_list.append(ppi)
                tokens_list.append(num_soft_tokens)

            # Pad video features to maximum size so they can be stacked
            max_frames = 0
            max_patches = 0
            for ppi in ppi_list:
                max_frames = tf.maximum(max_frames, tf.shape(ppi)[0])
                max_patches = tf.maximum(max_patches, tf.shape(ppi)[1])

            padded_pv_list = []
            padded_ppi_list = []

            for pv, ppi in zip(pv_list, ppi_list):
                frames = tf.shape(ppi)[0]
                patches = tf.shape(ppi)[1]

                pad_frames = max_frames - frames
                pad_patches = max_patches - patches

                pv = tf.pad(
                    pv,
                    [[0, pad_frames], [0, pad_patches], [0, 0]],
                    constant_values=0.0,
                )
                ppi = tf.pad(
                    ppi,
                    [[0, pad_frames], [0, pad_patches], [0, 0]],
                    constant_values=-1,
                )

                padded_pv_list.append(pv)
                padded_ppi_list.append(ppi)

            pixel_values = tf.stack(padded_pv_list, axis=0)
            pixel_position_ids = tf.stack(padded_ppi_list, axis=0)

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

            ragged_tokens = tf.RaggedTensor.from_row_lengths(
                values=tf.concat(tokens_list, axis=0),
                row_lengths=tf.stack([tf.shape(t)[0] for t in tokens_list]),
            )
            texts = tf.map_fn(
                _replace_video,
                (texts, ragged_tokens),
                fn_output_signature=tf.string,
            )

        audio_mel = None
        audio_mel_mask = None

        if self.audio_converter is not None and audio is not None:
            if isinstance(audio, tf.RaggedTensor):
                audio_list = [audio[i] for i in range(audio.shape[0])]
            elif isinstance(audio, list):
                audio_list = audio
            else:
                if len(tf.shape(audio)) == 1 and not batched:
                    audio_list = [audio]
                else:
                    audio_list = [audio[i] for i in range(tf.shape(audio)[0])]

            mel_list = []
            tokens_list = []
            output_lengths_list = []
            max_len = 0

            for aud in audio_list:
                len_i = tf.shape(aud)[0]
                aud = _to_dense_tensor(aud)
                aud = tf.expand_dims(aud, 0)
                mel = self.audio_converter(aud)
                mel = tf.convert_to_tensor(mel)
                mel = mel[0]

                # HF derivation: left-pad 160, unfold 321, hop 160
                # -> (len_i - 1) // hop frames.
                output_lengths_i = tf.maximum(
                    (len_i - 1) // self.audio_converter.stride, 0
                )
                output_lengths_i = tf.minimum(
                    output_lengths_i, tf.shape(mel)[0]
                )
                num_audio_tokens = tf.maximum((output_lengths_i + 3) // 4, 1)

                mel_list.append(mel[:output_lengths_i])
                output_lengths_list.append(output_lengths_i)
                tokens_list.append(num_audio_tokens)
                max_len = tf.maximum(max_len, output_lengths_i)

            padded_mel_list = []
            mask_list = []
            for mel, out_len in zip(mel_list, output_lengths_list):
                pad_l = max_len - out_len
                padded_mel = tf.pad(mel, [[0, pad_l], [0, 0]])
                padded_mel_list.append(padded_mel)
                mask_list.append(
                    tf.concat(
                        [
                            tf.ones([out_len], dtype=tf.int32),
                            tf.zeros([pad_l], dtype=tf.int32),
                        ],
                        axis=0,
                    )
                )

            audio_mel = tf.stack(padded_mel_list, axis=0)
            audio_mel = tf.expand_dims(audio_mel, 1)
            audio_mel_mask = tf.stack(mask_list, axis=0)
            audio_mel_mask = tf.expand_dims(audio_mel_mask, 1)

            num_tokens_tensor = tf.stack(tokens_list)

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

        token_ids = self.tokenizer(texts)
        token_ids, segment_ids = self.packer(
            token_ids,
            sequence_length=sequence_length,
            add_start_value=True,
            add_end_value=True,
        )
        padding_mask = token_ids != self.tokenizer.pad_token_id

        batch_size = tf.shape(token_ids)[0]

        vision_indices = None
        vision_mask = None

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
            vision_indices = self._get_vision_indices(vision_mask)
            vision_indices = tf.cast(vision_indices, "int32")
            vision_mask = tf.cast(vision_mask, "int32")
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
            vision_mask = tf.zeros_like(token_ids, dtype="int32")
            vision_indices = tf.zeros([batch_size, 0], dtype="int32")

        audio_indices = None
        audio_mask = None

        if audio_mel is not None:
            audio_mask = token_ids == self.tokenizer.token_to_id(
                self.audio_placeholder
            )
            audio_indices = self._get_audio_indices(audio_mask)
            audio_indices = tf.cast(audio_indices, "int32")
            audio_mask = tf.cast(audio_mask, "int32")
        elif self.audio_converter is not None:
            input_feat_size = self.audio_converter.num_mels
            audio_mel = tf.zeros(
                [batch_size, 0, 1, input_feat_size], dtype="float32"
            )
            audio_mel_mask = tf.zeros([batch_size, 0, 1], dtype="int32")
            audio_indices = tf.zeros([batch_size, 0], dtype="int32")
            audio_mask = tf.zeros_like(token_ids, dtype="int32")

        output = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

        if self.image_converter is not None or self.video_converter is not None:
            output["pixel_values"] = pixel_values
            output["pixel_position_ids"] = pixel_position_ids
            output["vision_indices"] = vision_indices
            output["vision_mask"] = vision_mask

        if self.audio_converter is not None:
            output["audio_mel"] = audio_mel
            output["audio_mel_mask"] = tf.cast(audio_mel_mask, "int32")
            output["audio_indices"] = audio_indices
            output["audio_mask"] = audio_mask

        if not batched:
            output = keras.tree.map_structure(
                lambda x: tf.squeeze(x, axis=0), output
            )

        return keras.utils.pack_x_y_sample_weight(output, y, sample_weight)

    def _python_indices(self, token_ids, placeholder_ids):
        """Numpy twin of `_get_vision_indices` / `_get_audio_indices`.

        Args:
            token_ids: int numpy array of shape `(batch, seq)`.
            placeholder_ids: the token ids that mark a soft-token slot.

        Returns:
            `(mask, indices)`: an int32 `(batch, seq)` mask and the int32
            `(batch, max_tokens)` positions of the set bits, right-padded
            with 0 to the longest row, exactly like the TF path.
        """
        mask = np.isin(token_ids, list(placeholder_ids))
        counts = mask.sum(axis=1)
        max_len = int(counts.max()) if counts.size else 0
        indices = np.zeros((token_ids.shape[0], max_len), dtype="int32")
        for row, (m, n) in enumerate(zip(mask, counts)):
            indices[row, :n] = np.flatnonzero(m)
        return mask.astype("int32"), indices

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

        if tf is not None:
            if isinstance(images, tf.RaggedTensor):
                images = [r.numpy() for r in images]
            elif isinstance(images, tf.Tensor):
                images = images.numpy()

            if isinstance(videos, tf.RaggedTensor):
                videos = [r.numpy() for r in videos]
            elif isinstance(videos, tf.Tensor):
                videos = videos.numpy()

            if isinstance(audio, tf.RaggedTensor):
                audio = [r.numpy() for r in audio]
            elif isinstance(audio, tf.Tensor):
                audio = audio.numpy()

        if texts is not None:
            texts, is_batched, outer_shape = canonicalize_python_string_inputs(
                texts
            )
            if outer_shape is not None:
                raise ValueError(
                    "EmbeddingGemma2TextEmbedderPreprocessor does not "
                    "support rank>=2 text inputs."
                )
        else:
            is_batched = True
            if images is not None:
                if isinstance(images, (list, tuple)):
                    b = len(images)
                else:
                    images = np.asarray(images)
                    if len(images.shape) == 3:
                        is_batched = False
                        b = 1
                    else:
                        b = images.shape[0]
                texts = [self.image_placeholder] * b
            elif videos is not None:
                if isinstance(videos, (list, tuple)):
                    b = len(videos)
                else:
                    videos = np.asarray(videos)
                    if len(videos.shape) == 4:
                        is_batched = False
                        b = 1
                    else:
                        b = videos.shape[0]
                texts = [self.video_placeholder] * b
            elif audio is not None:
                if isinstance(audio, (list, tuple)):
                    b = len(audio)
                else:
                    audio = np.asarray(audio)
                    if len(audio.shape) == 1:
                        is_batched = False
                        b = 1
                    else:
                        b = audio.shape[0]
                texts = [self.audio_placeholder] * b
            else:
                texts = [""]

        # Vision handling
        pixel_values = None
        pixel_position_ids = None

        if self.image_converter is not None and images is not None:
            if isinstance(images, (list, tuple)):
                images_list = [np.asarray(e) for e in images]
            elif isinstance(images, np.ndarray):
                if len(images.shape) == 3 and not is_batched:
                    images_list = [images]
                else:
                    images_list = [images[i] for i in range(images.shape[0])]
            else:
                images_list = [np.asarray(images)]

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
                if len(videos.shape) == 4 and not is_batched:
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
                if len(audio.shape) == 1 and not is_batched:
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
                len_i = aud_arr.shape[0]
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

        batch_size = token_ids.shape[0]

        vision_indices = None
        vision_mask = None

        if pixel_values is not None:
            img_tok = self.tokenizer.token_to_id(self.image_placeholder)
            vid_tok = self.tokenizer.token_to_id(self.video_placeholder)
            vision_mask, vision_indices = self._python_indices(
                token_ids, [img_tok, vid_tok]
            )
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
            vision_mask = np.zeros_like(token_ids, dtype="int32")
            vision_indices = np.zeros((batch_size, 0), dtype="int32")

        audio_indices = None
        audio_mask = None

        if audio_mel is not None:
            aud_tok = self.tokenizer.token_to_id(self.audio_placeholder)
            audio_mask, audio_indices = self._python_indices(
                token_ids, [aud_tok]
            )
        elif self.audio_converter is not None:
            input_feat_size = self.audio_converter.num_mels
            audio_mel = np.zeros(
                (batch_size, 0, 1, input_feat_size), dtype="float32"
            )
            audio_mel_mask = np.zeros((batch_size, 0, 1), dtype="int32")
            audio_indices = np.zeros((batch_size, 0), dtype="int32")
            audio_mask = np.zeros_like(token_ids, dtype="int32")

        output = {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

        if self.image_converter is not None or self.video_converter is not None:
            output["pixel_values"] = pixel_values
            output["pixel_position_ids"] = pixel_position_ids
            output["vision_indices"] = vision_indices
            output["vision_mask"] = vision_mask

        if self.audio_converter is not None:
            output["audio_mel"] = audio_mel
            output["audio_mel_mask"] = np.array(audio_mel_mask, dtype="int32")
            output["audio_indices"] = audio_indices
            output["audio_mask"] = audio_mask

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
