import math
import re

import keras
import numpy as np
import tensorflow as tf

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.multi_segment_packer import (
    MultiSegmentPacker,
)
from keras_hub.src.models.causal_lm_preprocessor import CausalLMPreprocessor
from keras_hub.src.models.gemma4.gemma4_audio_converter import (
    Gemma4AudioConverter,
)
from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.models.gemma4.gemma4_image_converter import (
    Gemma4ImageConverter,
)
from keras_hub.src.models.gemma4.gemma4_tokenizer import Gemma4Tokenizer
from keras_hub.src.models.gemma4.gemma4_video_converter import (
    Gemma4VideoConverter,
)
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import strip_to_ragged


def _get_num_vision_tokens(
    h, w, patch_size, max_soft_tokens, pooling_kernel_size
):
    total_px = h * w
    max_patches = max_soft_tokens * (pooling_kernel_size**2)
    target_px = max_patches * (patch_size**2)
    factor = math.sqrt(target_px / total_px)
    ideal_h = factor * h
    ideal_w = factor * w
    side_mult = pooling_kernel_size * patch_size

    target_h = int(math.floor(ideal_h / side_mult)) * side_mult
    target_w = int(math.floor(ideal_w / side_mult)) * side_mult

    target_h = max(target_h, side_mult)
    target_w = max(target_w, side_mult)

    n_h = target_h // patch_size
    n_w = target_w // patch_size

    return (n_h * n_w) // (pooling_kernel_size**2)


@keras_hub_export("keras_hub.models.Gemma4CausalLMPreprocessor")
class Gemma4CausalLMPreprocessor(CausalLMPreprocessor):
    """Gemma4 Causal LM preprocessor.

    This preprocessing layer is meant for use with
    `keras_hub.models.Gemma4CausalLM`. It can be configured in three modes:
    text-only, text + image/video, and text + audio, based on whether
    `image_converter`, `video_converter`, or `audio_converter` are provided.
    It returns outputs in a `(x, y, sample_weight)` format, where the `y` label
    is the next token id in the `x` sequence. `sample_weight` is 0 for "prompt"
    tokens and 1 for "response" tokens, so that the loss is computed only on
    the "response" tokens.

    For image inputs, this layer replaces each `<|image|>` placeholder in the
    prompt with `num_vision_tokens_per_image` soft tokens wrapped in
    `<|image>...<image|>` markers. It also returns indices of where these
    vision tokens are present so that image embeddings can be placed at the
    correct positions in the sequence.

    For video inputs, each `<|video|>` placeholder is replaced with a sequence
    of per-frame blocks. Each block contains a timestamp and
    `num_vision_tokens_per_frame` soft tokens wrapped in `<|image>...<image|>`
    markers. The actual token count per frame is computed dynamically from the
    input frame dimensions.

    For audio inputs, each `<|audio|>` placeholder is expanded to the exact
    number of audio tokens required for the clip, computed dynamically from the
    mel-spectrogram length.

    By default, per-frame timestamps are computed from sequential indices
    `[0, 1, ..., N-1]` at `video_fps`. When your video was sampled at
    irregular intervals (e.g. every 8th frame of a 30 fps source), set
    `preprocessor.video_metadata` to a list of per-sample dicts before
    calling the preprocessor. Each dict accepts a `"frames_indices"` key
    (`list[int]`, the source frame indices that were sampled) and an optional
    `"fps"` key (`float`, defaults to `preprocessor.video_fps`). When
    `video_metadata` is `None` (the default) the preprocessor falls back to
    sequential indices at `video_fps`, so existing code is unaffected.

    Examples:

    Using `video_metadata` to pass real frame indices and fps.
    ```python
    # One dict per sample in the batch.
    preprocessor.video_metadata = [
        {"frames_indices": [0, 8, 16, 24], "fps": 30.0},
    ]
    output = preprocessor({
        "prompts": ["Describe this video: <|video|>"],
        "responses": [""],
        "videos": [my_video_frames],  # shape (N_frames, H, W, 3)
    })
    preprocessor.video_metadata = None  # reset to default after use
    ```

    For use with generation, the layer also exposes two methods
    `generate_preprocess()` and `generate_postprocess()`. When this preprocessor
    is attached to a `keras_hub.models.Gemma4CausalLM` instance, these methods
    will be called implicitly in `generate()`. They can also be called
    standalone (e.g. to precompute preprocessing inputs for generation in a
    separate process).

    Args:
        tokenizer: A `keras_hub.models.Gemma4Tokenizer` instance.
        image_converter: A `keras_hub.layers.Gemma4ImageConverter` instance.
            Defaults to `None`.
        audio_converter: A `keras_hub.layers.Gemma4AudioConverter` instance.
            Defaults to `None`.
        video_converter: A `keras_hub.layers.Gemma4VideoConverter` instance.
            Defaults to `None`.
        sequence_length: int. The length of the packed inputs. Defaults to
            `1024`.
        add_start_token: bool. If `True`, the preprocessor will prepend the
            tokenizer start token to each input sequence. Defaults to `True`.
        add_end_token: bool. If `True`, the preprocessor will append the
            tokenizer end token to each input sequence. Defaults to `True`.
        max_images_per_prompt: int. Maximum number of images per sample in
            the batch. Defaults to `2`.
        num_vision_tokens_per_image: int. Number of vision placeholder tokens
            per image. Defaults to `280`.
        max_audio_clips_per_prompt: int. Maximum number of audio clips per
            sample in the batch. Defaults to `1`.
        num_audio_tokens_per_clip: int. Legacy parameter, no longer used for
            token calculation as audio expansion is now fully dynamic. Defaults
            to `750`.
        audio_input_feat_size: int. Number of mel-spectrogram frequency bins.
            Defaults to `128`.
        num_frames_per_video: int. Number of frames sampled from each video.
            Defaults to `32`.
        num_vision_tokens_per_frame: int. Fallback number of vision placeholder
            tokens per video frame, used when a video converter is configured
            but no video input is provided. The actual count is computed
            dynamically from frame dimensions when videos are present. Defaults
            to `70`.
        video_fps: float. Frames-per-second used to compute per-frame
            timestamps in the expanded prompt. Defaults to `24.0`.
    """

    backbone_cls = Gemma4Backbone
    tokenizer_cls = Gemma4Tokenizer
    image_converter_cls = Gemma4ImageConverter
    audio_converter_cls = Gemma4AudioConverter
    video_converter_cls = Gemma4VideoConverter

    def __init__(
        self,
        tokenizer,
        image_converter=None,
        audio_converter=None,
        video_converter=None,
        sequence_length=1024,
        add_start_token=True,
        add_end_token=True,
        max_images_per_prompt=2,
        num_vision_tokens_per_image=280,
        max_audio_clips_per_prompt=1,
        num_audio_tokens_per_clip=750,
        audio_input_feat_size=128,
        num_frames_per_video=32,
        num_vision_tokens_per_frame=70,
        video_fps=24.0,
        **kwargs,
    ):
        super().__init__(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            add_start_token=add_start_token,
            add_end_token=add_end_token,
            **kwargs,
        )

        # Ensure values fit.
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

        self.audio_converter = audio_converter
        self.max_audio_clips_per_prompt = max_audio_clips_per_prompt
        self.num_audio_tokens_per_clip = num_audio_tokens_per_clip

        self.video_converter = video_converter
        self.num_frames_per_video = num_frames_per_video
        self.num_vision_tokens_per_frame = num_vision_tokens_per_frame
        self.video_fps = video_fps

        # Number of mel-spectrogram frequency bins expected by the audio
        # encoder. Used to produce correctly-shaped dummy audio tensors.
        self.audio_input_feat_size = audio_input_feat_size

        # The preprocessor and model are "text-only" if no converters are
        # passed.
        self.text_only_model = (
            self.image_converter is None
            and self.audio_converter is None
            and self.video_converter is None
        )

        if self.image_converter is None:
            self.image_placeholder = None
            self.start_of_image_token = None
            self.end_of_image_token = None
        else:
            self.image_placeholder = "<|image|>"
            self.start_of_image_token = "<|image>"
            self.end_of_image_token = "<image|>"

        if self.audio_converter is None:
            self.audio_placeholder = None
            self.start_of_audio_token = None
            self.end_of_audio_token = None
        else:
            self.audio_placeholder = "<|audio|>"
            self.start_of_audio_token = "<|audio>"
            self.end_of_audio_token = "<audio|>"

        if self.video_converter is None:
            self.video_placeholder = None
            self.start_of_video_token = None
            self.end_of_video_token = None
        else:
            self.video_placeholder = "<|video|>"
            self.start_of_video_token = "<|video>"
            self.end_of_video_token = "<video|>"

    def build(self, input_shape):
        # Defer packer creation to `build()` so that we can be sure tokenizer
        # assets have loaded when restoring a saved model.
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sep_value=[],
            sequence_length=self.sequence_length,
        )
        self.built = True

    def _get_vision_indices(self, vision_mask, max_tokens=None):
        """Computes indices given vision mask, and pads with 0.

        Args:
            vision_mask: Bool tensor of shape ``[B, S]``.
            max_tokens: Fixed output width (number of vision slots per sample).
                When provided, the result is always padded/truncated to this
                length, ensuring a static second dimension across all batches
                (required for XLA compilation and consistent batching).
                When ``None``, pads to the maximum actual count in the batch.
        """
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
                limit=tf.shape(vision_mask)[0],
                dtype=tf.int32,
            ),
        )

        batched_vision_indices = tf.math.subtract(
            batched_vision_indices,
            tf.expand_dims(to_subtract, axis=-1),
        )

        # Pad to `max_tokens` when specified (fixed capacity for all batches),
        # otherwise pad to the maximum actual count in this batch.
        pad_shape = [None, max_tokens] if max_tokens is not None else None
        batched_vision_indices = batched_vision_indices.to_tensor(
            default_value=0,
            shape=pad_shape,
        )
        return batched_vision_indices

    def _get_audio_indices(self, audio_mask):
        """Computes indices given audio mask, and pads with 0."""
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
                limit=tf.shape(audio_mask)[0],
                dtype=tf.int32,
            ),
        )

        batched_audio_indices = tf.math.subtract(
            batched_audio_indices,
            tf.expand_dims(to_subtract, axis=-1),
        )

        # Pad the indices to the max length in the batch (dynamic).
        batched_audio_indices = batched_audio_indices.to_tensor(
            default_value=0,
        )
        return batched_audio_indices

    def _format_output(
        self,
        token_ids,
        pixel_values,
        pixel_position_ids,
        vision_mask,
        response_mask,
        padding_mask,
        audio_mel=None,
        audio_mel_mask=None,
        audio_indices=None,
        audio_mask=None,
        return_labels=False,
        text_only_input=False,
        batched=False,
    ):
        if return_labels:
            # Target `y` will be the next token.
            y = token_ids[..., 1:]
            # Only compute the loss for labels in the response.
            sample_weight = response_mask[..., 1:]

            # The last token does not have a next token. Remove it.
            token_ids = token_ids[..., :-1]
            vision_mask = vision_mask[..., :-1]
            response_mask = response_mask[..., :-1]
            padding_mask = padding_mask[..., :-1]
            if audio_mask is not None:
                audio_mask = audio_mask[..., :-1]
                audio_indices = self._get_audio_indices(audio_mask)

        batch_size = tf.shape(vision_mask)[0]
        seq_len = tf.shape(token_ids)[-1]
        # Use sequential position IDs for all tokens (including audio tokens).
        # HF does not provide position_ids from the processor; the model
        # defaults to torch.arange(seq_len), i.e. plain sequential IDs.
        # Matching this is essential for correct RoPE embeddings on audio
        # token positions.
        position_ids = tf.range(seq_len, dtype=tf.int32)
        position_ids = tf.expand_dims(position_ids, axis=0)
        position_ids = tf.tile(position_ids, [batch_size, 1])

        if text_only_input:
            vision_indices = tf.ones(
                shape=[batch_size, 0],
                dtype=tf.int32,
            )
        else:
            # Pad to a fixed capacity so all batches have the same shape.
            # For images: max_images_per_prompt × tokens_per_image.
            # For videos: num_frames × max_soft_tokens from the converter
            #   (authoritative upper bound; avoids dependence on the separate
            #   `num_vision_tokens_per_frame` fallback parameter).
            if self.video_converter is not None:
                max_vision_tokens = (
                    self.num_frames_per_video
                    * self.video_converter.max_soft_tokens
                )
            elif self.image_converter is not None:
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
            patch_dim = (
                3 * self.image_converter.patch_size**2
                if self.image_converter is not None
                else 48
            )
            pixel_values = tf.zeros(
                (batch_size, 0, 1, patch_dim), dtype="float32"
            )
        if pixel_position_ids is None:
            pixel_position_ids = tf.zeros((batch_size, 0, 1, 2), dtype="int32")

        x = {
            "pixel_values": pixel_values
            if (pixel_values is None or batched)
            else tf.squeeze(pixel_values, axis=0),
            "pixel_position_ids": pixel_position_ids
            if (pixel_position_ids is None or batched)
            else tf.squeeze(pixel_position_ids, axis=0),
            # Text
            "token_ids": (
                token_ids if batched else tf.squeeze(token_ids, axis=0)
            ),
            "vision_indices": (
                vision_indices
                if batched
                else tf.squeeze(vision_indices, axis=0)
            ),
            "vision_mask": (
                vision_mask if batched else tf.squeeze(vision_mask, axis=0)
            ),
            "padding_mask": (
                padding_mask if batched else tf.squeeze(padding_mask, axis=0)
            ),
            "position_ids": (
                position_ids if batched else tf.squeeze(position_ids, axis=0)
            ),
        }

        if audio_mel is not None:
            x["audio_mel"] = (
                audio_mel if batched else tf.squeeze(audio_mel, axis=0)
            )
            x["audio_mel_mask"] = (
                audio_mel_mask
                if batched
                else tf.squeeze(audio_mel_mask, axis=0)
            )
            x["audio_indices"] = (
                audio_indices if batched else tf.squeeze(audio_indices, axis=0)
            )
            x["audio_mask"] = (
                audio_mask if batched else tf.squeeze(audio_mask, axis=0)
            )
        elif self.audio_input_feat_size > 0:
            audio_mel_dummy = tf.zeros(
                shape=[batch_size, 0, 1, self.audio_input_feat_size],
                dtype="float32",
            )
            audio_mel_mask_dummy = tf.zeros(
                shape=[batch_size, 0, 1], dtype=tf.int32
            )
            audio_indices_dummy = tf.zeros(
                shape=[batch_size, 0], dtype=tf.int32
            )
            audio_mask_dummy = tf.zeros(
                shape=[batch_size, tf.shape(token_ids)[1]], dtype=tf.bool
            )
            if not batched:
                audio_mel_dummy = tf.squeeze(audio_mel_dummy, axis=0)
                audio_mel_mask_dummy = tf.squeeze(audio_mel_mask_dummy, axis=0)
                audio_indices_dummy = tf.squeeze(audio_indices_dummy, axis=0)
                audio_mask_dummy = tf.squeeze(audio_mask_dummy, axis=0)
            x["audio_mel"] = audio_mel_dummy
            x["audio_mel_mask"] = audio_mel_mask_dummy
            x["audio_indices"] = audio_indices_dummy
            x["audio_mask"] = audio_mask_dummy

        if return_labels:
            if not batched:
                y = tf.squeeze(y, axis=0)
                sample_weight = tf.squeeze(sample_weight, 0)

            return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)
        else:
            return x

    def _preprocess_images(self, images, batched):
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
            images = images.to_tensor(
                shape=[None, self.max_images_per_prompt, None, None, 3],
                default_value=0,
            )
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

        original_images_shape = tf.shape(images)
        images = tf.reshape(
            images,
            [
                -1,
                original_images_shape[-3],
                original_images_shape[-2],
                original_images_shape[-1],
            ],
        )
        images_dict = self.image_converter.call(images)
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
                -1,
                self.image_converter.patch_size**2 * 3,
            ],
        )
        pixel_position_ids = tf.reshape(
            pixel_position_ids,
            [
                original_images_shape[0],
                original_images_shape[1],
                -1,
                2,
            ],
        )
        return {
            "pixel_values": pixel_values,
            "pixel_position_ids": pixel_position_ids,
        }

    def _preprocess_audio(self, audio, batched):
        """Converts raw audio into Mel spectrograms."""
        if not batched or (hasattr(audio, "shape") and len(audio.shape) == 1):
            # Expand dims so rank >= 2
            audio = tf.expand_dims(audio, axis=0)

        if isinstance(audio, (list, np.ndarray)):
            # convert list of clips/waveform to Ragged
            audio = tf.ragged.constant(audio)
        elif hasattr(audio, "shape") and not hasattr(audio, "to_tensor"):
            # Covert dense TF Tensor to Ragged for variable length handlers
            # coords structure
            audio = tf.RaggedTensor.from_tensor(audio)

        # Flatten clips per Sample if shape is (B, Clips, T) or just (B, T)
        # Assuming simple 1 clip per prompt (B, T) Ragged Tensor
        audio_tensor = audio.to_tensor(shape=[None, None], default_value=0.0)

        mel = self.audio_converter(audio_tensor)  # (B, Seq, Feat)

        # The audio converter runs as a Keras layer and may return a CUDA
        # torch tensor on GPU. Move to CPU so subsequent TF ops can accept it
        # (mirrors the same guard in _preprocess_images).
        if keras.config.backend() == "torch":
            if not isinstance(mel, tf.Tensor):
                mel = mel.cpu()

        # Expand dims to model expectation of Clips step: (B, 1, Seq, Feat)
        mel = tf.expand_dims(mel, axis=1)

        # Audio feature mask creation from Ragged tensor row lengths
        row_lengths = audio.row_lengths()
        # Compute Mel output lengths: stride or max step coords triggers
        stride = self.audio_converter.stride
        output_lengths = tf.cast(row_lengths // stride, tf.int32)

        mask = tf.sequence_mask(
            output_lengths, maxlen=tf.shape(mel)[2], dtype=tf.int32
        )
        mask = tf.expand_dims(mask, axis=1)  # (B, 1, Seq)

        return mel, mask

    def _preprocess_videos(self, videos, batched):
        if "jax" in str(type(videos)):
            videos = tf.convert_to_tensor(np.array(videos))
        elif "torch.Tensor" in str(type(videos)):
            videos = tf.convert_to_tensor(videos.detach().cpu().float().numpy())
        elif isinstance(videos, np.ndarray):
            videos = tf.convert_to_tensor(videos)
        elif isinstance(videos, list):
            videos = tf.ragged.constant(videos)

        if isinstance(videos, tf.RaggedTensor):
            if not batched:
                videos = tf.expand_dims(videos, axis=0)
            videos = videos.to_tensor(default_value=0)
        elif not isinstance(videos, tf.Tensor):
            raise ValueError(
                "`videos` should be a list, ragged tensor, or dense tensor."
                f" Received: `type(videos)` = {type(videos)}"
            )

        if len(videos.shape) == 4:
            videos = tf.expand_dims(videos, axis=0)

        videos_dict = self.video_converter.call(videos)
        pixel_values = videos_dict["pixel_values"]
        pixel_position_ids = videos_dict["pixel_position_ids"]

        if keras.config.backend() == "torch":
            if not isinstance(pixel_values, tf.Tensor):
                pixel_values = pixel_values.cpu()
            if not isinstance(pixel_position_ids, tf.Tensor):
                pixel_position_ids = pixel_position_ids.cpu()

        return {
            "pixel_values": pixel_values,
            "pixel_position_ids": pixel_position_ids,
        }

    def _compute_video_n_tokens(self, videos):
        """Return the actual number of vision tokens per frame for this input.

        Derives H and W from the raw video tensor and calls
        `_get_num_vision_tokens` so that the prompt expansion matches the
        number of patches the video converter will actually produce.
        """
        # Drill down through list-of-videos to the first array-like leaf.
        v = videos
        while isinstance(v, (list, tuple)):
            v = v[0]

        if isinstance(v, np.ndarray):
            shape = v.shape
        elif hasattr(v, "numpy"):  # eager tf.Tensor / torch.Tensor
            shape = v.numpy().shape
        elif hasattr(v, "shape"):  # symbolic tf.Tensor or torch.Tensor
            shape = tuple(int(d) for d in v.shape)
        else:
            raise ValueError(
                f"Cannot determine video frame dimensions from input of "
                f"type {type(v)}. Expected a numpy array, tf.Tensor, or "
                f"torch.Tensor."
            )

        # Accepted shapes: (H,W,C), (F,H,W,C), (B,F,H,W,C)
        if len(shape) < 3:
            raise ValueError(
                f"Video input has unexpected shape {shape}. Expected at "
                f"least 3 dimensions (..., H, W, C)."
            )
        h, w = int(shape[-3]), int(shape[-2])

        return _get_num_vision_tokens(
            h,
            w,
            self.video_converter.patch_size,
            self.video_converter.max_soft_tokens,
            self.video_converter.pooling_kernel_size,
        )

    def _build_video_replacement(self, n_tokens, frames_indices, fps):
        """Build the video replacement string for a single video.

        Uses source frame indices and fps to compute accurate per-frame
        timestamps, matching HF Gemma4Processor behavior.

        Args:
            n_tokens: int. Vision tokens per frame.
            frames_indices: list[int]. Source video frame indices.
            fps: float. Source video frames-per-second.

        Returns:
            str. Replacement string for a single ``<|video|>`` placeholder.
        """
        frame_strs = []
        for idx in frames_indices:
            seconds = idx / fps
            mm = int(seconds // 60)
            ss = int(seconds % 60)
            timestamp = f"{mm:02d}:{ss:02d}"
            frame_strs.append(
                f"{timestamp} <|image>{'<|video|>' * n_tokens}<image|>"
            )
        return " ".join(frame_strs)

    def _expand_video_prompt(self, prompts, videos):
        """Expand ``<|video|>`` placeholders with per-frame token sequences.

        Reads ``self.video_metadata`` if it has been set before calling this
        preprocessor.  Each element of the list should be a dict with keys:

        Each dict accepts a `"frames_indices"` key (`list[int]`, the source
        frame indices sampled from the original video, e.g. `[0, 8, 16, ...]`)
        and an optional `"fps"` key (`float`, the source video fps).

        When `video_metadata` is `None` (the default), falls back to sequential
        indices `[0, 1, ..., N-1]` at `self.video_fps`, preserving existing
        behaviour.

        Examples:

        Using `video_metadata` when prompt expansion tokens are needed.
        ```python
        preprocessor.video_metadata = [
            {"frames_indices": [0, 8, 16], "fps": 30.0},
        ]
        prompts = preprocessor._expand_video_prompt(
            tf.constant(["<|video|>"]), videos=None
        )
        preprocessor.video_metadata = None  # reset after use
        ```

        Args:
            prompts: A `tf.Tensor` of strings (batch of prompts).
            videos: Raw video input, or `None`.

        Returns:
            A `tf.Tensor` of strings with `<|video|>` placeholders replaced.
        """
        n_tokens = (
            self._compute_video_n_tokens(videos)
            if videos is not None
            else self.num_vision_tokens_per_frame
        )
        vid_pattern = re.escape(self.video_placeholder)
        video_metadata = getattr(self, "video_metadata", None)

        if video_metadata is None:
            # Default: sequential indices [0, 1, ..., N-1] at self.video_fps.
            frames_indices = list(range(self.num_frames_per_video))
            replacement = self._build_video_replacement(
                n_tokens, frames_indices, self.video_fps
            )
            return tf.strings.regex_replace(prompts, vid_pattern, replacement)

        # Per-sample metadata provided via the self.video_metadata attribute.
        # Cannot be passed through the @preprocessing_function input dict
        # because the decorator converts dict values to tensors, losing the
        # Python list type.
        if not isinstance(video_metadata, (list, tuple)):
            video_metadata = [video_metadata]
        vid_re = re.compile(vid_pattern)
        prompts_list = [
            p.decode("utf-8") if isinstance(p, bytes) else str(p)
            for p in prompts.numpy()
        ]
        for b, meta in enumerate(video_metadata):
            frames_indices = meta["frames_indices"]
            fps = meta.get("fps", self.video_fps)
            rep = self._build_video_replacement(n_tokens, frames_indices, fps)
            prompts_list[b] = vid_re.sub(rep, prompts_list[b], count=1)
        return tf.constant(prompts_list, dtype=tf.string)

    @preprocessing_function
    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        sequence_length = sequence_length or self.sequence_length

        # === Input extraction and validation ===
        prompts, responses = x["prompts"], x["responses"]
        tf.debugging.assert_shapes([(prompts, ("N",)), (responses, ("N",))])

        batched = True
        if isinstance(prompts, str):
            batched = False
            prompts = [prompts]
            responses = [responses]
        if isinstance(prompts, tf.Tensor) and len(prompts.shape) == 0:
            batched = False
            prompts = tf.expand_dims(prompts, axis=0)
            responses = tf.expand_dims(responses, axis=0)

        pixel_values = x.get("pixel_values", None)
        pixel_position_ids = x.get("pixel_position_ids", None)

        images = x.get("images", None)
        videos = x.get("videos", None)
        audio = x.get("audio", None)
        audio_mel = x.get("audio_mel", None)
        audio_mel_mask = x.get("audio_mel_mask", None)

        if self.text_only_model and audio is not None:
            raise ValueError(
                "The initialized preprocessor/model is text-only, but "
                "`audio` is not `None`."
            )

        if self.image_converter is not None:
            num_tokens = self.num_vision_tokens_per_image
            img_pattern = re.escape(self.image_placeholder)
            prompts = tf.strings.regex_replace(
                prompts,
                img_pattern,
                f"{self.start_of_image_token}"
                + self.image_placeholder * num_tokens
                + f"{self.end_of_image_token}",
            )

        if self.video_converter is not None:
            prompts = self._expand_video_prompt(prompts, videos)

        if self.audio_converter is not None:
            if audio is not None and audio_mel is None:
                audio_mel, audio_mel_mask = self._preprocess_audio(
                    audio, batched
                )
                output_lengths = tf.reduce_sum(audio_mel_mask, axis=[1, 2])
                exact_tokens = (output_lengths + 3) // 4
                num_audio_tokens = tf.reduce_max(exact_tokens)
                num_audio_tokens = tf.maximum(num_audio_tokens, 1)

                repeated_placeholders = tf.repeat(
                    self.audio_placeholder, num_audio_tokens
                )
                joined_placeholders = tf.strings.reduce_join(
                    repeated_placeholders, axis=0
                )
                replacement = tf.strings.join(
                    [
                        self.start_of_audio_token,
                        joined_placeholders,
                        self.end_of_audio_token,
                    ]
                )
            else:
                replacement = self.audio_placeholder

            prompts = tf.strings.regex_replace(
                prompts,
                re.escape(self.audio_placeholder),
                replacement,
            )

        # Tokenise the inputs.
        prompts = self.tokenizer(prompts)
        responses = self.tokenizer(responses)

        token_ids, segment_ids = self.packer(
            (prompts, responses),
            sequence_length=sequence_length + 1,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        response_mask = segment_ids == 1
        padding_mask = token_ids != self.tokenizer.pad_token_id

        audio_indices = None
        audio_mask = None

        if (
            audio is not None or audio_mel is not None
        ) and self.audio_converter is not None:
            # audio_mel and audio_mel_mask were calculated earlier!

            # Clip audio_mel to match the number of available placeholders.
            placeholder_id = self._audio_placeholder_id
            is_placeholder = token_ids == placeholder_id
            num_placeholders = tf.reduce_sum(
                tf.cast(is_placeholder, tf.int32), axis=1
            )

            # Max frames needed: 4 * num_placeholders
            max_frames = tf.reduce_max(num_placeholders) * 4

            # Clip audio_mel to max_frames
            audio_mel = audio_mel[:, :, :max_frames, :]
            audio_mel_mask = audio_mel_mask[:, :, :max_frames]

            # Dynamically clip unused placeholders to remove the gap!
            # Use ceil division to match audio encoder subsampling (stride 4)
            exact_tokens = (tf.reduce_sum(audio_mel_mask, axis=[1, 2]) + 3) // 4

            placeholder_counts = tf.cumsum(
                tf.cast(is_placeholder, tf.int32), axis=1
            )
            placeholder_counts = tf.where(is_placeholder, placeholder_counts, 0)

            exact_tokens_expanded = tf.expand_dims(
                tf.cast(exact_tokens, tf.int32), axis=1
            )
            is_unused = is_placeholder & (
                placeholder_counts > exact_tokens_expanded
            )
            keep_mask = ~is_unused

            ragged_t_ids = tf.ragged.boolean_mask(token_ids, keep_mask)
            ragged_p_mask = tf.ragged.boolean_mask(padding_mask, keep_mask)
            ragged_s_ids = tf.ragged.boolean_mask(segment_ids, keep_mask)

            batch_size = tf.shape(token_ids)[0]
            seq_len = tf.shape(token_ids)[1]

            token_ids = ragged_t_ids.to_tensor(
                default_value=self.tokenizer.pad_token_id,
                shape=[batch_size, seq_len],
            )
            padding_mask = ragged_p_mask.to_tensor(
                default_value=False,
                shape=[batch_size, seq_len],
            )
            segment_ids = ragged_s_ids.to_tensor(
                default_value=0,
                shape=[batch_size, seq_len],
            )

            response_mask = segment_ids == 1
            audio_mask = token_ids == self._audio_placeholder_id
            audio_indices = self._get_audio_indices(audio_mask)

        # === Text-only Model ===
        if self.text_only_model:
            seq_len = tf.shape(token_ids)[1] - 1
            pos_ids = tf.range(seq_len, dtype=tf.int32)[tf.newaxis, :]
            pos_ids = tf.tile(pos_ids, [tf.shape(token_ids)[0], 1])
            x = {
                "token_ids": token_ids[..., :-1],
                "padding_mask": padding_mask[..., :-1],
                "position_ids": pos_ids,
            }
            y = token_ids[..., 1:]
            sample_weight = response_mask[..., 1:]

            if not batched:
                x["token_ids"] = tf.squeeze(x["token_ids"], axis=0)
                x["padding_mask"] = tf.squeeze(x["padding_mask"], axis=0)
                y = tf.squeeze(y, axis=0)
                sample_weight = tf.squeeze(sample_weight, axis=0)

            return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

        batch_size = tf.shape(prompts)[0]

        # === Vision & Audio processing ===
        if images is not None and self.image_converter is not None:
            vision_dict = self._preprocess_images(images, batched)
            pixel_values = vision_dict["pixel_values"]
            pixel_position_ids = vision_dict["pixel_position_ids"]
            vision_mask = token_ids == self.tokenizer.image_placeholder_id
        elif videos is not None and self.video_converter is not None:
            videos_dict = self._preprocess_videos(videos, batched)
            pixel_values = videos_dict["pixel_values"]
            pixel_position_ids = videos_dict["pixel_position_ids"]
            vision_mask = token_ids == self.tokenizer.video_placeholder_id
        elif pixel_values is not None:
            pixel_values = (
                pixel_values if batched else tf.expand_dims(pixel_values, 0)
            )
            pixel_position_ids = (
                pixel_position_ids
                if batched
                else tf.expand_dims(pixel_position_ids, 0)
            )
            vision_mask = token_ids == self.tokenizer.image_placeholder_id
        else:
            if self.image_converter is not None:
                patch_dim = self.image_converter.patch_size**2 * 3
                pixel_values = tf.ones(
                    [batch_size, 0, 0, patch_dim], dtype="float32"
                )
                pixel_position_ids = tf.zeros(
                    [batch_size, 0, 0, 2], dtype="int32"
                )
            else:
                pixel_values = None
                pixel_position_ids = None
            vision_mask = tf.zeros_like(token_ids, dtype=bool)

        # Audio variables are now pre-computed and clipped above if needed.

        return self._format_output(
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
            token_ids=token_ids,
            vision_mask=vision_mask,
            response_mask=response_mask,
            padding_mask=padding_mask,
            audio_mel=audio_mel,
            audio_mel_mask=audio_mel_mask,
            audio_indices=audio_indices,
            audio_mask=audio_mask,
            return_labels=True,
            text_only_input=False,
            batched=batched,
        )

    @preprocessing_function
    def generate_preprocess(
        self,
        x,
        sequence_length=None,
    ):
        """Convert strings to integer token input for generation."""
        if not self.built:
            self.build(None)

        if isinstance(x, dict):
            pixel_values = x.get("pixel_values", None)
            pixel_position_ids = x.get("pixel_position_ids", None)
            images = x.get("images", None)
            videos = x.get("videos", None)
            audio = x.get("audio", None)
            responses = x.get("responses", None)
            prompts = x["prompts"]
        else:
            pixel_values = None
            pixel_position_ids = None
            images = None
            videos = None
            audio = None
            responses = None
            prompts = x

        batched = True
        if isinstance(prompts, str):
            batched = False
            prompts = [prompts]
            if responses is not None:
                responses = [responses]
        if isinstance(prompts, tf.Tensor) and len(prompts.shape) == 0:
            batched = False
            prompts = tf.expand_dims(prompts, axis=0)
            if responses is not None:
                responses = tf.expand_dims(responses, axis=0)

        if self.text_only_model and (
            pixel_values is not None or images is not None or audio is not None
        ):
            raise ValueError(
                "The initialized preprocessor/model is text-only, but "
                "`images`/`pixel_values` or `audio` is not `None`."
            )

        if self.image_converter is not None:
            num_tokens = self.num_vision_tokens_per_image
            img_pattern = re.escape(self.image_placeholder)
            prompts = tf.strings.regex_replace(
                prompts,
                img_pattern,
                f"{self.start_of_image_token}"
                + self.image_placeholder * num_tokens
                + f"{self.end_of_image_token}",
            )

        if self.video_converter is not None:
            prompts = self._expand_video_prompt(prompts, videos)

        audio_mel = None
        audio_mel_mask = None
        replacement = self.audio_placeholder

        if audio is not None and self.audio_converter is not None:
            audio_mel, audio_mel_mask = self._preprocess_audio(audio, batched)
            # audio_mel_mask is (B, 1, Seq)
            output_lengths = tf.reduce_sum(audio_mel_mask, axis=[1, 2])
            exact_tokens = (output_lengths + 3) // 4

            num_audio_tokens = tf.reduce_max(exact_tokens)
            num_audio_tokens = tf.maximum(num_audio_tokens, 1)

            repeated_placeholders = tf.repeat(
                self.audio_placeholder, num_audio_tokens
            )
            joined_placeholders = tf.strings.reduce_join(
                repeated_placeholders, axis=0
            )

            replacement = tf.strings.join(
                [
                    self.start_of_audio_token,
                    joined_placeholders,
                    self.end_of_audio_token,
                ]
            )

        if self.audio_converter is not None:
            # Replace a single <|audio|> placeholder with the full expanded
            # sequence: <|audio><|audio|>*N<audio|>. Mirrors how image
            # expansion works (user writes <|image|>, gets wrapped).
            prompts = tf.strings.regex_replace(
                prompts,
                re.escape(self.audio_placeholder),
                replacement,
            )

        prompts = self.tokenizer(prompts)

        if responses is not None:
            responses = self.tokenizer(responses)
            segments = (prompts, responses)
        else:
            segments = (prompts,)

        token_ids, segment_ids = self.packer(
            segments,
            sequence_length=sequence_length or self.sequence_length,
            add_end_value=False,
        )
        response_mask = segment_ids == 1
        padding_mask = token_ids != self.tokenizer.pad_token_id

        # === Text Model ===
        if self.text_only_model:
            return {
                "token_ids": (
                    token_ids if batched else tf.squeeze(token_ids, axis=0)
                ),
                "padding_mask": (
                    padding_mask
                    if batched
                    else tf.squeeze(padding_mask, axis=0)
                ),
            }

        batch_size = tf.shape(prompts)[0]

        # === Vision & Audio processing ===
        if images is not None and self.image_converter is not None:
            vision_dict = self._preprocess_images(images, batched)
            pixel_values = vision_dict["pixel_values"]
            pixel_position_ids = vision_dict["pixel_position_ids"]
            vision_mask = token_ids == self.tokenizer.image_placeholder_id
        elif videos is not None and self.video_converter is not None:
            videos_dict = self._preprocess_videos(videos, batched)
            pixel_values = videos_dict["pixel_values"]
            pixel_position_ids = videos_dict["pixel_position_ids"]
            vision_mask = token_ids == self.tokenizer.video_placeholder_id
        elif pixel_values is not None:
            pixel_values = (
                pixel_values if batched else tf.expand_dims(pixel_values, 0)
            )
            pixel_position_ids = (
                pixel_position_ids
                if batched
                else tf.expand_dims(pixel_position_ids, 0)
            )
            vision_mask = token_ids == self.tokenizer.image_placeholder_id
        else:
            if self.image_converter is not None:
                patch_dim = self.image_converter.patch_size**2 * 3
                pixel_values = tf.ones(
                    [batch_size, 0, 0, patch_dim], dtype="float32"
                )
                pixel_position_ids = tf.zeros(
                    [batch_size, 0, 0, 2], dtype="int32"
                )
            else:
                pixel_values = None
                pixel_position_ids = None
            vision_mask = tf.zeros_like(token_ids, dtype=bool)

        audio_indices = None
        audio_mask = None

        if audio is not None and self.audio_converter is not None:
            # audio_mel and audio_mel_mask were calculated earlier in
            # generate_preprocess.
            placeholder_id = tf.cast(
                self._audio_placeholder_id, token_ids.dtype
            )

            # Identify all audio placeholder positions.
            is_placeholder = tf.equal(token_ids, placeholder_id)

            # Use cumsum to rank placeholders per sample.
            placeholder_counts = tf.cumsum(
                tf.cast(is_placeholder, tf.int32), axis=1
            )
            placeholder_counts = tf.where(is_placeholder, placeholder_counts, 0)

            # Create a mask to identify unused placeholders.
            exact_tokens_expanded = tf.expand_dims(
                tf.cast(exact_tokens, tf.int32), axis=1
            )
            is_unused = is_placeholder & (
                placeholder_counts > exact_tokens_expanded
            )

            # Remove unused placeholders to keep the sequence compact,
            # consistent with the training (call) path.
            keep_mask = ~is_unused
            batch_size = tf.shape(token_ids)[0]
            seq_len = tf.shape(token_ids)[1]

            ragged_t_ids = tf.ragged.boolean_mask(token_ids, keep_mask)
            ragged_p_mask = tf.ragged.boolean_mask(padding_mask, keep_mask)
            ragged_v_mask = tf.ragged.boolean_mask(vision_mask, keep_mask)
            ragged_s_ids = tf.ragged.boolean_mask(segment_ids, keep_mask)

            token_ids = ragged_t_ids.to_tensor(
                default_value=self.tokenizer.pad_token_id,
                shape=[batch_size, seq_len],
            )
            padding_mask = ragged_p_mask.to_tensor(
                default_value=False,
                shape=[batch_size, seq_len],
            )
            vision_mask = ragged_v_mask.to_tensor(
                default_value=False,
                shape=[batch_size, seq_len],
            )
            segment_ids = ragged_s_ids.to_tensor(
                default_value=0,
                shape=[batch_size, seq_len],
            )
            response_mask = segment_ids == 1

            audio_mask = tf.equal(token_ids, placeholder_id)
            audio_indices = self._get_audio_indices(audio_mask)

        return self._format_output(
            token_ids=token_ids,
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
            vision_mask=vision_mask,
            response_mask=response_mask,
            padding_mask=padding_mask,
            audio_mel=audio_mel,
            audio_mel_mask=audio_mel_mask,
            audio_indices=audio_indices,
            audio_mask=audio_mask,
            return_labels=False,
            text_only_input=False,
            batched=batched,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_converter": None
                if self.image_converter is None
                else keras.layers.serialize(self.image_converter),
                "audio_converter": None
                if self.audio_converter is None
                else keras.layers.serialize(self.audio_converter),
                "video_converter": None
                if self.video_converter is None
                else keras.layers.serialize(self.video_converter),
                "num_vision_tokens_per_image": self.num_vision_tokens_per_image,
                "max_images_per_prompt": self.max_images_per_prompt,
                "num_audio_tokens_per_clip": self.num_audio_tokens_per_clip,
                "max_audio_clips_per_prompt": self.max_audio_clips_per_prompt,
                "audio_input_feat_size": self.audio_input_feat_size,
                "num_frames_per_video": self.num_frames_per_video,
                "num_vision_tokens_per_frame": self.num_vision_tokens_per_frame,
                "video_fps": self.video_fps,
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
                "audio_converter": None
                if config.get("audio_converter") is None
                else keras.layers.deserialize(config["audio_converter"]),
                "video_converter": None
                if config.get("video_converter") is None
                else keras.layers.deserialize(config["video_converter"]),
            }
        )
        return super().from_config(config)

    @preprocessing_function
    def generate_postprocess(
        self,
        x,
    ):
        """Convert integer token output to strings for generation."""
        if not self.built:
            self.build(None)

        token_ids = keras.ops.convert_to_numpy(x["token_ids"])
        padding_mask = keras.ops.convert_to_numpy(x["padding_mask"])
        ids_to_strip = self.tokenizer.special_token_ids

        # Do not strip the SoI token — it is provided by the user.
        if self.tokenizer.start_of_image_token_id in ids_to_strip:
            ids_to_strip.remove(self.tokenizer.start_of_image_token_id)

        # Ensure audio soft tokens are stripped even when the tokenizer was
        # loaded without has_audio_tokens=True (audio_placeholder_id == -1).
        # In that case _audio_placeholder_id resolves the real vocab ID via
        # token_to_id(), and the other registered special tokens (start/end)
        # are looked up here for completeness.
        if self.audio_converter is not None:
            for tok_id in [
                self._audio_placeholder_id,
                self.tokenizer.token_to_id(self.start_of_audio_token),
                self.tokenizer.token_to_id(self.end_of_audio_token),
            ]:
                if tok_id is not None and tok_id not in ids_to_strip:
                    ids_to_strip.append(tok_id)

        token_ids = strip_to_ragged(token_ids, padding_mask, ids_to_strip)
        output = self.tokenizer.detokenize(token_ids)

        # Safety net: strip residual audio token strings that may survive if
        # the SentencePiece model encodes them as byte-fallback pieces rather
        # than dedicated vocab entries (making ID-based stripping ineffective).
        if self.audio_converter is not None:
            for tok_str in [
                self.audio_placeholder,  # "<|audio|>"
                self.start_of_audio_token,  # "<|audio>"
                self.end_of_audio_token,  # "<audio|>"
            ]:
                if tok_str is not None:
                    output = tf.strings.regex_replace(
                        output, re.escape(tok_str), ""
                    )
        return output

    @property
    def _audio_placeholder_id(self):
        """Real vocab ID for the audio soft token.

        `tokenizer.audio_placeholder_id` is -1 when the tokenizer was loaded
        with `has_audio_tokens=False` (the default). In that case the token
        still exists in the sentencepiece vocabulary, so we fall back to a
        direct vocab lookup so that audio_mask and stripping work correctly.
        """
        tok_id = self.tokenizer.audio_placeholder_id
        if tok_id == -1 and self.audio_placeholder is not None:
            tok_id = self.tokenizer.token_to_id(self.audio_placeholder)
        return tok_id

    @property
    def _video_placeholder_id(self):
        """Real vocab ID for the video soft token."""
        tok_id = self.tokenizer.video_placeholder_id
        if tok_id == -1 and self.video_placeholder is not None:
            tok_id = self.tokenizer.token_to_id(self.video_placeholder)
        return tok_id

    @property
    def max_images_per_prompt(self):
        return self._max_images_per_prompt

    @max_images_per_prompt.setter
    def max_images_per_prompt(self, value):
        self._max_images_per_prompt = value

    @property
    def num_audio_tokens_per_clip(self):
        return self._num_audio_tokens_per_clip

    @num_audio_tokens_per_clip.setter
    def num_audio_tokens_per_clip(self, value):
        self._num_audio_tokens_per_clip = value
