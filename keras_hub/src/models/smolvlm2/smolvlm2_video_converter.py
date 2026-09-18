from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.video_converter import VideoConverter
from keras_hub.src.models.smolvlm2.smolvlm2_backbone import SmolVLM2Backbone
from keras_hub.src.models.smolvlm2.smolvlm2_image_converter import (
    SmolVLM2ImageConverter,
)
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.layers.SmolVLM2VideoConverter")
class SmolVLM2VideoConverter(VideoConverter):
    """Video converter for SmolVLM2 models.

    This layer processes video inputs by uniformly sampling frames and
    passing each frame through ``SmolVLM2ImageConverter`` with
    ``do_image_splitting=False``.

    Each frame is resized to ``(max_image_size, max_image_size)``,
    rescaled, and normalized — identical to an unsplit single image.

    The output is a dict with:
    - ``"pixel_values"``: float32 tensor of shape
      ``(num_frames, max_image_size, max_image_size, 3)``.
    - ``"num_frames"``: int32 scalar. Number of sampled frames.

    Args:
        max_image_size: int. Side length of each frame after resizing.
            Default 512 (from HF's `max_image_size.longest_edge`).
        size: int. Longest edge for initial resize before squaring.
            Default 2048 (from HF's `video_sampling.video_size`).
        num_frames: int. Maximum number of frames to sample. Frames are
            sampled uniformly over the clip. Default 64 (from HF's
            `video_sampling.max_frames`).
        fps: int or float. Frame rate the sampled frames are assumed to
            have. Only used to build per-frame timestamps in
            `SmolVLM2CausalLMPreprocessor`; it does not affect sampling
            here. Default 1 (from HF's `video_sampling.fps`).
    """

    backbone_cls = SmolVLM2Backbone

    def __init__(
        self,
        max_image_size=512,
        size=2048,
        num_frames=64,
        fps=1,
        interpolation="bicubic",
        antialias=True,
        **kwargs,
    ):
        # Pop image-converter-specific kwargs before passing to super.
        scale = kwargs.pop("scale", None)
        offset = kwargs.pop("offset", None)
        super().__init__(scale=scale, offset=offset, **kwargs)
        # Replace the generic converter created by `VideoConverter` with
        # the SmolVLM2 one, so there is a single frame converter.
        self.image_converter = SmolVLM2ImageConverter(
            max_image_size=max_image_size,
            size=size,
            do_image_splitting=False,
            scale=scale,
            offset=offset,
            interpolation=interpolation,
            antialias=antialias,
        )
        self.max_image_size = max_image_size
        self.size = size
        self.num_frames = num_frames
        self.fps = fps
        self.interpolation = interpolation
        self.antialias = antialias

    @property
    def frame_converter(self):
        """The `SmolVLM2ImageConverter` applied to each sampled frame."""
        return self.image_converter

    @preprocessing_function
    def call(self, inputs):
        """Process a video into per-frame pixel values.

        Args:
            inputs: uint8 or float32 tensor `(T, H, W, 3)` with pixel
                values in `[0, 255]`.

        Returns:
            dict with `"pixel_values"` `(num_sampled, max_image_size,
            max_image_size, 3)` and `"num_frames"` int32 scalar.
        """
        if len(inputs.shape) != 4:
            raise ValueError(
                "`SmolVLM2VideoConverter` expects a single video of shape "
                "`(num_frames, height, width, channels)`. Received inputs "
                f"with shape {tuple(inputs.shape)}."
            )
        video = ops.cast(inputs, "float32")
        total_frames = video.shape[0]
        if total_frames is None:
            raise ValueError(
                "`SmolVLM2VideoConverter` needs a static frame count to "
                "sample frames uniformly. Received a video with shape "
                f"{tuple(inputs.shape)}."
            )

        # Uniform frame sampling.
        sample_count = min(total_frames, self.num_frames)
        if sample_count < total_frames:
            indices = ops.cast(
                ops.linspace(0, total_frames - 1, sample_count), "int32"
            )
            video = ops.take(video, indices, axis=0)

        # All frames share the same target size, so resize them in one
        # batched call instead of looping frame by frame.
        pixel_values = self.image_converter(video)["pixel_values"]

        return {
            "pixel_values": pixel_values,
            "num_frames": ops.convert_to_tensor(sample_count, dtype="int32"),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_image_size": self.max_image_size,
                "size": self.size,
                "num_frames": self.num_frames,
                "fps": self.fps,
                "interpolation": self.interpolation,
                "antialias": self.antialias,
            }
        )
        return config
