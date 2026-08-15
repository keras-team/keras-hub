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
            Default 512 (from HF's ``max_image_size.longest_edge``).
        size: int. Longest edge for initial resize before squaring.
            Default 2048 (from HF's ``video_sampling.video_size``).
        num_frames: int. Maximum number of frames to sample. Default
            64 (from HF's ``video_sampling.max_frames``).
        fps: int or float. Target frames per second for sampling.
            Default 1 (from HF's ``video_sampling.fps``).
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
        # Internal image converter with splitting disabled.
        self.frame_converter = SmolVLM2ImageConverter(
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

    @preprocessing_function
    def call(self, inputs):
        """Process a video into per-frame pixel values.

        Args:
            inputs: uint8 or float32 tensor ``(T, H, W, 3)`` with pixel
                values in ``[0, 255]``.
        Returns:
            dict with ``"pixel_values"`` ``(num_sampled, max_image_size,
            max_image_size, 3)`` and ``"num_frames"`` int32 scalar.
        """
        video = ops.cast(inputs, "float32")
        total_frames = int(ops.shape(video)[0])

        # Uniform frame sampling.
        sample_count = min(total_frames, self.num_frames)
        if sample_count < total_frames:
            indices = ops.cast(
                ops.linspace(0, total_frames - 1, sample_count), "int32"
            )
            video = ops.take(video, indices, axis=0)
        actual_frames = int(ops.shape(video)[0])

        # Process each frame individually through the image converter.
        frame_outputs = []
        for i in range(actual_frames):
            frame = video[i]  # (H, W, 3)
            result = self.frame_converter(frame)
            # Each frame → (1, ms, ms, 3) since do_image_splitting=False.
            frame_outputs.append(result["pixel_values"])

        # Stack all frames: (num_frames, ms, ms, 3).
        # Each frame_output is (1, ms, ms, 3), so squeeze and re-stack.
        frames = [f[0] for f in frame_outputs]
        pixel_values = ops.stack(frames, axis=0)

        return {
            "pixel_values": pixel_values,
            "num_frames": ops.convert_to_tensor(actual_frames, dtype="int32"),
        }

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_image_size": self.max_image_size,
                "size": self.size,
                "num_frames": self.num_frames,
                "fps": self.fps,
            }
        )
        return config
