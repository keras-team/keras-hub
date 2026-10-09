from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.embedding_gemma2.embedding_gemma2_backbone import (
    EmbeddingGemma2Backbone,
)
from keras_hub.src.models.embedding_gemma2.embedding_gemma2_image_converter import (  # noqa: E501
    EmbeddingGemma2ImageConverter,
)
from keras_hub.src.models.gemma4.gemma4_video_converter import (
    Gemma4VideoConverter,
)


@keras_hub_export("keras_hub.models.EmbeddingGemma2VideoConverter")
class EmbeddingGemma2VideoConverter(Gemma4VideoConverter):
    """Video converter for EmbeddingGemma2.

    This layer handles video inputs by sampling frames and delegating to
    `EmbeddingGemma2ImageConverter` for frame-level processing.

    Unlike `Gemma4VideoConverter` which always samples exactly `num_frames`,
    this converter keeps all frames up to `max_frames`. If the video exceeds
    `max_frames`, it uniformly samples `max_frames` frames.
    """

    backbone_cls = EmbeddingGemma2Backbone

    def __init__(
        self,
        patch_size=16,
        max_soft_tokens=70,
        pooling_kernel_size=3,
        max_frames=32,
        **kwargs,
    ):
        super().__init__(
            patch_size=patch_size,
            max_soft_tokens=max_soft_tokens,
            pooling_kernel_size=pooling_kernel_size,
            num_frames=max_frames,
            **kwargs,
        )
        self.image_converter = EmbeddingGemma2ImageConverter(
            patch_size=patch_size,
            max_soft_tokens=max_soft_tokens,
            pooling_kernel_size=pooling_kernel_size,
            **kwargs,
        )
        self.max_frames = max_frames

    def _get_indices(self, total_frames):
        """Uniform sampling if exceeding max_frames, else keep all."""
        if total_frames <= self.max_frames:
            return ops.arange(total_frames, dtype="int32")
        else:
            import numpy as np

            return ops.cast(
                ops.convert_to_tensor(
                    np.linspace(
                        0.0,
                        total_frames - 1.0,
                        self.max_frames,
                    )
                ),
                "int32",
            )

    def call(self, inputs):
        # Standardize inputs to list of tensors
        if isinstance(inputs, list):
            videos = inputs
        elif hasattr(inputs, "shape") and len(inputs.shape) == 5:
            videos = [inputs[i] for i in range(inputs.shape[0])]
        else:
            videos = [inputs]

        all_pixel_values = []
        all_pixel_position_ids = []

        for video in videos:
            shape = ops.shape(video)
            total_frames = int(shape[0])

            indices = self._get_indices(total_frames)
            sampled_video = ops.take(video, indices, axis=0)

            image_outputs = self.image_converter(sampled_video)

            pixel_values = image_outputs["pixel_values"]
            pixel_position_ids = image_outputs["pixel_position_ids"]

            all_pixel_values.append(pixel_values)
            all_pixel_position_ids.append(pixel_position_ids)

        return {
            "pixel_values": all_pixel_values,
            "pixel_position_ids": all_pixel_position_ids,
        }

    @property
    def num_vision_tokens_per_image(self):
        return self.image_converter.num_vision_tokens_per_image

    def get_config(self):
        config = super().get_config()
        config.update({"max_frames": self.max_frames})
        if "num_frames" in config:
            del config["num_frames"]
        return config
