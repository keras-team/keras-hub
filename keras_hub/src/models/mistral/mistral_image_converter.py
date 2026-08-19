import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.mistral.mistral_backbone import MistralBackbone
from keras_hub.src.models.mistral.mistral_vision_encoder import (
    compute_pixtral_resize_size,
)

# CLIP normalization stats, in [0, 255] pixel-value units.
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


@keras_hub_export("keras_hub.layers.Mistral3ImageConverter")
class Mistral3ImageConverter(ImageConverter):
    """Converts raw images into `MistralBackbone`'s multimodal inputs.

    Each image is resized so its longest edge is at most `longest_edge`
    pixels (aspect ratio preserved), then rounded up to a `patch_size`
    multiple, matching HF's `PixtralImageProcessor`. Since every image in a
    call can resize to a different shape, resizing runs in a Python loop
    over `inputs` (a list of variable-size images) rather than through the
    base class's single-`Resizing`-layer `call()`.

    Args:
        longest_edge: int. The maximum size of an image's longer side after
            resizing. Defaults to `1540`.
        patch_size: int. The patch size each resized dimension must be a
            multiple of. Defaults to `14`.
        scale: float, tuple of floats, or `None`. Per-channel scale applied
            after resizing. Defaults to the CLIP normalization scale.
        offset: float, tuple of floats, or `None`. Per-channel offset
            applied after resizing. Defaults to the CLIP normalization
            offset.
    """

    backbone_cls = MistralBackbone

    def __init__(
        self,
        longest_edge=1540,
        patch_size=14,
        scale=None,
        offset=None,
        **kwargs,
    ):
        if scale is None:
            scale = [1.0 / 255.0 / s for s in _CLIP_STD]
        if offset is None:
            offset = [-m / s for m, s in zip(_CLIP_MEAN, _CLIP_STD)]
        # `image_size=None` skips the base class's single-size `Resizing`
        # sublayer: Pixtral's resize target is dynamic per image, so
        # resizing is instead done per image in `call()`.
        super().__init__(
            image_size=None,
            scale=scale,
            offset=offset,
            dtype="float32",
            **kwargs,
        )
        self.longest_edge = longest_edge
        self.patch_size = patch_size

    def call(self, inputs):
        resized_images = []
        image_sizes = []
        for image in inputs:
            image = ops.convert_to_numpy(image).astype("float32")
            height, width = image.shape[0], image.shape[1]
            resized_height, resized_width = compute_pixtral_resize_size(
                height, width, self.longest_edge, self.patch_size
            )
            # HF's default resample for Pixtral is bicubic.
            image = ops.image.resize(
                image,
                size=(resized_height, resized_width),
                interpolation="bicubic",
            )
            image = ops.convert_to_numpy(image)
            scale = np.array(self.scale, dtype="float32")
            offset = np.array(self.offset, dtype="float32")
            image = image * scale + offset
            # Channels-last `(H, W, 3)` -> channels-first `(3, H, W)`, to
            # match `MistralBackbone`'s `pixel_values` input layout.
            image = np.transpose(image, (2, 0, 1))
            resized_images.append(image)
            image_sizes.append((resized_height, resized_width))

        max_height = max(size[0] for size in image_sizes)
        max_width = max(size[1] for size in image_sizes)

        padded_images = []
        for image, (resized_height, resized_width) in zip(
            resized_images, image_sizes
        ):
            pad_height = max_height - resized_height
            pad_width = max_width - resized_width
            padded_images.append(
                np.pad(
                    image,
                    ((0, 0), (0, pad_height), (0, pad_width)),
                )
            )

        pixel_values = np.stack(padded_images, axis=0).astype("float32")
        image_sizes = np.array(image_sizes, dtype="int32")
        return pixel_values, image_sizes

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "longest_edge": self.longest_edge,
                "patch_size": self.patch_size,
            }
        )
        return config
