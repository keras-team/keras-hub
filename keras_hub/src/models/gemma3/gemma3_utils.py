from enum import Enum
from typing import Optional, Union, Tuple, Dict, List
import tensorflow as tf
from keras import ops
import logging
import numpy as np

import math
import itertools
import re

import PIL.Image
import PIL.ImageOps

logger = logging.getLogger(__name__)

class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )

class ChannelDimension(ExplicitEnum):
    FIRST = "channels_first"
    LAST = "channels_last"

def infer_channel_dimension_format(
    image: np.ndarray, num_channels: Optional[Union[int, Tuple[int, ...]]] = None
) -> ChannelDimension:
    """
    Infers the channel dimension format of `image`.

    Args:
        image (`np.ndarray`):
            The image to infer the channel dimension of.
        num_channels (`int` or `Tuple[int, ...]`, *optional*, defaults to `(1, 3)`):
            The number of channels of the image.

    Returns:
        The channel dimension of the image.
    """
    num_channels = num_channels if num_channels is not None else (1, 3)
    num_channels = (num_channels,) if isinstance(num_channels, int) else num_channels

    if image.ndim == 3:
        first_dim, last_dim = 0, 2
    elif image.ndim == 4:
        first_dim, last_dim = 1, 3
    else:
        raise ValueError(f"Unsupported number of image dimensions: {image.ndim}")

    image_shape = image.shape

    if image_shape[first_dim] in num_channels and image_shape[last_dim] in num_channels:
        logger.warning(
            f"The channel dimension is ambiguous. Got image shape {image.shape}. Assuming channels are the first dimension."
        )
        return ChannelDimension.FIRST
    elif image_shape[first_dim] in num_channels:
        return ChannelDimension.FIRST
    elif image_shape[last_dim] in num_channels:
        return ChannelDimension.LAST
    raise ValueError("Unable to infer channel dimension format")

def get_image_size(image: np.ndarray, channel_dim: ChannelDimension = None) -> Tuple[int, int]:
    """
    Returns the (height, width) dimensions of the image.

    Args:
        image (`np.ndarray`):
            The image to get the dimensions of.
        channel_dim (`ChannelDimension`, *optional*):
            Which dimension the channel dimension is in. If `None`, will infer the channel dimension from the image.

    Returns:
        A tuple of the image's height and width.
    """
    if channel_dim is None:
        channel_dim = infer_channel_dimension_format(image)

    image_shape = image.shape

    if channel_dim == ChannelDimension.FIRST:
        return image_shape[-2], image_shape[-1]
    elif channel_dim == ChannelDimension.LAST:
        return image_shape[-3], image_shape[-2]
    else:
        raise ValueError(f"Unsupported data format: {channel_dim}")

def pan_and_scan(
    image: np.ndarray,
    pan_and_scan_min_crop_size: int,
    pan_and_scan_max_num_crops: int,
    pan_and_scan_min_ratio_to_activate: float,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
):
    height, width = get_image_size(image)

    # Square or landscape image.
    if width >= height:
        # Only apply PaS if the image is sufficiently exaggerated
        if width / height < pan_and_scan_min_ratio_to_activate:
            return []

        # Select ideal number of crops close to the image aspect ratio and such that crop_size > min_crop_size.
        num_crops_w = int(math.floor(width / height + 0.5))  # Half round up rounding.
        num_crops_w = min(int(math.floor(width / pan_and_scan_min_crop_size)), num_crops_w)

        # Make sure the number of crops is in range [2, pan_and_scan_max_num_crops].
        num_crops_w = max(2, num_crops_w)
        num_crops_w = min(pan_and_scan_max_num_crops, num_crops_w)
        num_crops_h = 1

    # Portrait image.
    else:
        # Only apply PaS if the image is sufficiently exaggerated
        if height / width < pan_and_scan_min_ratio_to_activate:
            return []

        # Select ideal number of crops close to the image aspect ratio and such that crop_size > min_crop_size.
        num_crops_h = int(math.floor(height / width + 0.5))
        num_crops_h = min(int(math.floor(height / pan_and_scan_min_crop_size)), num_crops_h)

        # Make sure the number of crops is in range [2, pan_and_scan_max_num_crops].
        num_crops_h = max(2, num_crops_h)
        num_crops_h = min(pan_and_scan_max_num_crops, num_crops_h)
        num_crops_w = 1

    crop_size_w = int(math.ceil(width / num_crops_w))
    crop_size_h = int(math.ceil(height / num_crops_h))

    # Don't apply PaS if crop size is too small.
    if min(crop_size_w, crop_size_h) < pan_and_scan_min_crop_size:
        return []

    crop_positions_w = [crop_size_w * i for i in range(num_crops_w)]
    crop_positions_h = [crop_size_h * i for i in range(num_crops_h)]

    if input_data_format == ChannelDimension.LAST:
        image_crops = [
            image[pos_h : pos_h + crop_size_h, pos_w : pos_w + crop_size_w]
            for pos_h, pos_w in itertools.product(crop_positions_h, crop_positions_w)
        ]
    else:
        image_crops = [
            image[:, pos_h : pos_h + crop_size_h, pos_w : pos_w + crop_size_w]
            for pos_h, pos_w in itertools.product(crop_positions_h, crop_positions_w)
        ]

    return image_crops

def _process_images_for_pan_and_scan(
    images: np.ndarray,
    pan_and_scan_min_crop_size: int,
    pan_and_scan_max_num_crops: int,
    pan_and_scan_min_ratio_to_activate: float,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
):
    batched_pas_images_list = []
    num_crops = []

    for image in images:
        pas_images = pan_and_scan(
            image=image,
            pan_and_scan_min_crop_size=pan_and_scan_min_crop_size,
            pan_and_scan_max_num_crops=pan_and_scan_max_num_crops,
            pan_and_scan_min_ratio_to_activate=pan_and_scan_min_ratio_to_activate,
            input_data_format=input_data_format,
        )

        batched_pas_images_list.append([image] + pas_images)
        num_crops.append(len(pas_images))

    return batched_pas_images_list, num_crops

def do_pan_and_scan(
    inputs: dict,
    pan_and_scan_min_crop_size: int,
    pan_and_scan_max_num_crops: int,
    pan_and_scan_min_ratio_to_activate: float,
):

    crops_and_prompts = dict()
    crops_and_prompts['crops'] = []
    crops_and_prompts['modified_prompts'] = []
    images = inputs.get("images", None)
    prompts = inputs["prompts"]
    image_tag = "<img>"

    input_data_format = infer_channel_dimension_format(images[0][0])

    image = [
            _process_images_for_pan_and_scan(
                images=image,
                pan_and_scan_min_crop_size=pan_and_scan_min_crop_size,
                pan_and_scan_max_num_crops=pan_and_scan_max_num_crops,
                pan_and_scan_min_ratio_to_activate=pan_and_scan_min_ratio_to_activate,
                input_data_format=input_data_format,
            )
            for image in images
        ]

    images_and_crops_list = [images for images, _, in image]
    num_crops = [num_crops for _, num_crops in image]

    for batch_idx, (images_and_crops, prompt_text, num_of_crops) in enumerate(zip(images_and_crops_list, prompts, num_crops)):

        image_tag_indexes = [m.start() for m in re.finditer(image_tag, prompt_text)]

        if len(images_and_crops) != len(image_tag_indexes):
            raise ValueError(
                f"Prompt contained {len(image_tag_indexes)} image tokens but received {len(images_and_crops)} images."
            )

        for num, idx in reversed(list(zip(num_of_crops, image_tag_indexes))):
            if num:
                formatted_image_text = (
                    f"Here is the original image {image_tag} and here are some crops to help you see better "
                    + " ".join([image_tag] * num)
                )
                prompt_text = prompt_text[:idx] + formatted_image_text + prompt_text[idx + len(image_tag) :]

        crops_and_prompts['crops'].append(images_and_crops)
        crops_and_prompts['modified_prompts'].append(prompt_text)

    return crops_and_prompts

def to_pil_image(image, rescale=None):

      if isinstance(image, np.ndarray):
          if rescale is None:
              # rescale default to the array being of floating type.
              rescale = isinstance(image.flat[0], np.floating)
          # If the channel as been moved to first dim, we put it back at the end.
          if image.ndim == 3 and image.shape[0] in [1, 3]:
              image = image.transpose(1, 2, 0)
          if rescale:
              image = image * 255
          image = image.astype(np.uint8)
          return PIL.Image.fromarray(image)
      return image

def resize(image, resample: PIL.Image.Resampling = PIL.Image.Resampling.BILINEAR):
    height = 896
    width = 896
    size = (height, width)
    if not isinstance(image, PIL.Image.Image):
        image = to_pil_image(image)
    return image.resize(size, resample=resample)

