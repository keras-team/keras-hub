import keras
from keras import ops
from keras_hub.src.layers.preprocessing.video_converter import VideoConverter
from keras_hub.src.models.gemma4.gemma4_image_converter import Gemma4ImageConverter
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.utils.preset_utils import load_json

@keras_hub_export("keras_hub.layers.Gemma4VideoConverter")
class Gemma4VideoConverter(VideoConverter):
    """Video converter for Gemma4.

    This layer handles video inputs by sampling frames and delegating to
    `Gemma4ImageConverter` for frame-level processing.

    Args:
        patch_size: int. Size of each square patch in pixels. Defaults to
            `16`.
        max_soft_tokens: int. Maximum number of pooled soft tokens per video
            frame. Defaults to `70`.
        pooling_kernel_size: int. Spatial pooling kernel size applied after
            the vision encoder. Defaults to `3`.
        num_frames: int. Number of frames to uniformly sample from the video.
            Defaults to `32`.
    """

    def __init__(
        self,
        patch_size=16,
        max_soft_tokens=70,
        pooling_kernel_size=3,
        num_frames=32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_converter = Gemma4ImageConverter(
            patch_size=patch_size,
            max_soft_tokens=max_soft_tokens,
            pooling_kernel_size=pooling_kernel_size,
            **kwargs
        )
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.max_soft_tokens = max_soft_tokens
        self.pooling_kernel_size = pooling_kernel_size

    def call(self, inputs):
        # inputs can be a 5D tensor (batch, frames, height, width, channels)
        # or a list of videos!
        
        if isinstance(inputs, list):
            # Handle list of videos
            outputs = []
            for video in inputs:
                # video is likely a 4D tensor (frames, height, width, channels)
                out = self._process_single_video(video)
                outputs.append(out)
            return outputs
        else:
            # Assume 5D tensor (batch, frames, height, width, channels)
            shape = ops.shape(inputs)
            batch_size = shape[0]
            total_frames = shape[1]
            h, w, c = shape[2], shape[3], shape[4]
            
            # Sample frames
            indices = ops.cast(ops.linspace(0, total_frames - 1, self.num_frames), "int32")
            sampled_inputs = ops.take(inputs, indices, axis=1)
            
            # Flatten batch and temporal dims
            flat_inputs = ops.reshape(sampled_inputs, (batch_size * self.num_frames, h, w, c))
            
            # Process frames as images
            image_outputs = self.image_converter(flat_inputs)
            
            # Extract pixel_values and pixel_position_ids
            pixel_values = image_outputs["pixel_values"]
            pixel_position_ids = image_outputs["pixel_position_ids"]
            
            # Reshape back to 5D
            pv_shape = ops.shape(pixel_values)
            pos_shape = ops.shape(pixel_position_ids)
            
            pixel_values = ops.reshape(
                pixel_values,
                (batch_size, self.num_frames, pv_shape[1], pv_shape[2]),
            )
            pixel_position_ids = ops.reshape(
                pixel_position_ids,
                (batch_size, self.num_frames, pos_shape[1], pos_shape[2]),
            )
            
            return {
                "pixel_values": pixel_values,
                "pixel_position_ids": pixel_position_ids,
            }

    def _process_single_video(self, video):
        # video is 4D (frames, height, width, channels)
        shape = ops.shape(video)
        total_frames = shape[0]
        h, w, c = shape[1], shape[2], shape[3]
        
        # Sample frames
        indices = ops.cast(ops.linspace(0, total_frames - 1, self.num_frames), "int32")
        sampled_video = ops.take(video, indices, axis=0)
        
        # Process as images (batch dimension is num_frames!)
        image_outputs = self.image_converter(sampled_video)
        
        pixel_values = image_outputs["pixel_values"]
        pixel_position_ids = image_outputs["pixel_position_ids"]
        
        # Add batch dimension of 1!
        pixel_values = ops.expand_dims(pixel_values, 0)
        pixel_position_ids = ops.expand_dims(pixel_position_ids, 0)
        
        return {
            "pixel_values": pixel_values,
            "pixel_position_ids": pixel_position_ids,
        }


    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "max_soft_tokens": self.max_soft_tokens,
                "pooling_kernel_size": self.pooling_kernel_size,
                "num_frames": self.num_frames,
            }
        )
        return config
