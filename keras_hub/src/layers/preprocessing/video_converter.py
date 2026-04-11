import keras
from keras import ops
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.layers.preprocessing.preprocessing_layer import PreprocessingLayer
from keras_hub.src.utils.preset_utils import builtin_presets
from keras_hub.src.utils.preset_utils import find_subclass
from keras_hub.src.utils.preset_utils import get_preset_loader
from keras_hub.src.utils.preset_utils import get_preset_saver
from keras_hub.src.utils.python_utils import classproperty

@keras_hub_export("keras_hub.layers.VideoConverter")
class VideoConverter(PreprocessingLayer):
    """Base class for video preprocessing layers.

    `VideoConverter` tasks handle resizing and normalizing video inputs.
    It delegates to `ImageConverter` for frame-level processing.

    Args:
        image_size: The target size of the frames.
        scale: The scale factor to apply to pixels.
        offset: The offset to apply to pixels.
    """

    backbone_cls = None

    @classproperty
    def presets(cls):
        """List built-in presets for a `VideoConverter` subclass."""
        return builtin_presets(cls)

    @classmethod
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        """Instantiate a `keras_hub.layers.VideoConverter` from a model preset.

        A preset is a directory of configs, weights and other file assets used
        to save and load a pre-trained model. The `preset` can be passed as
        one of:

        1. a built-in preset identifier like `'gemma4_2b_it'`
        2. a Kaggle Models handle like
           `'kaggle://user/gemma4/keras/gemma4_2b_it'`
        3. a Hugging Face handle like `'hf://google/gemma-4-2b-it'`
        4. a path to a local preset directory like `'./gemma4_2b_it'`

        You can run `cls.presets.keys()` to list all built-in presets available
        on the class.

        This constructor can be called in one of two ways. Either from the base
        class like `keras_hub.models.VideoConverter.from_preset()`, or from a
        model class like `keras_hub.models.Gemma4VideoConverter.from_preset()`.
        If calling from the base class, the subclass of the returning object
        will be inferred from the config in the preset directory.

        Args:
            preset: string. A built-in preset identifier, a Kaggle Models
                handle, a Hugging Face handle, or a path to a local directory.

        Examples:
        ```python
        # Load a video converter from a preset.
        converter = keras_hub.layers.VideoConverter.from_preset(
            "hf://google/gemma-4-2b-it"
        )
        ```
        """
        loader = get_preset_loader(preset)
        backbone_cls = loader.check_backbone_class()
        if cls.backbone_cls != backbone_cls:
            cls = find_subclass(preset, cls, backbone_cls)
        return loader.load_video_converter(cls, **kwargs)

    def save_to_preset(self, preset_dir):
        """Save video converter to a preset directory.

        Args:
            preset_dir: The path to the local model preset directory.
        """
        saver = get_preset_saver(preset_dir)
        saver.save_video_converter(self)

    def __init__(
        self,
        image_size=None,
        scale=None,
        offset=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_converter = ImageConverter(
            image_size=image_size, scale=scale, offset=offset, **kwargs
        )
        self.image_size = image_size
        self.scale = scale
        self.offset = offset

    def call(self, inputs):
        if isinstance(inputs, list):
            # List of videos (each video is a 4D tensor or list of frames)
            outputs = []
            for video in inputs:
                out = self.image_converter(video)
                outputs.append(out)
            return outputs
        else:
            # Assume 5D tensor (batch, frames, height, width, channels)
            shape = ops.shape(inputs)
            batch_size = shape[0]
            num_frames = shape[1]
            h, w, c = shape[2], shape[3], shape[4]

            # Flatten batch and temporal dims
            flat_inputs = ops.reshape(inputs, (batch_size * num_frames, h, w, c))

            # Process frames as images
            flat_outputs = self.image_converter(flat_inputs)

            # Reshape back to 5D
            output_shape = ops.shape(flat_outputs)
            outputs = ops.reshape(
                flat_outputs,
                (batch_size, num_frames, output_shape[1], output_shape[2], output_shape[3]),
            )
            return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "scale": self.scale,
                "offset": self.offset,
            }
        )
        return config
