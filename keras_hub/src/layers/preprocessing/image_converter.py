# Copyright 2024 The KerasHub Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.preprocessing_layer import (
    PreprocessingLayer,
)
from keras_hub.src.utils.preset_utils import IMAGE_CONVERTER_CONFIG_FILE
from keras_hub.src.utils.preset_utils import builtin_presets
from keras_hub.src.utils.preset_utils import find_subclass
from keras_hub.src.utils.preset_utils import get_preset_loader
from keras_hub.src.utils.preset_utils import save_serialized_object
from keras_hub.src.utils.python_utils import classproperty


@keras_hub_export("keras_hub.layers.ImageConverter")
class ImageConverter(PreprocessingLayer):
    """Convert raw image for models that support image input.

    This class converts from raw images of any size, to preprocessed
    images for pretrained model inputs. It is meant to be a convenient way to
    write custom preprocessing code that is not model specific. This layer
    should be instantiated via the `from_preset()` constructor, which will
    create the correct subclass of this layer for the model preset.

    The layer will take as input a raw image tensor in the channels last or
    channels first format, and output a preprocessed image input for modeling.
    The exact structure of the output will vary per model, though in most cases
    this layer will simply resize the image to the size needed by the model
    input.

    Examples:
    ```python
    # Resize images for `"pali_gemma_3b_224"`.
    converter = keras_hub.layers.ImageConverter.from_preset("pali_gemma_3b_224")
    converter(np.ones(2, 512, 512, 3)) # Output shape: (2, 224, 224, 3)
    # Resize images for `"pali_gemma_3b_448"`.
    converter = keras_hub.layers.ImageConverter.from_preset("pali_gemma_3b_448")
    converter(np.ones(2, 512, 512, 3)) # Output shape: (2, 448, 448, 3)
    ```
    """

    backbone_cls = None

    def image_size(self):
        """Returns the default size of a single image."""
        return (None, None)

    @classproperty
    def presets(cls):
        """List built-in presets for an `ImageConverter` subclass."""
        return builtin_presets(cls)

    @classmethod
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        """Instantiate a `keras_hub.layers.ImageConverter` from a model preset.

        A preset is a directory of configs, weights and other file assets used
        to save and load a pre-trained model. The `preset` can be passed as
        one of:

        1. a built-in preset identifier like `'pali_gemma_3b_224'`
        2. a Kaggle Models handle like
           `'kaggle://user/paligemma/keras/pali_gemma_3b_224'`
        3. a Hugging Face handle like `'hf://user/pali_gemma_3b_224'`
        4. a path to a local preset directory like `'./pali_gemma_3b_224'`

        You can run `cls.presets.keys()` to list all built-in presets available
        on the class.

        This constructor can be called in one of two ways. Either from the base
        class like `keras_hub.models.ImageConverter.from_preset()`, or from a
        model class like
        `keras_hub.models.PaliGemmaImageConverter.from_preset()`. If calling
        from the base class, the subclass of the returning object will be
        inferred from the config in the preset directory.

        Args:
            preset: string. A built-in preset identifier, a Kaggle Models
                handle, a Hugging Face handle, or a path to a local directory.
            load_weights: bool. If `True`, the weights will be loaded into the
                model architecture. If `False`, the weights will be randomly
                initialized.

        Examples:
        ```python
        # Resize images for `"pali_gemma_3b_224"`.
        converter = keras_hub.layers.ImageConverter.from_preset(
            "pali_gemma_3b_224"
        )
        converter(np.ones(2, 512, 512, 3)) # Output shape: (2, 224, 224, 3)
        # Override arguments on the base class.
        converter = keras_hub.layers.ImageConverter.from_preset(
            "pali_gemma_3b_448",
            crop_to_aspect_ratio=False,
        )
        converter(np.ones(2, 512, 512, 3)) # (2, 448, 448, 3)
        ```
        """
        loader = get_preset_loader(preset)
        backbone_cls = loader.check_backbone_class()
        if cls.backbone_cls != backbone_cls:
            cls = find_subclass(preset, cls, backbone_cls)
        return loader.load_image_converter(cls, **kwargs)

    def save_to_preset(self, preset_dir):
        """Save image converter to a preset directory.

        Args:
            preset_dir: The path to the local model preset directory.
        """
        save_serialized_object(
            self,
            preset_dir,
            config_file=IMAGE_CONVERTER_CONFIG_FILE,
        )
