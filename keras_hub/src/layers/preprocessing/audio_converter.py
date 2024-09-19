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
from keras_hub.src.utils.preset_utils import AUDIO_CONVERTER_CONFIG_FILE
from keras_hub.src.utils.preset_utils import builtin_presets
from keras_hub.src.utils.preset_utils import find_subclass
from keras_hub.src.utils.preset_utils import get_preset_loader
from keras_hub.src.utils.preset_utils import save_serialized_object
from keras_hub.src.utils.python_utils import classproperty


@keras_hub_export("keras_hub.layers.AudioConverter")
class AudioConverter(PreprocessingLayer):
    """Convert raw audio for models that support audio input.

    This class converts from raw audio tensors of any length, to preprocessed
    audio for pretrained model inputs. It is meant to be a convenient way to
    write custom preprocessing code that is not model specific. This layer
    should be instantiated via the `from_preset()` constructor, which will
    create the correct subclass of this layer for the model preset.

    The layer will take as input a raw audio tensor with shape `(batch_size,
    num_samples)`, and output a preprocessed audio input for modeling. The exact
    structure of the preprocessed input will vary per model. Preprocessing
    will often include computing a spectogram of the raw audio signal.

    Examples:
    ```python
    # Load an audio converter from a preset.
    converter = keras_hub.layers.AudioConverter.from_preset("whisper_base_en")
    # Convert some raw audio input.
    converter(np.ones(2, 1_000))
    ```
    """

    backbone_cls = None

    def audio_shape(self):
        """Returns the preprocessed size of a single audio sample."""
        return (None,)

    @classproperty
    def presets(cls):
        """List built-in presets for an `AudioConverter` subclass."""
        return builtin_presets(cls)

    @classmethod
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        """Instantiate a `keras_hub.layers.AudioConverter` from a model preset.

        A preset is a directory of configs, weights and other file assets used
        to save and load a pre-trained model. The `preset` can be passed as
        one of:

        1. a built-in preset identifier like `'whisper_base_en'`
        2. a Kaggle Models handle like
           `'kaggle://user/whisper/keras/whisper_base_en'`
        3. a Hugging Face handle like `'hf://user/whisper_base_en'`
        4. a path to a local preset directory like `'./whisper_base_en'`

        You can run `cls.presets.keys()` to list all built-in presets available
        on the class.

        This constructor can be called in one of two ways. Either from the base
        class like `keras_hub.models.AudioConverter.from_preset()`, or from a
        model class like `keras_hub.models.WhisperAudioConverter.from_preset()`.
        If calling from the base class, the subclass of the returning object
        will be inferred from the config in the preset directory.

        Args:
            preset: string. A built-in preset identifier, a Kaggle Models
                handle, a Hugging Face handle, or a path to a local directory.
            load_weights: bool. If `True`, the weights will be loaded into the
                model architecture. If `False`, the weights will be randomly
                initialized.

        Examples:
        ```python
        # Load an audio converter from a preset.
        converter = keras_hub.layers.AudioConverter.from_preset(
            "whisper_base_en"
        )
        # Convert some raw mono channel audio input.
        converter(np.ones(2, 1_000))
        ```
        """
        loader = get_preset_loader(preset)
        backbone_cls = loader.check_backbone_class()
        if cls.backbone_cls != backbone_cls:
            cls = find_subclass(preset, cls, backbone_cls)
        return loader.load_audio_converter(cls, **kwargs)

    def save_to_preset(self, preset_dir):
        """Save audio converter to a preset directory.

        Args:
            preset_dir: The path to the local model preset directory.
        """
        save_serialized_object(
            self,
            preset_dir,
            config_file=AUDIO_CONVERTER_CONFIG_FILE,
        )
