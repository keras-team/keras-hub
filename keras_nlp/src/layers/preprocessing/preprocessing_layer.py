# Copyright 2023 The KerasNLP Authors
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

import keras
from keras import tree

from keras_nlp.src.utils.tensor_utils import assert_tf_libs_installed
from keras_nlp.src.utils.tensor_utils import (
    convert_to_backend_tensor_or_python_list,
)

try:
    import tensorflow as tf
except ImportError:
    tf = None


class PreprocessingLayer(keras.layers.Layer):
    """Preprocessing layer base class."""

    def __init__(self, **kwargs):
        assert_tf_libs_installed(self.__class__.__name__)

        super().__init__(**kwargs)
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True
        # Most pre-preprocessing has no build.
        if not hasattr(self, "build"):
            self.built = True

    def get_build_config(self):
        return None

    def __call__(self, *args, **kwargs):
        # Always place on CPU for preprocessing, to avoid expensive back and
        # forth copies to GPU before the trainable model.
        with tf.device("cpu"):
            outputs = super().__call__(*args, **kwargs)

            # Jax and Torch lack native string and ragged types.
            # If we are running on those backends and not running with tf.data
            # (we are outside a tf.function), we covert all ragged and string
            # tensor to pythonic types.
            is_tf_backend = keras.config.backend() == "tensorflow"
            is_in_tf_graph = not tf.executing_eagerly()
            if not is_tf_backend and not is_in_tf_graph:
                outputs = tree.map_structure(
                    convert_to_backend_tensor_or_python_list, outputs
                )

        return outputs
