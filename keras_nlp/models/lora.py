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
import re

from keras_nlp.backend import config
from keras_nlp.backend import keras
from keras_nlp.layers.modeling.lora_dense import LoraDense
from keras_nlp.utils.keras_utils import replace_layer


def add_lora_layers(
    model,
    lora_layer_paths,
    trainable_weight_paths,
    rank,
    alpha,
):
    """Replace some dense layers in a model with lora layers."""
    if not config.multi_backend():
        raise ValueError(
            "`add_lora_layers` only works with multi-backend Keras. Set "
            "the `KERAS_BACKEND` environment variable to use this API."
        )

    if not model.built:
        raise ValueError(
            "`add_lora_layers` should only be called on a `built` model. "
            f"Recieved model={model}, which has not been built."
        )

    # Patch model to tie state before exiting stateless scope.
    for weight in model.weights:
        # Update `trainable` to match `trainable_weight_paths`.
        path = weight.path
        if any(re.search(pattern, path) for pattern in trainable_weight_paths):
            weight.trainable = True
        else:
            weight.trainable = False

    for layer in model._flatten_layers():
        allowed_layers = (keras.layers.Dense, keras.layers.EinsumDense)
        if not isinstance(layer, allowed_layers):
            continue

        # Strip the `/kernel` suffix to get the layer path only.
        path = layer.kernel.path.rsplit("/", maxsplit=1)[0]
        if not any(re.search(pattern, path) for pattern in lora_layer_paths):
            continue

        lora_layer = LoraDense(
            layer,
            rank=rank,
            alpha=alpha,
            freeze_kernel=False,
            freeze_bias=False,
        )

        if not replace_layer(model, layer, lora_layer):
            raise ValueError(
                f"Could not replace layer {layer} in `add_lora_layers()`. "
                "Only layers which are direct attributes of other layers "
                "can be replaced, e.g. `attention_layer.query_dense`. "
                "Update `lora_layer_paths` to target different layers. "
                f"Received: lora_layer_paths={lora_layer_paths}"
            )

    if model.compiled:
        # Clear any compiled functions, they must be recreated with new graph.
        cfg = model.get_compile_config()
        cfg = keras.saving.deserialize_keras_object(cfg)
        model.compile(**cfg)


def merge_lora_layers(model):
    """Merge lora updates back into regular dense layers."""
    # Combine all lora weight updates into the original dense kernels.
    for layer in model._flatten_layers():
        if not isinstance(layer, LoraDense):
            continue
        layer.merge_weights()
        replace_layer(model, layer, layer.inner_dense)

    # Unfreeze all weights.
    for weight in model.weights:
        weight.trainable = True
