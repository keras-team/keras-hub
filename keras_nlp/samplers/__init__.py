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

from tensorflow import keras

from keras_nlp.samplers.beam_sampler import BeamSampler
from keras_nlp.samplers.greedy_sampler import GreedySampler
from keras_nlp.samplers.sampler import Sampler
from keras_nlp.samplers.top_k_sampler import TopKSampler
from keras_nlp.samplers.top_p_sampler import TopPSampler


def serialize(sampler):
    return keras.utils.serialize_keras_object(sampler)


def deserialize(config, custom_objects=None):
    """Return a `Sampler` object from its config."""
    all_classes = {
        "beam": BeamSampler,
        "greedy": GreedySampler,
        "top_k": TopKSampler,
        "top_p": TopPSampler,
    }
    return keras.utils.deserialize_keras_object(
        config,
        module_objects=all_classes,
        custom_objects=custom_objects,
        printable_module_name="samplers",
    )


def get(identifier):
    """Retrieve a KerasNLP sampler by the identifier.

    The `identifier` may be the string name of a sampler class or class.

    >>> identifier = 'greedy'
    >>> sampler = keras_nlp.samplers.get(identifier)

    You can also specify `config` of the sampler to this function by passing
    dict containing `class_name` and `config` as an identifier. Also note that
    the `class_name` must map to a `Sampler` class.

    >>> cfg = {'class_name': 'keras_nlp>GreedySampler', 'config': {}}
    >>> sampler = keras_nlp.samplers.get(cfg)

    In the case that the `identifier` is a class, this method will return a new
    instance of the class by its constructor.

    Args:
        identifier: String or dict that contains the sampler name or
            configurations.

    Returns:
        Sampler instance base on the input identifier.

    Raises:
        ValueError: If the input identifier is not a supported type or in a bad
            format.
    """

    if identifier is None:
        return None
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, str):
        if not identifier.islower():
            raise KeyError(
                "`keras_nlp.samplers.get()` must take a lowercase string "
                f"identifier, but received: {identifier}."
            )
        return deserialize(identifier)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError(
            "Could not interpret sampler identifier: " + str(identifier)
        )
