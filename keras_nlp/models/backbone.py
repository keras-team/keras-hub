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

import os

from keras_nlp.backend import keras
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring


@keras.saving.register_keras_serializable(package="keras_nlp")
class Backbone(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._token_embedding = None

    def __setattr__(self, name, value):
        # Work around torch setattr for properties.
        if name in ["token_embedding"]:
            return object.__setattr__(self, name, value)
        return super().__setattr__(name, value)

    @property
    def token_embedding(self):
        """A `keras.layers.Embedding` instance for embedding token ids.

        This layer integer token ids to the hidden dim of the model.
        """
        return self._token_embedding

    @token_embedding.setter
    def token_embedding(self, value):
        # Workaround tf.keras h5 checkpoint loading, which is sensitive to layer
        # count mismatches and does not deduplicate layers. This could go away
        # if we update our checkpoints to the newer `.weights.h5` format.
        self._setattr_tracking = False
        self._token_embedding = value
        self._setattr_tracking = True

    def get_config(self):
        # Don't chain to super here. The default `get_config()` for functional
        # models is nested and cannot be passed to our Backbone constructors.
        return {
            "name": self.name,
            "trainable": self.trainable,
        }

    @classmethod
    def from_config(cls, config):
        # The default `from_config()` for functional models will return a
        # vanilla `keras.Model`. We override it to get a subclass instance back.
        return cls(**config)

    @classproperty
    def presets(cls):
        return {}

    @classmethod
    def from_preset(
        cls,
        preset,
        load_weights=True,
        **kwargs,
    ):
        """Instantiate {{model_name}} model from preset architecture and weights.

        Args:
            preset: string. Must be one of "{{preset_names}}".
            load_weights: Whether to load pre-trained weights into model.
                Defaults to `True`.

        Examples:
        ```python
        # Load architecture and weights from preset
        model = keras_nlp.models.{{model_name}}.from_preset(
            "{{example_preset_name}}"
        )

        # Load randomly initialized model from preset architecture
        model = keras_nlp.models.{{model_name}}.from_preset(
            "{{example_preset_name}}",
            load_weights=False
        )
        ```
        """

        if not cls.presets:
            raise NotImplementedError(
                "No presets have been created for this class."
            )

        if preset not in cls.presets:
            raise ValueError(
                "`preset` must be one of "
                f"""{", ".join(cls.presets)}. Received: {preset}."""
            )
        metadata = cls.presets[preset]
        config = metadata["config"]
        model = cls.from_config({**config, **kwargs})

        if not load_weights:
            return model

        weights = keras.utils.get_file(
            "model.h5",
            metadata["weights_url"],
            cache_subdir=os.path.join("models", preset),
            file_hash=metadata["weights_hash"],
        )
        model.load_weights(weights)
        return model

    def __init_subclass__(cls, **kwargs):
        # Use __init_subclass__ to setup a correct docstring for from_preset.
        super().__init_subclass__(**kwargs)

        # If the subclass does not define from_preset, assign a wrapper so that
        # each class can have a distinct docstring.
        if "from_preset" not in cls.__dict__:

            def from_preset(calling_cls, *args, **kwargs):
                return super(cls, calling_cls).from_preset(*args, **kwargs)

            cls.from_preset = classmethod(from_preset)

        # Format and assign the docstring unless the subclass has overridden it.
        if cls.from_preset.__doc__ is None:
            cls.from_preset.__func__.__doc__ = Backbone.from_preset.__doc__
            format_docstring(
                model_name=cls.__name__,
                example_preset_name=next(iter(cls.presets), ""),
                preset_names='", "'.join(cls.presets),
            )(cls.from_preset.__func__)

        # If the subclass does not define from_preset, assign a wrapper so that
        # each class can have a distinct docstring.
        if "create_layout_map" not in cls.__dict__:

            def create_layout_map(calling_cls, *args, **kwargs):
                return super(cls, calling_cls).create_layout_map(
                    *args, **kwargs
                )

            cls.create_layout_map = classmethod(create_layout_map)

        # Format and assign the docstring unless the subclass has overridden it.
        if cls.create_layout_map.__doc__ is None:
            cls.create_layout_map.__func__.__doc__ = (
                Backbone.create_layout_map.__doc__
            )
            format_docstring(
                model_name=cls.__name__,
            )(cls.create_layout_map.__func__)

    @classmethod
    def create_layout_map(cls, device_mesh):
        """Create a layout map for model parallel training a {{model_name}}.

        This method takes in a `keras.distribution.DeviceMesh` and returns a
        `keras.distribution.LayoutMap` that will correctly distribute weights
        for a backbone in a model parallel setting.

        Args:
            device_mesh: A 2D `keras.distribution.DeviceMesh` describing the
                arrangement of devices for running distributed computation. The
                first dimension in the mesh is expected to be for data parallel
                distribution, and the second for model parallel distribution.

        Returns:
            A `keras.distribution.LayoutMap` which contains the proper layout to
            weights mapping for the model parallel setting.

        Examples:
        ```python
        device_mesh = keras.distribution.DeviceMesh(
            shape=(2, 4),
            axis_names=('batch', 'model'),
            devices=keras.distribution.list_devices(),
        )
        layout_map = keras_nlp.models.{{model_name}}.create_layout_map(
            device_mesh,
        )
        distribution = keras.distribution.ModelParallel(device_mesh, layout_map)
        keras.distribution.set_distribution(distribution)
        ```
        """
        # We assert the mesh is 2D, and assume the first mesh dim is for data
        # parallel and the second dim is for model parallel.
        mesh_shape = device_mesh.shape
        if len(mesh_shape) != 2:
            raise ValueError(f"Expect a 2D DeviceMesh, received {device_mesh}")
        _, model_dim = device_mesh.axis_names

        layout_map = keras.distribution.LayoutMap(device_mesh=device_mesh)
        # Embedding sharding
        layout_map[r"embeddings"] = [None, model_dim]
        # Transformer block sharding
        layout_map[r"(query|key|value)/kernel"] = [None, None, model_dim]
        layout_map[r"(query|key|value)/bias"] = [model_dim, None]
        layout_map[r"feedforward_intermediate_dense/kernel"] = [
            None,
            model_dim,
        ]
        layout_map[r"feedforward_intermediate_dense/bias"] = [model_dim]
        layout_map[r"feedforward_output_dense/kernel"] = [model_dim, None]
        layout_map[r"feedforward_output_dense/bias"] = [None]
        return layout_map
