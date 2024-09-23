# Copyright 2024 The KerasNLP Authors
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


class FeaturePyramid(keras.layers.Layer):
    def __init__(
        self,
        min_level,
        max_level,
        num_filters=256,
        activation="relu",
        kernel_initializer="VarianceScaling",
        bias_initializer="zeros",
        batch_norm_momentum=0.99,
        batch_norm_epsilon=0.001,
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_level = min_level
        self.max_level = max_level
        self.num_filters = num_filters
        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_epsilon = batch_norm_epsilon
        if kernel_regularizer is not None:
            self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        else:
            self.kernel_regularizer = None
        if bias_regularizer is not None:
            self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        else:
            self.bias_regularizer = None
        self.data_format = keras.backend.image_data_format()
        self.batch_norm_axis = -1 if self.data_format == "channels_last" else 1

    def build(self, input_shape):
        input_levels = [int(level[1]) for level in input_shape]
        backbone_max_level = min(max(input_levels), self.max_level)

        # Build lateral layers
        self.later_layers = {}
        for i in range(self.min_level, backbone_max_level + 1):
            level = f"P{i}"
            self.later_layers[level] = keras.layers.Conv2D(
                filters=self.num_filters,
                kernel_size=1,
                padding="same",
                data_format=self.data_format,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                dtype=self.dtype_policy,
                name=f"layer_{level}",
            )
            self.later_layers[level].build(input_shape[level])

        # Build output layers
        self.output_layers = {}
        for i in range(self.min_level, backbone_max_level + 1):
            level = f"P{i}"
            self.output_layers[level] = keras.layers.Conv2D(
                filters=self.num_filters,
                kernel_size=3,
                padding="same",
                data_format=self.data_format,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                dtype=self.dtype_policy,
                name=f"layer_{level}",
            )
            self.output_layers[level].build((None, None, None, 256))

        # Build coarser layers
        for i in range(backbone_max_level + 1, self.max_level + 1):
            level = f"P{i}"
            self.output_layers[level] = keras.layers.Conv2D(
                filters=self.num_filters,
                strides=2,
                kernel_size=3,
                padding="same",
                data_format=self.data_format,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                dtype=self.dtype_policy,
                name=f"coarser_{level}",
            )
            self.output_layers[level].build((None, None, None, 256))

        # Build batch norm layers
        self.output_batch_norms = {}
        for level in range(self.min_level, self.max_level + 1):
            self.output_batch_norms[f"P{level}"] = (
                keras.layers.BatchNormalization(
                    axis=self.batch_norm_axis,
                    momentum=self.batch_norm_epsilon,
                    epsilon=self.batch_norm_epsilon,
                    name=f"norm_P{level}",
                )
            )
            self.output_batch_norms[f"P{level}"].build((None, None, None, 256))

        # The same upsampling layer is used for all levels
        self.top_down_op = keras.layers.UpSampling2D(
            size=2,
            data_format=self.data_format,
            dtype=self.dtype_policy,
            name="upsampling",
        )
        # The same merge layer is used for all levels
        self.merge_op = keras.layers.Add(
            dtype=self.dtype_policy, name="merge_op"
        )

        self.built = True

    def call(self, inputs):
        output_features = {}

        # Get the backbone max level
        input_levels = [int(level[1]) for level in inputs]
        backbone_max_level = min(max(input_levels), self.max_level)

        for i in range(backbone_max_level, self.min_level - 1, -1):
            level = f"P{i}"
            output = self.later_layers[level](inputs[level])
            if i < backbone_max_level:
                # for the top most output, it doesn't need to merge with any
                # upper stream outputs
                upstream_output = self.top_down_op(output_features[f"P{i+1}"])
                output = self.merge_op([output, upstream_output])
            output_features[level] = output

        # Post apply the output layers so that we don't leak them to the down
        # stream level
        for i in range(backbone_max_level, self.min_level - 1, -1):
            level = f"P{i}"
            output_features[level] = self.output_layers[level](
                output_features[level]
            )

        for i in range(backbone_max_level + 1, self.max_level + 1):
            level = f"P{i}"
            feats_in = output_features[f"P{i-1}"]
            if i > backbone_max_level + 1:
                feats_in = self.activation(feats_in)
            output_features[level] = self.output_layers[level](feats_in)

        return output_features

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "min_level": self.min_level,
                "max_level": self.max_level,
                "num_filters": self.num_filters,
                "activation": keras.activations.serialize(self.activation),
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "batch_norm_momentum": self.batch_norm_momentum,
                "batch_norm_epsilon": self.batch_norm_epsilon,
                "kernel_regularizer": (
                    keras.regularizers.serialize(self.kernel_regularizer)
                    if self.kernel_regularizer is not None
                    else None
                ),
                "bias_regularizer": (
                    keras.regularizers.serialize(self.bias_regularizer)
                    if self.bias_regularizer is not None
                    else None
                ),
            }
        )

        return config

    def compute_output_shape(self, input_shape):
        output_shape = {}

        for i in range(self.min_level, self.max_level):
            level = f"P{i}"
            if self.data_format == "channels_last":
                output_shape[level] = (None, None, None, 256)
            else:
                output_shape[level] = (None, 256, None, None)

        return output_shape
