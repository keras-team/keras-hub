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
import math

from keras import layers
from keras import ops

from keras_hub.src.utils.keras_utils import standardize_data_format


class VAEAttention(layers.Layer):
    def __init__(self, filters, groups=32, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.data_format = standardize_data_format(data_format)
        gn_axis = -1 if self.data_format == "channels_last" else 1

        self.group_norm = layers.GroupNormalization(
            groups=groups,
            axis=gn_axis,
            epsilon=1e-6,
            dtype=self.dtype_policy,
            name="group_norm",
        )
        self.query_conv2d = layers.Conv2D(
            filters,
            1,
            1,
            data_format=self.data_format,
            dtype=self.dtype_policy,
            name="query_conv2d",
        )
        self.key_conv2d = layers.Conv2D(
            filters,
            1,
            1,
            data_format=self.data_format,
            dtype=self.dtype_policy,
            name="key_conv2d",
        )
        self.value_conv2d = layers.Conv2D(
            filters,
            1,
            1,
            data_format=self.data_format,
            dtype=self.dtype_policy,
            name="value_conv2d",
        )
        self.softmax = layers.Softmax(dtype="float32")
        self.output_conv2d = layers.Conv2D(
            filters,
            1,
            1,
            data_format=self.data_format,
            dtype=self.dtype_policy,
            name="output_conv2d",
        )

        self.groups = groups
        self._inverse_sqrt_filters = 1.0 / math.sqrt(float(filters))

    def build(self, input_shape):
        self.group_norm.build(input_shape)
        self.query_conv2d.build(input_shape)
        self.key_conv2d.build(input_shape)
        self.value_conv2d.build(input_shape)
        self.output_conv2d.build(input_shape)

    def call(self, inputs, training=None):
        x = self.group_norm(inputs)
        query = self.query_conv2d(x)
        key = self.key_conv2d(x)
        value = self.value_conv2d(x)

        if self.data_format == "channels_first":
            query = ops.transpose(query, (0, 2, 3, 1))
            key = ops.transpose(key, (0, 2, 3, 1))
            value = ops.transpose(value, (0, 2, 3, 1))
        shape = ops.shape(inputs)
        b = shape[0]
        query = ops.reshape(query, (b, -1, self.filters))
        key = ops.reshape(key, (b, -1, self.filters))
        value = ops.reshape(value, (b, -1, self.filters))

        # Compute attention.
        query = ops.multiply(
            query, ops.cast(self._inverse_sqrt_filters, query.dtype)
        )
        # [B, H0 * W0, C], [B, H1 * W1, C] -> [B, H0 * W0, H1 * W1]
        attention_scores = ops.einsum("abc,adc->abd", query, key)
        attention_scores = ops.cast(
            self.softmax(attention_scores), self.compute_dtype
        )
        # [B, H2 * W2, C], [B, H0 * W0, H1 * W1] -> [B, H1 * W1 ,C]
        attention_output = ops.einsum("abc,adb->adc", value, attention_scores)
        x = ops.reshape(attention_output, shape)

        x = self.output_conv2d(x)
        if self.data_format == "channels_first":
            x = ops.transpose(x, (0, 3, 1, 2))
        x = ops.add(x, inputs)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "groups": self.groups,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
