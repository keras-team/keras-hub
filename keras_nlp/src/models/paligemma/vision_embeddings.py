# Copyright 2024 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writingf, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from keras_nlp.src.backend import keras
from keras_nlp.src.backend import ops


class VisionEmbeddings(keras.layers.Layer):
    def __init__(
        self,
        hidden_dim,
        image_size=224,
        patch_size=14,
        num_channels=3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.patch_embedding = keras.layers.Conv2D(
            filters=self.hidden_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            activation=None,
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = keras.layers.Embedding(
            self.num_positions,
            self.hidden_dim,
            name="position_embedding",
        )

        self.position_ids = ops.expand_dims(
            ops.arange(self.num_positions), axis=0
        )

    def build(self, input_shape):
        self.patch_embedding.build(input_shape)
        self.position_embedding.build([1, self.num_positions])
        self.built = True

    def call(self, input_tokens):
        x = self.patch_embedding(input_tokens)
        input_shape = ops.shape(x)
        x = ops.reshape(x, [input_shape[0], -1, input_shape[-1]])
        x = x + self.position_embedding(self.position_ids)
        return x

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            self.num_patches,
            self.hidden_dim,
        )
