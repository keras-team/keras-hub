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

"""Falcon MLP"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import gelu
from keras_nlp.api_export import keras_nlp_export


@keras_nlp_export("keras_nlp.layers.FalconMLP")
class FalconMLP(keras.layers.Layer):
    """
    FalconMLP is a multi-layer perceptron used in the Falcon models.

    Args:
        config (object): Configuration object containing the MLP parameters.

    Attributes:
        dense_h_to_4h (Dense): Dense layer from hidden size to 4 times hidden size.
        act (function): Activation function (GELU).
        dense_4h_to_h (Dense): Dense layer from 4 times hidden size to hidden size.
        hidden_dropout (float): Dropout rate for the hidden layer.

    Methods:
        call(x, training=None): Performs the forward pass of the MLP.

    """

    def __init__(self, config):
        super(FalconMLP, self).__init__()
        hidden_size = config.hidden_size

        self.dense_h_to_4h = Dense(4 * hidden_size, use_bias=config.bias)
        self.act = gelu
        self.dense_4h_to_h = Dense(hidden_size, use_bias=config.bias)
        self.hidden_dropout = config.hidden_dropout

    def call(self, x, training=None):
        """
        Performs the forward pass of the MLP.

        Args:
            x (tf.Tensor): Input tensor.
            training (bool): Whether the layer is called in training mode or not.

        Returns:
            tf.Tensor: Output tensor.

        """
        x = self.act(self.dense_h_to_4h(x))
        x = self.dense_4h_to_h(x)
        return x
