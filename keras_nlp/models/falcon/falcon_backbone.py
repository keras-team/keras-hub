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

"""Falcon backbone model."""

import copy
import tensorflow as tf
from tensorflow import keras
from keras_nlp.api_export import keras_nlp_export
from keras_nlp.models.backbone import Backbone
# from keras_nlp.models.falcon.falcon_presets import backbone_presets
from keras_nlp.models.falcon.falcon_decoder import FalconDecoderLayer
def _falcon_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)

from keras_nlp.utils.python_utils import classproperty


@keras_nlp_export("keras_nlp.models.FalconBackbone")
class FalconBackbone(Backbone):
    """A Falcon decoder network.

    This class implements a Transformer-based decoder model as described in
    ["Falcon: Open Pre-trained Transformer Language Models"]().
    The default constructor gives a fully customizable, randomly initialized Falcon
    model with any number of layers, heads, and embedding dimensions. To load
    preset architectures and weights, use the `from_preset()` constructor.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://huggingface.co/tiiuae/).
    
    Args:
   

    Examples:
    ```python
    input_data = {
        "token_ids": tf.ones(shape=(1, 12), dtype="int64"),
        "padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
    }

    # Pretrained Falcon model
    model = keras_nlp.models.FalconBackbone.from_preset("falcon_base")
    model(input_data)

    # Randomly initialized Falcon encoder with custom config.
    model = keras_nlp.models.FalconBackbone(
   
    )
    model(input_data)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        num_layers,
        num_heads,
        hidden_dim,
        intermediate_dim,
        dropout=0.0,
        layer_norm_epsilon=1e-5,
        max_sequence_length=512,
        **kwargs,
        ):
        

            # Inputs
            token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
            padding_mask = keras.Input( shape=(None,), dtype="int32", name="padding_mask")

            # Embed tokens.
            token_embedding = keras.layers.Embedding(
                input_dim=vocabulary_size , #vocabulary_size,
                output_dim=hidden_dim,
                embeddings_initializer=_falcon_kernel_initializer(stddev=0.01),
                name="token_embedding",
            )(token_ids)

            x = keras.layers.Dropout(
                dropout,
                name="embeddings_dropout",
            )(token_embedding)

            # Apply successive transformer decoder blocks.
            for i in range(num_layers):
                x = FalconDecoderLayer(
                        num_heads,
                        hidden_dim,
                        dropout=0.0,
                        layer_norm_epsilon=1e-5,
                        max_sequence_length=512,
                name=f"transformer_layer_{i}",
                )(x, decoder_padding_mask=padding_mask)

            sequence_output = keras.layers.LayerNormalization(
                name="layer_norm",
                axis=-1,
                epsilon=layer_norm_epsilon,
                dtype=tf.float32,
            )(x)

            # Instantiate using Functional API Model constructor
            super().__init__(
                inputs={
                    "token_ids": token_ids,
                    "padding_mask": padding_mask,
                },
                outputs=sequence_output,
                **kwargs,
            )
            # All references to `self` below this line
            self.vocabulary_size = vocabulary_size
            self.num_layers = num_layers
            self.num_heads = num_heads
            self.hidden_dim = hidden_dim
            self.intermediate_dim = intermediate_dim
            self.dropout = dropout
            self.max_sequence_length = max_sequence_length
            self.layer_norm_epsilon = layer_norm_epsilon

    def get_config(self):
            config = super().get_config()
            config.update(
                {
                    "vocabulary_size": self.vocabulary_size,
                    "num_layers": self.num_layers,
                    "num_heads": self.num_heads,
                    "hidden_dim": self.hidden_dim,
                    "intermediate_dim": self.intermediate_dim,
                    "dropout": self.dropout,
                    "max_sequence_length": self.max_sequence_length,
                    "layer_norm_epsilon": self.layer_norm_epsilon,
                }
            )
            return config

    @property
    def token_embedding(self):
            return self.get_layer("token_embedding")

