# Copyright 2022 The KerasNLP Authors
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

"""OPT backbone model."""

import copy

import tensorflow as tf
from tensorflow import keras
from tensorflow.experimental import dtensor
from tensorflow.experimental.dtensor import Layout

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.layers.token_and_position_embedding import (
    TokenAndPositionEmbedding,
)
from keras_nlp.layers.transformer_decoder import TransformerDecoder
from keras_nlp.models.backbone import Backbone
from keras_nlp.models.opt.opt_presets import backbone_presets
from keras_nlp.utils.python_utils import classproperty


def opt_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


@keras_nlp_export("keras_nlp.models.OPTBackbone")
class OPTBackbone(Backbone):
    """An OPT decoder network.

    This class implements a Transformer-based decoder model as described in
    ["OPT: Open Pre-trained Transformer Language Models"](https://arxiv.org/abs/2205.01068).
    The default constructor gives a fully customizable, randomly initialized OPT
    model with any number of layers, heads, and embedding dimensions. To load
    preset architectures and weights, use the `from_preset()` constructor.

    Disclaimer: Pre-trained models are provided on an "as is" basis, without
    warranties or conditions of any kind. The underlying model is provided by a
    third party and subject to a separate license, available
    [here](https://github.com/facebookresearch/fairseq/).

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer decoder layers.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The hidden size of the transformer decoder layers.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer decoder layer.
        dropout: float. Dropout probability for the Transformer decoder.
        max_sequence_length: int. The maximum sequence length that this decoder
            can consume. If `None`, `max_sequence_length` uses the value from
            sequence length. This determines the variable shape for positional
            embeddings.

    Examples:
    ```python
    input_data = {
        "token_ids": tf.ones(shape=(1, 12), dtype=tf.int64),
        "padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
    }

    # Pretrained OPT decoder
    model = keras_nlp.models.OPTBackbone.from_preset("opt_125m_en")
    model(input_data)

    # Randomly initialized OPT decoder model with a custom config
    model = keras_nlp.models.OPTBackbone(
        vocabulary_size=50265,
        num_layers=4,
        num_heads=4,
        hidden_dim=256,
        intermediate_dim=512,
        max_sequence_length=128,
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
        dropout=0.1,
        max_sequence_length=2048,
        **kwargs,
    ):
        # Decoder inputs.
        token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # Embed tokens and positions.
        x = TokenAndPositionEmbedding(
            vocabulary_size=vocabulary_size,
            sequence_length=max_sequence_length,
            embedding_dim=hidden_dim,
            embeddings_initializer=opt_kernel_initializer(),
            name="embeddings",
        )(token_ids)

        # Apply successive transformer decoder blocks.
        for i in range(num_layers):
            x = TransformerDecoder(
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation="relu",
                layer_norm_epsilon=1e-5,
                normalize_first=True,
                kernel_initializer=opt_kernel_initializer(),
                name=f"transformer_layer_{i}",
            )(x, decoder_padding_mask=padding_mask)

        # Add a final layer norm.
        x = keras.layers.LayerNormalization(
            name="layer_norm",
            axis=-1,
            epsilon=1e-5,
            dtype=tf.float32,
        )(x)

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs={
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            },
            outputs=x,
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

    def get_config(self):
        return {
            "vocabulary_size": self.vocabulary_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "intermediate_dim": self.intermediate_dim,
            "dropout": self.dropout,
            "max_sequence_length": self.max_sequence_length,
        }

    @property
    def token_embedding(self):
        return self.get_layer("embeddings").token_embedding

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    @classmethod
    def create_layout_map(cls, mesh):
        """Create a DTensor layout map for an OPTBackbone.

        Given a DTensor mesh describing a list of devices, this method returns a
        DTensor layout map for creating a `keras_nlp.models.OPTBackbone`
        instance. This mapping describes how to distribute all model weights
        across multiple devices. For an overview of DTensor concepts, see
        [this guide](https://www.tensorflow.org/guide/dtensor_overview).

        Args:
            mesh: A 2D `tf.experimental.dtensor.Mesh` describing the arrangement
                of devices for running distributed computation. The
                first dimension in the mesh is expected to be for data parallel
                distribution, and the second for model parallel distribution.

        Returns:
            A `tf.keras.dtensor.experimental.LayoutMap` which contains the
            proper layout to weights mapping for the model parallel setting.

        Examples:
        ```python
        keras.backend.experimental.enable_tf_random_generator()
        keras.utils.set_random_seed(1337)

        # Update both dimensions below for a multi-device setting.
        mesh = dtensor.create_mesh([("batch", 1), ("model", 1)])
        layout_map = keras_nlp.models.OPTBackbone.create_layout_map(mesh)

        with layout_map.scope():
            model = keras_nlp.models.OPTBackbone.from_preset("opt_125m_en")
        ```
        """
        # We assert the mesh is 2D, and assume the first mesh dim is for data
        # parallel and the second dim is for model parallel.
        mesh_shape = mesh.shape()
        if len(mesh_shape) != 2:
            raise ValueError(
                f"Expect to create layout based on 2D mesh, received {mesh}"
            )
        _, model_dim = mesh.dim_names
        unshard_dim = dtensor.UNSHARDED

        layout_map = keras.dtensor.experimental.LayoutMap(mesh=mesh)
        # Embedding sharding
        layout_map[r".*embeddings"] = Layout([unshard_dim, model_dim], mesh)

        # Transformer block sharding
        layout_map[r".*_(query|key|value)_dense.kernel"] = Layout(
            [unshard_dim, unshard_dim, model_dim], mesh
        )
        layout_map[r".*_(query|key|value)_dense.bias"] = Layout(
            [model_dim, unshard_dim], mesh
        )
        layout_map[r".*_feedforward_intermediate_dense.kernel"] = Layout(
            [unshard_dim, model_dim], mesh
        )
        layout_map[r".*_feedforward_intermediate_dense.bias"] = Layout(
            [model_dim], mesh
        )
        layout_map[r".*_feedforward_output_dense.kernel"] = Layout(
            [model_dim, unshard_dim], mesh
        )
        layout_map[r".*_feedforward_output_dense.bias"] = Layout(
            [unshard_dim], mesh
        )
        return layout_map
