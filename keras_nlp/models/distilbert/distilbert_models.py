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

"""DistilBERT backbone models."""

import os

import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers.token_and_position_embedding import (
    TokenAndPositionEmbedding,
)
from keras_nlp.layers.transformer_encoder import TransformerEncoder
from keras_nlp.models.distilbert.distilbert_checkpoints import checkpoints
from keras_nlp.models.distilbert.distilbert_checkpoints import (
    compatible_checkpoints,
)
from keras_nlp.models.distilbert.distilbert_checkpoints import vocabularies


def distilbert_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


def _handle_pretrained_model_arguments(
    distilbert_variant, weights, vocabulary_size
):
    """Look up pretrained defaults for model arguments.

    This helper will validate the `weights` and `vocabulary_size` arguments, and
    fully resolve them in the case we are loading pretrained weights.
    """
    if (vocabulary_size is None and weights is None) or (
        vocabulary_size and weights
    ):
        raise ValueError(
            "One of `vocabulary_size` or `weights` must be specified "
            "(but not both). "
            f"Received: weights={weights}, "
            f"vocabulary_size={vocabulary_size}"
        )

    if weights:
        arch_checkpoints = compatible_checkpoints(distilbert_variant)
        if weights not in arch_checkpoints:
            raise ValueError(
                "`weights` must be one of "
                f"""{", ".join(arch_checkpoints)}. """
                f"Received: {weights}"
            )
        metadata = checkpoints[weights]
        vocabulary = metadata["vocabulary"]
        vocabulary_size = vocabularies[vocabulary]["vocabulary_size"]

        # TODO(jbischof): consider changing format from `h5` to
        # `tf.train.Checkpoint` once
        # https://github.com/keras-team/keras/issues/16946 is resolved.
        weights = keras.utils.get_file(
            "model.h5",
            metadata["weights_url"],
            cache_subdir=os.path.join("models", weights),
            file_hash=metadata["weights_hash"],
        )

    return weights, vocabulary_size


class DistilBertCustom(keras.Model):
    """DistilBERT encoder network with custom hyperparmeters.

    This network implements a bi-directional Transformer-based encoder as
    described in ["DistilBERT, a distilled version of BERT: smaller, faster,
    cheaper and lighter"](https://arxiv.org/abs/1910.01108). It includes the
    embedding lookups and transformer layers, but not the masked language model
    or classification task networks.

    This class gives a fully customizable DistilBERT model with any number of layers,
    heads, and embedding dimensions. For specific DistilBERT architectures defined in
    the paper, see, for example, `keras_nlp.models.DistilBertBase`.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The size of the transformer encoding and pooler layers.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        dropout: float. Dropout probability for the Transformer encoder.
        max_sequence_length: int. The maximum sequence length that this encoder
            can consume. If None, `max_sequence_length` uses the value from
            sequence length. This determines the variable shape for positional
            embeddings.
        name: string, optional. Name of the model.
        trainable: boolean, optional. If the model's variables should be
            trainable.

    Examples:
    ```python
    # Randomly initialized DistilBERT encoder
    model = keras_nlp.models.DistilBertCustom(
        vocabulary_size=30522,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12,
        name="encoder",
    )

    # Call encoder on the inputs
    input_data = {
        "token_ids": tf.random.uniform(
            shape=(1, 12), dtype=tf.int64, maxval=model.vocabulary_size
        ),
        "padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)
        ),
    }
    output = model(input_data)
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
        max_sequence_length=512,
        name=None,
        trainable=True,
    ):

        # Inputs
        token_id_input = keras.Input(
            shape=(None,), dtype="int32", name="token_ids"
        )
        padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # Embed tokens and positions.
        x = TokenAndPositionEmbedding(
            vocabulary_size=vocabulary_size,
            sequence_length=max_sequence_length,
            embedding_dim=hidden_dim,
            embeddings_initializer=distilbert_kernel_initializer(),
            name="token_and_position_embedding",
        )(token_id_input)

        # Normalize and apply dropout to embeddings.
        x = keras.layers.LayerNormalization(
            name="embeddings_layer_norm",
            axis=-1,
            epsilon=1e-12,
            dtype=tf.float32,
        )(x)
        x = keras.layers.Dropout(
            dropout,
            name="embeddings_dropout",
        )(x)

        # Apply successive transformer encoder blocks.
        for i in range(num_layers):
            x = TransformerEncoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                activation=lambda x: keras.activations.gelu(
                    x, approximate=False
                ),
                dropout=dropout,
                kernel_initializer=distilbert_kernel_initializer(),
                name=f"transformer_layer_{i}",
            )(x, padding_mask=padding_mask)

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs={
                "token_ids": token_id_input,
                "padding_mask": padding_mask,
            },
            outputs=x,
            name=name,
            trainable=trainable,
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
            }
        )
        return config


MODEL_DOCSTRING = """DistilBert "{type}" architecture.

This network implements a bi-directional Transformer-based encoder as
described in ["DistilBERT, a distilled version of BERT: smaller, faster,
cheaper and lighter"](https://arxiv.org/abs/1910.01108). It includes the
embedding lookups and transformer layers, but not the masked language model
or classification task networks.

Args:
    weights: string, optional. Name of pretrained model to load weights.
        Should be one of {names}.
        If None, model is randomly initialized. Either `weights` or
        `vocabulary_size` must be specified, but not both.
    vocabulary_size: Int, optional. The size of the token vocabulary. Either
        `weights` or `vocabulary_size` must be specified, but not both.
    name: string, optional. Name of the model.
    trainable: boolean, optional. If the model's variables should be
        trainable.

Examples:
```python
# Randomly initialized DistilBert{type} encoder
model = keras_nlp.models.DistilBert{type}(vocabulary_size=10000)

# Call encoder on the inputs.
input_data = {{
    "token_ids": tf.random.uniform(
        shape=(1, 512), dtype=tf.int64, maxval=model.vocabulary_size
    ),
    "padding_mask": tf.constant([1] * 512, shape=(1, 512)),
}}
output = model(input_data)

# Load a pretrained model
model = keras_nlp.models.DistilBert{type}(
    weights="distilbert_base_uncased_en"
)
# Call encoder on the inputs.
output = model(input_data)
```
"""


def DistilBertBase(
    weights=None, vocabulary_size=None, name=None, trainable=True
):
    weights, vocabulary_size = _handle_pretrained_model_arguments(
        "DistilBertBase", weights, vocabulary_size
    )

    model = DistilBertCustom(
        vocabulary_size=vocabulary_size,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        dropout=0.1,
        max_sequence_length=512,
        name=name,
        trainable=trainable,
    )

    if weights is not None:
        model.load_weights(weights)

    return model


def model_class_by_name(classname):
    """Return model class given the class name."""
    return {
        "DistilBertBase": DistilBertBase,
    }[classname]


setattr(
    DistilBertBase,
    "__doc__",
    MODEL_DOCSTRING.format(
        type="Base", names=", ".join(compatible_checkpoints("DistilBertBase"))
    ),
)
