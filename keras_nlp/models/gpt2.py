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

"""GPT-2 model configurable class, preconfigured versions, and task heads."""


import os
from collections import defaultdict

import tensorflow as tf
from tensorflow import keras

from keras_nlp.layers import PositionEmbedding
from keras_nlp.layers import TransformerDecoder


def _gpt_2_kernel_initializer(stddev=0.02):
    return keras.initializers.RandomNormal(stddev=stddev)


# TODO(abheesht17): Remove "webtext" from URLs?
checkpoints = {
    "gpt2_base": {
        "model": "Gpt2Base",
        "vocabulary": "webtext",
        "description": (
            "Base size of GPT-2 with 124M parameters. Trained on WebText."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_base_webtext/model.h5",
        "weights_hash": "f4ea6e1b214516dd7de452461ee6e16e",
    },
    "gpt2_medium": {
        "model": "Gpt2Medium",
        "vocabulary": "webtext",
        "description": (
            "Medium size of GPT-2 with 355M parameters. Trained on WebText."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_medium_webtext/model.h5",
        "weights_hash": "580ff9b79c04fc90e6d6f47e975c5afe",
    },
    "gpt2_large": {
        "model": "Gpt2Large",
        "vocabulary": "webtext",
        "description": (
            "Large size of GPT-2 with 774M parameters. Trained on WebText."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_large_webtext/model.h5",
        "weights_hash": "67957cb3dfc9e965960dabe068811e1a",
    },
    "gpt2_extra_large": {
        "model": "Gpt2ExtraLarge",
        "vocabulary": "webtext",
        "description": (
            "Extra Large size of GPT-2 with 1558M parameters. "
            "Trained on WebText."
        ),
        "weights_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_extra_large_webtext/model.h5",
        "weights_hash": "d093c1ee0d9705d845c0190909aa2917",
    },
}

# Index checkpoints by arch compatibility.
checkpoints_per_arch = defaultdict(set)
for arch, metadata in checkpoints.items():
    checkpoints_per_arch[metadata["model"]].add(arch)


def compatible_checkpoints(arch):
    """Returns a list of compatible checkpoints per arch"""
    return checkpoints_per_arch[arch]


# TODO: Iron out this part after BPE tokenizer has been finalized. Also, check
# the to-do comment in `keras_nlp/models/bert.py`.
vocabularies = {
    "webtext": {
        "description": (
            "The BPE vocabulary for GPT-2 models trained on "
            "the WebText dataset."
        ),
        "vocabulary_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_base_webtext/vocab.json",
        "vocabulary_hash": "dffec25a898b1f5e569bec4dffd7e5c0",
        "vocabulary_size": 50257,
        "merges_url": "https://storage.googleapis.com/keras-nlp/models/gpt2_base_webtext/merges.txt",
        "merges_hash": "75a37753dd7a28a2c5df80c28bf06e4e",
    },
}


def _handle_pretrained_model_arguments(gpt2_variant, weights, vocabulary_size):
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
        arch_checkpoints = compatible_checkpoints(gpt2_variant)
        if weights not in arch_checkpoints:
            raise ValueError(
                "`weights` must be one of "
                f"""{", ".join(arch_checkpoints)}. """
                f"Received: {weights}."
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


class Gpt2Custom(keras.Model):
    """GPT-2 core network with customizable hyperparameters.

    This network implements a Transformer-based decoder network,
    Generative Pretrained Transformer-2 (GPT-2), as described in
    ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
    It includes the embedding lookups and transformer layers.

    This class gives a fully customizable GPT-2 model with any number of layers,
    heads, and embedding dimensions. For specific GPT-2 architectures
    defined in the paper, see, for example, `keras_nlp.models.Gpt2Base`.

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

    Example usage:
    ```python
    # Randomly initialized GPT-2 decoder
    model = keras_nlp.models.Gpt2Custom(
        vocabulary_size=50257,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=1024,
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
        max_sequence_length=1024,
        name=None,
        trainable=True,
    ):

        # Inputs
        token_ids = keras.Input(shape=(None,), dtype="int32", name="token_ids")
        padding_mask = keras.Input(
            shape=(None,), dtype="int32", name="padding_mask"
        )

        # Embed tokens, positions.
        token_embedding = keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=hidden_dim,
            embeddings_initializer=_gpt_2_kernel_initializer(stddev=0.01),
            name="token_embedding",
        )(token_ids)

        # Can't use `TokenAndPositionEmbedding` layer here because of different
        # initializers.
        position_embedding = PositionEmbedding(
            initializer=_gpt_2_kernel_initializer(stddev=0.02),
            sequence_length=max_sequence_length,
            name="position_embedding",
        )(token_embedding)

        # Sum and apply dropout to embeddings.
        x = keras.layers.Add()((token_embedding, position_embedding))
        x = keras.layers.Dropout(
            dropout,
            name="embeddings_dropout",
        )(x)

        # Apply successive transformer decoder blocks.
        for i in range(num_layers):
            x = TransformerDecoder(
                intermediate_dim=intermediate_dim,
                num_heads=num_heads,
                dropout=dropout,
                activation=lambda x: keras.activations.gelu(
                    x, approximate=True
                ),
                layer_norm_epsilon=1e-05,
                kernel_initializer=_gpt_2_kernel_initializer(stddev=0.02),
                normalize_first=True,
                name=f"transformer_layer_{i}",
            )(x, decoder_padding_mask=padding_mask)

        sequence_output = keras.layers.LayerNormalization(
            name="layer_norm",
            axis=-1,
            epsilon=1e-05,
            dtype=tf.float32,
        )(x)

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs={
                "token_ids": token_ids,
                "padding_mask": padding_mask,
            },
            outputs=sequence_output,
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


MODEL_DOCSTRING = """GPT-2 "{type}" architecture.

    This network implements a Transformer-based decoder as
    described in
    ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
    It includes the embedding lookups and transformer layers.

    Args:
        vocabulary_size: int, optional. The size of the token vocabulary.
        name: String, optional. Name of the model.
        trainable: boolean, optional. If the model's variables should be
            trainable.

    Example usage:
    ```python
    # Randomly initialized Gpt2{type} encoder
    model = keras_nlp.models.Gpt2{type}(weights=None, vocabulary_size=10000)

    # Call encoder on the inputs.
    input_data = {{
        "token_ids": tf.random.uniform(
            shape=(1, 1024), dtype=tf.int64, maxval=model.vocabulary_size
        ),
        "padding_mask": tf.constant([1] * 1024, shape=(1, 1024)),
    }}
    output = model(input_data)
"""


def Gpt2Base(
    weights="gpt2_base", vocabulary_size=None, name=None, trainable=True
):
    weights, vocabulary_size = _handle_pretrained_model_arguments(
        "Gpt2Base", weights, vocabulary_size
    )

    model = Gpt2Custom(
        vocabulary_size=vocabulary_size,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        dropout=0.1,
        max_sequence_length=1024,
        name=name,
        trainable=trainable,
    )

    if weights:
        model.load_weights(weights)

    return model


def Gpt2Medium(
    weights="gpt2_medium", vocabulary_size=None, name=None, trainable=True
):
    weights, vocabulary_size = _handle_pretrained_model_arguments(
        "Gpt2Medium", weights, vocabulary_size
    )

    model = Gpt2Custom(
        vocabulary_size=vocabulary_size,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        intermediate_dim=4096,
        dropout=0.1,
        max_sequence_length=1024,
        name=name,
        trainable=trainable,
    )

    if weights:
        model.load_weights(weights)

    return model


def Gpt2Large(
    weights="gpt2_large", vocabulary_size=None, name=None, trainable=True
):
    weights, vocabulary_size = _handle_pretrained_model_arguments(
        "Gpt2Large", weights, vocabulary_size
    )

    model = Gpt2Custom(
        vocabulary_size=vocabulary_size,
        num_layers=36,
        num_heads=20,
        hidden_dim=1280,
        intermediate_dim=5120,
        dropout=0.1,
        max_sequence_length=1024,
        name=name,
        trainable=trainable,
    )

    if weights:
        model.load_weights(weights)

    return model


def Gpt2ExtraLarge(
    weights="gpt2_extra_large", vocabulary_size=None, name=None, trainable=True
):
    weights, vocabulary_size = _handle_pretrained_model_arguments(
        "Gpt2ExtraLarge", weights, vocabulary_size
    )

    model = Gpt2Custom(
        vocabulary_size=vocabulary_size,
        num_layers=48,
        num_heads=25,
        hidden_dim=1600,
        intermediate_dim=6400,
        dropout=0.1,
        max_sequence_length=1024,
        name=name,
        trainable=trainable,
    )

    if weights:
        model.load_weights(weights)

    return model


def model_class_by_name(classname):
    """Return model class given the class name."""
    return {
        "Gpt2Base": Gpt2Base,
        "Gpt2Medium": Gpt2Medium,
        "Gpt2Large": Gpt2Large,
        "Gpt2ExtraLarge": Gpt2ExtraLarge,
    }[classname]


setattr(
    Gpt2Base,
    "__doc__",
    MODEL_DOCSTRING.format(type="Base"),
)
setattr(
    Gpt2Medium,
    "__doc__",
    MODEL_DOCSTRING.format(type="Medium"),
)
setattr(
    Gpt2Large,
    "__doc__",
    MODEL_DOCSTRING.format(type="Large"),
)
setattr(
    Gpt2ExtraLarge,
    "__doc__",
    MODEL_DOCSTRING.format(type="ExtraLarge"),
)
