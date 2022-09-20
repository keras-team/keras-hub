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

"""RoBERTa model configurable class, preconfigured versions, and task heads."""

import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow import keras

from keras_nlp.layers import TokenAndPositionEmbedding
from keras_nlp.layers import TransformerEncoder


def _roberta_kernel_initializer(stddev=0.02):
    return keras.initializers.TruncatedNormal(stddev=stddev)


class RobertaCustom(keras.Model):
    """RoBERTa encoder with a customizable set of hyperparameters.

    This network implements a bi-directional Transformer-based encoder as
    described in ["RoBERTa: A Robustly Optimized BERT Pretraining Approach"](https://arxiv.org/abs/1907.11692).
    It includes the embedding lookups and transformer layers, but does not
    include the masked language modeling head used during pretraining.

    This class gives a fully configurable RoBERTa model with any number of
    layers, heads, and embedding dimensions. For specific RoBERTa architectures
    defined in the paper, see, for example, `keras_nlp.models.RobertaBase`.

    Args:
        vocabulary_size: int. The size of the token vocabulary.
        num_layers: int. The number of transformer layers.
        num_heads: int. The number of attention heads for each transformer.
            The hidden size must be divisible by the number of attention heads.
        hidden_dim: int. The size of the transformer encoding layer.
        intermediate_dim: int. The output dimension of the first Dense layer in
            a two-layer feedforward network for each transformer.
        dropout: float. Dropout probability for the Transformer encoder.
        max_sequence_length: int. The maximum sequence length this encoder can
            consume. The sequence length of the input must be less than
            `max_sequence_length`.
        name: string, optional. Name of the model.
        trainable: boolean, optional. If the model's variables should be
            trainable.

    Example usage:
    ```python
    # Randomly initialized RoBERTa model
    model = keras_nlp.models.RobertaCustom(
        vocabulary_size=50265,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12
    )

    # Call encoder on the inputs.
    input_data = {
        "token_ids": tf.random.uniform(
            shape=(1, 12), dtype=tf.int64, maxval=50265),
        "padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)),
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

        # Index of classification token in the vocabulary
        cls_token_index = 0
        # Inputs
        token_id_input = keras.Input(
            shape=(None,), dtype=tf.int32, name="token_ids"
        )
        padding_mask = keras.Input(
            shape=(None,), dtype=tf.int32, name="padding_mask"
        )

        # Embed tokens and positions.
        embedding_layer = TokenAndPositionEmbedding(
            vocabulary_size=vocabulary_size,
            sequence_length=max_sequence_length,
            embedding_dim=hidden_dim,
            embeddings_initializer=_roberta_kernel_initializer(),
            name="embeddings",
        )
        embedding = embedding_layer(token_id_input)

        # Sum, normalize and apply dropout to embeddings.
        x = keras.layers.LayerNormalization(
            name="embeddings_layer_norm",
            axis=-1,
            epsilon=1e-5,  # Original paper uses this epsilon value
            dtype=tf.float32,
        )(embedding)
        x = keras.layers.Dropout(
            dropout,
            name="embeddings_dropout",
        )(x)

        # Apply successive transformer encoder blocks.
        for i in range(num_layers):
            x = TransformerEncoder(
                num_heads=num_heads,
                intermediate_dim=intermediate_dim,
                activation="gelu",
                dropout=dropout,
                kernel_initializer=_roberta_kernel_initializer(),
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
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_sequence_length = max_sequence_length
        self.intermediate_dim = intermediate_dim
        self.dropout = dropout
        self.cls_token_index = cls_token_index

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "hidden_dim": self.hidden_dim,
                "intermediate_dim": self.intermediate_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "max_sequence_length": self.max_sequence_length,
                "dropout": self.dropout,
                "cls_token_index": self.cls_token_index,
            }
        )
        return config


class RobertaMultiSegmentPacker(keras.layers.Layer):
    """Packs multiple sequences into a single fixed width model input.

    This layer packs multiple input sequences into a single fixed width sequence
    containing start and end delimiters, forming a dense input suitable for a
    classification task for RoBERTa.

    TODO(abheesht17): This is a temporary, unexported layer until we find a way
    to make the exported `MultiSegmentPacker` layer more generic.

    Takes as input a list or tuple of token segments. The layer will process
    inputs as follows:
     - Truncate all input segments to fit within `sequence_length` according to
       the `truncate` strategy.
     - Concatenate all input segments, adding a single `start_value` at the
       start of the entire sequence, `[end_value, end_value]` at the end of
       each segment save the last, and a single `end_value` at the end of the
       entire sequence.
     - Pad the resulting sequence to `sequence_length` using `pad_tokens`.

    Input should be either a `tf.RaggedTensor` or a dense `tf.Tensor`, and
    either rank-1 or rank-2.

    Args:
        sequence_length: The desired output length.
        start_value: The id or token that is to be placed at the start of each
            sequence (called `"<s>"` for RoBERTa). The dtype must match the dtype
            of the input tensors to the layer.
        end_value: The id or token that is to be placed at the end of each
            input segment (called `"</s>"` for RoBERTa). The dtype much match the
            dtype of the input tensors to the layer.
        pad_value: The id or token that is to be placed into the unused
            positions after the last segment in the sequence
            (called "[PAD]" for RoBERTa).
        truncate: The algorithm to truncate a list of batched segments to fit a
            per-example length limit. The value can be either `round_robin` or
            `waterfall`:
                - `"round_robin"`: Available space is assigned one token at a
                    time in a round-robin fashion to the inputs that still need
                    some, until the limit is reached.
                - `"waterfall"`: The allocation of the budget is done using a
                    "waterfall" algorithm that allocates quota in a
                    left-to-right manner and fills up the buckets until we run
                    out of budget. It support arbitrary number of segments.

    Returns:
        A dense, packed token sequence.
    """

    def __init__(
        self,
        sequence_length,
        start_value,
        end_value,
        pad_value=None,
        truncate="round_robin",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        if truncate not in ("round_robin", "waterfall"):
            raise ValueError(
                "Only 'round_robin' and 'waterfall' algorithms are "
                "supported. Received %s" % truncate
            )
        self.truncate = truncate
        self.start_value = start_value
        self.end_value = end_value
        self.pad_value = pad_value

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "start_value": self.start_value,
                "end_value": self.end_value,
                "pad_value": self.pad_value,
                "truncate": self.truncate,
            }
        )
        return config

    def _sanitize_inputs(self, inputs):
        """Force inputs to a list of rank 2 ragged tensors."""
        # Sanitize inputs.
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        if not inputs:
            raise ValueError("At least one input is required for packing")
        input_ranks = [x.shape.rank for x in inputs]
        if not all(0 < rank < 3 for rank in input_ranks):
            raise ValueError(
                "All inputs for packing must have rank 1 or 2. "
                f"Received input ranks: {input_ranks}"
            )
        if None in input_ranks or len(set(input_ranks)) > 1:
            raise ValueError(
                "All inputs for packing must have the same rank. "
                f"Received input ranks: {input_ranks}"
            )
        return inputs

    def _convert_dense(self, x):
        """Converts inputs to rank 2 ragged tensors."""
        if isinstance(x, tf.Tensor):
            return tf.RaggedTensor.from_tensor(x)
        else:
            return x

    def _trim_inputs(self, inputs):
        """Trim inputs to desired length."""
        # Special tokens include the start token at the beginning of the
        # sequence, two `end_value` at the end of every segment save the last,
        # and the `end_value` at the end of the sequence.
        num_special_tokens = 2 * len(inputs)
        if self.truncate == "round_robin":
            return tf_text.RoundRobinTrimmer(
                self.sequence_length - num_special_tokens
            ).trim(inputs)
        elif self.truncate == "waterfall":
            return tf_text.WaterfallTrimmer(
                self.sequence_length - num_special_tokens
            ).trim(inputs)
        else:
            raise ValueError("Unsupported truncate: %s" % self.truncate)

    def _combine_inputs(self, segments):
        """Combine inputs with start and end values added."""
        dtype = segments[0].dtype
        batch_size = segments[0].nrows()

        start_value = tf.convert_to_tensor(self.start_value, dtype=dtype)
        segment_split_values = tf.convert_to_tensor(
            [self.end_value, self.end_value], dtype=dtype
        )
        end_value = tf.convert_to_tensor(self.end_value, dtype=dtype)

        start_column = tf.fill((batch_size, 1), start_value)
        segment_end_columns = tf.repeat(
            segment_split_values[tf.newaxis, :], repeats=batch_size, axis=0
        )
        end_column = tf.fill((batch_size, 1), end_value)

        segments_to_combine = [start_column]
        for seg in segments[:-1]:
            # Combine all segments adding end tokens.
            segments_to_combine.append(seg)
            segments_to_combine.append(segment_end_columns)
        segments_to_combine.append(segments[-1])
        segments_to_combine.append(end_column)

        token_ids = tf.concat(segments_to_combine, 1)
        return token_ids

    def call(self, inputs):
        inputs = self._sanitize_inputs(inputs)

        # If rank 1, add a batch dim.
        rank_1 = inputs[0].shape.rank == 1
        if rank_1:
            inputs = [tf.expand_dims(x, 0) for x in inputs]
        inputs = [self._convert_dense(x) for x in inputs]

        segments = self._trim_inputs(inputs)
        token_ids = self._combine_inputs(segments)
        # Pad to dense tensor output.
        shape = tf.cast([-1, self.sequence_length], tf.int64)
        token_ids = token_ids.to_tensor(
            shape=shape, default_value=self.pad_value
        )
        # Remove the batch dim if added.
        if rank_1:
            token_ids = tf.squeeze(token_ids, 0)

        return token_ids


class RobertaClassifier(keras.Model):
    """RoBERTa encoder model with a classification head.

    Args:
        base_model: A `keras_nlp.models.Roberta` to encode inputs.
        num_classes: int. Number of classes to predict.
        hidden_dim: int. The size of the pooler layer.
        name: string, optional. Name of the model.
        trainable: boolean, optional. If the model's variables should be
            trainable.

    Example usage:
    ```python
    # Randomly initialized RoBERTa encoder
    model = keras_nlp.models.RobertaCustom(
        vocabulary_size=50265,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        intermediate_dim=3072,
        max_sequence_length=12
    )

    # Call classifier on the inputs.
    input_data = {
        "token_ids": tf.random.uniform(
            shape=(1, 12), dtype=tf.int64, maxval=model.vocabulary_size),
        "padding_mask": tf.constant(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], shape=(1, 12)),
    }
    classifier = keras_nlp.models.RobertaClassifier(model, 4)
    logits = classifier(input_data)
    ```
    """

    def __init__(
        self,
        base_model,
        num_classes,
        hidden_dim=None,
        dropout=0.0,
        name=None,
        trainable=True,
    ):
        inputs = base_model.input
        if hidden_dim is None:
            hidden_dim = base_model.hidden_dim

        x = base_model(inputs)[:, base_model.cls_token_index, :]
        x = keras.layers.Dropout(dropout, name="pooled_dropout")(x)
        x = keras.layers.Dense(
            hidden_dim, activation="tanh", name="pooled_dense"
        )(x)
        x = keras.layers.Dropout(dropout, name="classifier_dropout")(x)
        outputs = keras.layers.Dense(
            num_classes,
            kernel_initializer=_roberta_kernel_initializer(),
            name="logits",
        )(x)

        # Instantiate using Functional API Model constructor
        super().__init__(
            inputs=inputs, outputs=outputs, name=name, trainable=trainable
        )
        # All references to `self` below this line
        self.base_model = base_model
        self.num_classes = num_classes


def RobertaBase(vocabulary_size, name=None, trainable=True):
    """RoBERTa implementation using "Base" architecture.

    This network implements a bi-directional Transformer-based encoder as
    described in ["RoBERTa: A Robustly Optimized BERT Pretraining
    Approach"](https://arxiv.org/abs/1907.11692). It includes the
    embedding lookups and transformer layers, but does not include the masked
    language modeling head used during pretraining.

    Args:
        vocabulary_size: int, optional. The size of the token vocabulary.
        name: string, optional. Name of the model.
        trainable: boolean, optional. If the model's variables should be
            trainable.

    Example usage:
    ```python
    # Randomly initialized RobertaBase encoder
    model = keras_nlp.models.RobertaBase(vocabulary_size=10000)

    # Call encoder on the inputs.
    input_data = {
        "token_ids": tf.random.uniform(
            shape=(1, 512), dtype=tf.int64, maxval=model.vocabulary_size),
        "padding_mask": tf.ones((1, 512)),
    }
    output = model(input_data)
    ```
    """

    return RobertaCustom(
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
