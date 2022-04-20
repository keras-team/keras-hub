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

"""Masked Language Model classification head."""

import tensorflow as tf
from tensorflow import keras


class MLMClassificationHead(keras.layers.Layer):
    """Masked Language Model classification head.

    This layer takes two inputs:
     - `inputs`: which should be a tensor of encoded tokens with shape
            `(batch_size, sequence_length, encoding_dim)`.
     - `mask_positions`: which should be a tensor of integer positions to
            predict with shape `(batch_size, masks_per_sequence)`.

    The token encodings should usually be the last output of an encoder model,
    and mask positions should be the interger positions you would like to
    predict for the MLM task.

    The layer will first gather the token encodings at the mask positions. These
    gathered tokens will be passed through a dense layer the same size as
    encoding dimension, then transformed to predictions the same size as the
    input vocabulary. This layer will produce a single output with shape
    `(batch_size, masks_per_sequence, vocabulary_size)`, which can be used to
    compute an MLM loss function.

    Args:
        vocabulary_size: The total size of the vocabulary for predictions.
        embedding_weights: Optional. The weights of the word embedding used
            to transform input token ids. The transpose of this weight matrix
            will be used to project a token embedding vector to a prediction
            over all input words, as described in [1].
        inner_activation: The activation function of inner dense layer.
        outer_activation: The activation function for the outputs of the layer.
            Usually either `None` (return logits), or `"softmax"`
            (return probabilities).
        layer_norm_epsilon: float, defaults to 1e-5. The epsilon value in layer
            normalization components.
        kernel_initializer: string or tf.keras.initializers initializer,
            defaults to "glorot_uniform". The kernel initializer for
            the dense and multiheaded attention layers.
        bias_initializer: string or tf.keras.initializers initializer,
            defaults to "zeros". The bias initializer for
            the dense and multiheaded attention layers.
        name: string, defaults to None. The name of the layer.
        **kwargs: other keyword arguments.

    Examples:

    ```python
    inputs = {
        "tokens": keras.Input(shape=(100,), dtype="int32"),
        "mask_positions": keras.Input(shape=(20,), dtype="int32"),
    }

    # Encode the tokens, e.g. with a transformer.
    encoded_tokens = encoder_model(inputs["tokens"])

    # Predict an output word for each masked input token.
    outputs = keras_nlp.layers.MLMClassificationHead(
        vocabulary_size=10000,
        outer_activation="softmax",
    )(encoded_tokens, mask_positions=inputs["mask_indices"])

    # Build a model
    model = keras.Model(inputs, outpus)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    ```

    References:
        [1] [Press and Wolf, 2016](https://arxiv.org/abs/1608.05859)
    """

    def __init__(
        self,
        vocabulary_size=None,
        embedding_weights=None,
        inner_activation="relu",
        outer_activation=None,
        layer_norm_epsilon=1e-05,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.vocabulary_size = vocabulary_size
        self.embedding_weights = embedding_weights
        self.inner_activation = keras.activations.get(inner_activation)
        self.outer_activation = keras.activations.get(outer_activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self._built = False

        if vocabulary_size is None and embedding_weights is None:
            raise ValueError(
                "One of `vocabulary_size` or `embedding_weights` must be set. "
                "Received: `vocabulary_size=None`, `embedding_weights=None`"
            )

        if embedding_weights is not None:
            shape = embedding_weights.shape
            if vocabulary_size is not None and vocabulary_size != shape[0]:
                raise ValueError(
                    "`vocabulary_size` should match the first dimension of the "
                    "shape of `embedding_weights`. Received: "
                    f"`vocabulary_size={vocabulary_size}`, "
                    f"`embedding_weights.shape={shape}`"
                )
            self.vocabulary_size = shape[0]

    def _build(self, input_shape):
        # Create layers based on input shape.
        self._built = True
        feature_size = input_shape[-1]

        self._dense = keras.layers.Dense(
            feature_size,
            activation=self.inner_activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        self._layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )
        if self.embedding_weights is None:
            self._kernel = self.add_weight(
                name="output_kernel",
                shape=[feature_size, self.vocabulary_size],
                initializer=self.kernel_initializer,
                dtype=self.dtype,
            )
        self._bias = self.add_weight(
            name="output_bias",
            shape=[self.vocabulary_size],
            initializer=self.bias_initializer,
            dtype=self.dtype,
        )

    def call(self, inputs, mask_positions):
        if not self._built:
            self._build(inputs.shape)

        # Gather the encoded tokens at the masked indices.
        x = tf.gather(inputs, mask_positions, axis=1, batch_dims=1)

        # Apply a trainable linear transformation and a layer norm.
        x = self._dense(x)
        x = self._layer_norm(x)

        # Transform encodings to vocabulary_size predictions.
        if self.embedding_weights is None:
            outputs = tf.matmul(x, self._kernel)
        else:
            outputs = tf.matmul(
                x,
                tf.cast(self.embedding_weights, self.compute_dtype),
                transpose_b=True,
            )
        outputs = outputs + self._bias

        # Apply a final activation.
        if self.outer_activation is not None:
            outputs = self.outer_activation(outputs)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "inner_activation": keras.activations.serialize(
                    self.inner_activation
                ),
                "outer_activation": keras.activations.serialize(
                    self.outer_activation
                ),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
            }
        )
        return config
