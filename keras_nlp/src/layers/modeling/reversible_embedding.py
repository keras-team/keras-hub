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

import keras
from keras import ops

from keras_nlp.src.api_export import keras_nlp_export


@keras_nlp_export("keras_nlp.layers.ReversibleEmbedding")
class ReversibleEmbedding(keras.layers.Embedding):
    """An embedding layer which can project backwards to the input dim.

    This layer is an extension of `keras.layers.Embedding` for language models.
    This layer can be called "in reverse" with `reverse=True`, in which case the
    layer will linearly project from `output_dim` back to `input_dim`.

    By default, the reverse projection will use the transpose of the
    `embeddings` weights to project to `input_dim` (weights are "tied"). If
    `tie_weights=False`, the model will use a separate, trainable variable for
    reverse projection.

    This layer has no bias terms.

    Args:
        input_dim: Integer. Size of the vocabulary,
            i.e. maximum integer index + 1.
        output_dim: Integer. Dimension of the dense embedding.
        tie_weights: Boolean, whether or not the matrix for embedding and
            the matrix for the `reverse` projection should share the same
            weights.
        embeddings_initializer: Initializer for the `embeddings`
            matrix (see `keras.initializers`).
        embeddings_regularizer: Regularizer function applied to
            the `embeddings` matrix (see `keras.regularizers`).
        embeddings_constraint: Constraint function applied to
            the `embeddings` matrix (see `keras.constraints`).
        mask_zero: Boolean, whether or not the input value 0 is a special
            "padding" value that should be masked out.
        reverse_dtype: The dtype for the reverse projection computation.
            Defaults to the `compute_dtype` of the layer.
        **kwargs: other keyword arguments passed to `keras.layers.Embedding`,
            including `name`, `trainable`, `dtype` etc.

    Call arguments:
        inputs: The tensor inputs to the layer.
        reverse: Boolean. If `True` the layer will perform a linear projection
            from `output_dim` to `input_dim`, instead of a normal embedding
            call. Default to `False`.

    Example:
    ```python
    batch_size = 16
    vocab_size = 100
    hidden_dim = 32
    seq_length = 50

    # Generate random inputs.
    token_ids = np.random.randint(vocab_size, size=(batch_size, seq_length))

    embedding = keras_nlp.layers.ReversibleEmbedding(vocab_size, hidden_dim)
    # Embed tokens to shape `(batch_size, seq_length, hidden_dim)`.
    hidden_states = embedding(token_ids)
    # Project hidden states to shape `(batch_size, seq_length, vocab_size)`.
    logits = embedding(hidden_states, reverse=True)
    ```

    References:
    - [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
    - [Press and Wolf, 2016](https://arxiv.org/abs/1608.05859)
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        tie_weights=True,
        embeddings_initializer="uniform",
        embeddings_regularizer=None,
        embeddings_constraint=None,
        mask_zero=False,
        reverse_dtype=None,
        **kwargs,
    ):
        super().__init__(
            input_dim,
            output_dim,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            embeddings_constraint=embeddings_constraint,
            mask_zero=mask_zero,
            **kwargs,
        )
        self.tie_weights = tie_weights
        self.reverse_dtype = reverse_dtype

    def build(self, inputs_shape=None):
        super().build(inputs_shape)
        if not self.tie_weights and self.quantization_mode != "int8":
            self.reverse_embeddings = self.add_weight(
                name="reverse_embeddings",
                shape=(self.output_dim, self.input_dim),
                initializer=self.embeddings_initializer,
                dtype=self.dtype,
            )

    def call(self, inputs, reverse=False):
        if reverse:
            if self.tie_weights:
                kernel = ops.transpose(ops.convert_to_tensor(self.embeddings))
            else:
                kernel = self.reverse_embeddings
            if self.reverse_dtype is not None:
                inputs = ops.cast(inputs, self.reverse_dtype)
                kernel = ops.cast(kernel, self.reverse_dtype)
            return ops.matmul(inputs, kernel)

        return super().call(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "tie_weights": self.tie_weights,
                "reverse_dtype": self.reverse_dtype,
            }
        )
        return config

    def save_own_variables(self, store):
        if not self.built:
            return
        super().save_own_variables(store)
        target_variables = []
        if not self.tie_weights:
            # Store the reverse embedding weights as the last weights.
            target_variables.append(self.reverse_embeddings)
            if self.quantization_mode == "int8":
                target_variables.append(self.reverse_embeddings_scale)
            for i, variable in enumerate(target_variables, start=len(store)):
                store[str(i)] = variable

    def load_own_variables(self, store):
        if not self.built:
            self.build()
        super().load_own_variables(store)
        if not self.tie_weights:
            # Last weights in the stores are the reverse embedding weights.
            target_variables = [self.reverse_embeddings]
            if self.quantization_mode == "int8":
                target_variables.append(self.reverse_embeddings_scale)
            for i, variable in enumerate(
                target_variables, start=len(store) - len(target_variables)
            ):
                variable.assign(store[str(i)])

    def compute_output_spec(self, inputs, reverse=False):
        output_shape = list(inputs.shape)
        if reverse:
            output_shape[-1] = self.input_dim
        else:
            output_shape += [self.output_dim]
        return keras.KerasTensor(output_shape, dtype=self.dtype)

    # Quantization-related (int8) methods

    def quantized_call(self, inputs, reverse=False):
        # TODO (hongyu): This function could be removed once we add `*args` and
        # `**kwargs` for `Embedding.quantized_call`
        if self.quantization_mode == "int8":
            return self._int8_call(inputs, reverse=reverse)
        else:
            self._quantization_mode_error(self.quantization_mode)

    def _int8_build(
        self,
        embeddings_initializer="zeros",
        embeddings_scale_initializer="ones",
        reverse_embeddings_initializer="zeros",
        reverse_embeddings_scale_initializer="ones",
    ):
        super()._int8_build(
            embeddings_initializer, embeddings_scale_initializer
        )
        self.inputs_quantizer = keras.quantizers.AbsMaxQuantizer(axis=-1)
        if not self.tie_weights:
            self.reverse_embeddings = self.add_weight(
                name="reverse_embeddings",
                shape=(self.output_dim, self.input_dim),
                initializer=reverse_embeddings_initializer,
                dtype="int8",
                trainable=False,
            )
            self.reverse_embeddings_scale = self.add_weight(
                name="reverse_embeddings_scale",
                shape=(self.input_dim,),
                initializer=reverse_embeddings_scale_initializer,
                trainable=False,
            )

    def _int8_call(self, inputs, reverse=False):
        if reverse:
            if self.tie_weights:
                kernel = ops.transpose(self._embeddings)
                scale = ops.transpose(self.embeddings_scale)
            else:
                kernel = self.reverse_embeddings
                scale = self.reverse_embeddings_scale
            inputs, inputs_scale = self.inputs_quantizer(inputs)
            outputs = ops.matmul(inputs, kernel)
            # De-scale outputs
            outputs = ops.cast(outputs, self.compute_dtype)
            outputs = ops.divide(outputs, ops.multiply(inputs_scale, scale))
            return outputs

        return super()._int8_call(inputs)

    def quantize(self, mode):
        import gc

        if type(self) is not ReversibleEmbedding:
            raise NotImplementedError(
                f"Layer {self.__class__.__name__} does not have a `quantize()` "
                "method implemented."
            )
        self._check_quantize_args(mode, self.compute_dtype)

        self._tracker.unlock()
        if mode == "int8":
            embeddings, embeddings_scale = keras.quantizers.abs_max_quantize(
                self._embeddings, axis=-1
            )
            embeddings_scale = ops.squeeze(embeddings_scale, axis=-1)
            self._untrack_variable(self._embeddings)
            del self._embeddings
            if not self.tie_weights:
                reverse_embeddings, reverse_embeddings_scale = (
                    keras.quantizers.abs_max_quantize(
                        self.reverse_embeddings, axis=0
                    )
                )
                reverse_embeddings_scale = ops.squeeze(
                    reverse_embeddings_scale, axis=0
                )
                self._untrack_variable(self.reverse_embeddings)
                del self.reverse_embeddings
            else:
                reverse_embeddings = None
                reverse_embeddings_scale = None
            self._int8_build(
                lambda shape, dtype: embeddings,
                lambda shape, dtype: embeddings_scale,
                lambda shape, dtype: reverse_embeddings,
                lambda shape, dtype: reverse_embeddings_scale,
            )
        else:
            raise self._quantization_mode_error(mode)
        self._tracker.lock()

        if self.dtype_policy.quantization_mode is None:
            policy = keras.dtype_policies.get(
                f"{mode}_from_{self.dtype_policy.name}"
            )
            self.dtype_policy = policy
        gc.collect()
