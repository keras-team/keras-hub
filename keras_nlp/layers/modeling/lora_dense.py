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
import re

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.backend import config
from keras_nlp.backend import keras
from keras_nlp.backend import ops


def validate_einsum_equation(equation):
    # For simplicity, we greatly restrict possible einsum equations. The final
    # axis of the input must be the first axis of our kernel, and must not
    # appear in our output.
    left, right, output = re.split(",|->", equation)
    valid = (
        left[-1] == right[0]
        and left[-1] not in output
        and set(left[:-1]).isdisjoint(set(right[1:]))
    )
    if not valid:
        raise ValueError(
            "When passing a `EinsumDense` layer to a `LoraDense` layer, the "
            "einsum `equation` must always have the form `*x,x*->*`, where "
            "each `*` can be any sequence. Conceptually, the `equation` should "
            "always represent a dense matmul on the last axis of the input. "
            f"Received invalid equation `'{equation}'`."
        )


@keras_nlp_export("keras_nlp.layers.LoraDense")
class LoraDense(keras.layers.Layer):
    """A LoRA adapter layer for a dense input layer.

    This layer implements a low-rank decomposition of a dense transformation, as
    described in [LoRA: Low-Rank Adaptation Of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf)
    This layer can be used to replace a dense layer with a layer whose
    parameters are mostly frozen.

    By default, this layer takes in an `inner_dense` layer, freezes its
    parameters, and builds a low-rank decomposed update to sum with the original
    `inner_dense` output. These update parameters can be merged back into the
    `inner_dense` kernel by calling `merge_weights()`.

    Args:
        inner_dense: A `keras.layers.Dense` or `keras.layers.EinsumDense`.
            The inner dense layer to freeze and wrap with the `LoraDense`
            layer. Note that for `EinsumDense` layers, the einsum equation must
            represent a dense transformation on the last axis of the input,
            though adding new axes to the output (e.g. a multi-head axis) is
            allowed.
        rank: int. The inner rank of the decomposed dense transformation. The
            lower this number, the fewer trainable parameters the layer will
            have.
        alpha: float. A constant value used for scaling the lora update. The
            lora update to the original dense transformation will be scaled by
            `alpha / rank`.
        lora_a_initializer: The initializer to use for the inner projection
            from layer inputs to the inner `rank` intermediate outputs.
        freeze_kernel: If true, the kernel of the inner dense layer will have
            `trainable` set to `False`.
        freeze_bias: If true, the kernel of the inner dense layer will have
            `trainable` set to `False`.
        **kwargs: other keyword arguments.

    Examples:

    Wrap a `Dense` layer.
    ```python
    batch_size, feature_size = 4, 16
    rank = 4
    inputs = np.random.uniform(size=(batch_size, feature_size))
    inner_dense = keras.layers.Dense(feature_size)
    lora_dense = keras_nlp.layers.LoraDense(inner_dense, rank=4)
    # Output with inner dense begins equal.
    assert np.allclose(inner_dense(inputs), lora_dense(inputs))

    # Add some random updates to the lora parameters.
    lora_dense.lora_a.assign(np.random.uniform(size=(feature_size, rank)))
    lora_dense.lora_b.assign(np.random.uniform(size=(rank, feature_size)))
    assert not np.allclose(inner_dense(inputs), lora_dense(inputs))

    # Merge the lora dense and output
    lora_dense.merge_weights()
    assert np.allclose(inner_dense(inputs), lora_dense(inputs))
    ```

    Wrap an `EinsumDense` layer with a multi-head projection.
    ```python
    batch_size, sequence_length, feature_size = 4, 10, 16
    num_heads = 2
    rank = 4
    inputs = np.random.uniform(size=(batch_size, sequence_length, feature_size))
    inner_dense = keras.layers.EinsumDense(
        "abc,cde->abde",
        output_shape=(sequence_length, num_heads, feature_size // num_heads),
    )
    lora_dense = keras_nlp.layers.LoraDense(inner_dense, rank=4)
    # Output shape (4, 10, 2, 8)
    lora_dense(inputs)
    ```
    """

    def __init__(
        self,
        inner_dense,
        rank=8,
        alpha=8.0,
        lora_a_initializer="variance_scaling",
        freeze_kernel=True,
        freeze_bias=True,
        **kwargs,
    ):
        # Default to the same dtype as our inner layer.
        if "dtype" not in kwargs:
            kwargs["dtype"] = inner_dense.dtype_policy
        super().__init__(**kwargs)

        if not config.multi_backend():
            raise ValueError(
                "Lora only works with multi-backend Keras 3. Please set the "
                "`KERAS_BACKEND` environment variable to use this API."
            )

        if isinstance(inner_dense, keras.layers.Dense):
            self.inner_dense = inner_dense
        elif isinstance(inner_dense, keras.layers.EinsumDense):
            self.inner_dense = inner_dense
            validate_einsum_equation(inner_dense.equation)
        else:
            raise ValueError(
                "Only `Dense` and `EinsumDense` inner layers are supported. "
                f"Received: inner_dense={inner_dense}"
            )

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.freeze_kernel = freeze_kernel
        self.freeze_bias = freeze_bias
        self.lora_a_initializer = keras.initializers.get(lora_a_initializer)

        if inner_dense.built:
            self.build_from_config(inner_dense.get_build_config())

    def build(self, inputs_shape):
        if not self.inner_dense.built:
            self.inner_dense.build(inputs_shape)

        if self.freeze_kernel and self.inner_dense.kernel is not None:
            self.inner_dense.kernel.trainable = False

        if self.freeze_bias and self.inner_dense.bias is not None:
            self.inner_dense.bias.trainable = False

        input_dim = inputs_shape[-1]
        self.lora_a = self.add_weight(
            name="lora_a",
            shape=(input_dim, self.rank),
            initializer=self.lora_a_initializer,
        )
        kernel_shape = self.inner_dense.kernel.shape
        self.lora_b = self.add_weight(
            name="lora_b",
            shape=(self.rank,) + kernel_shape[1:],
            initializer="zeros",
        )
        self.built = True

    def merge_weights(self):
        """Merge lora updates into the wrapped dense layer.

        This function should only be called outside of any compiled context
        (e.g. not during `fit()`, `predict()` or `evaluate()`). It will merge
        the updates from the lora layers into the original dense layer, and
        re-initialize the lora variables.
        """
        if not self.built:
            return

        # Compute matmul of lora_a and lora_b to get a kernel sized update.
        update = ops.tensordot(self.lora_a, self.lora_b, axes=([-1], [0]))
        update = update * ops.cast(self.scale, update.dtype)
        # Add lora updates back into the inner dense kernel.
        self.inner_dense.kernel.assign_add(update)
        # Re-initialize lora weights.
        self.lora_a.assign(
            self.lora_a_initializer(self.lora_a.shape, self.lora_a.dtype)
        )
        self.lora_b.assign(ops.zeros_like(self.lora_b))

    def call(self, inputs):
        original_output = self.inner_dense(inputs)
        # Compute the low-rank intermediate output.
        update = ops.matmul(inputs, self.lora_a)
        # Use the matching dense computation for a Dense or EinsumDense.
        if isinstance(self.inner_dense, keras.layers.Dense):
            update = ops.matmul(update, self.lora_b)
        else:
            update = ops.einsum(self.inner_dense.equation, update, self.lora_b)
        # Scale and sum the lora update with the original frozen output.
        return original_output + update * ops.cast(self.scale, update.dtype)

    @classmethod
    def from_config(cls, config):
        config["inner_dense"] = keras.layers.deserialize(config["inner_dense"])
        return super().from_config(config)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "inner_dense": keras.layers.serialize(self.inner_dense),
                "rank": self.rank,
                "alpha": self.alpha,
                "lora_a_initializer": keras.initializers.serialize(
                    self.lora_a_initializer
                ),
                "freeze_kernel": self.freeze_kernel,
                "freeze_bias": self.freeze_bias,
            }
        )
        return config
