"""DoRA (Weight-Decomposed Low-Rank Adaptation) Dense Layer Implementation.

This module implements the DoRA dense layer that decomposes weights
into magnitude and direction components, applying low-rank
adaptation for efficient fine-tuning.

Backend-compatible with TensorFlow, PyTorch, and JAX.

Reference: DoRA: Weight-Decomposed Low-Rank Adaptation
"""

import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export


@keras_hub_export("keras_hub.layers.DoRADense")
class DoRADense(layers.Layer):
    """DoRA (Weight-Decomposed Low-Rank Adaptation) Dense layer.

    DoRA decomposes the weight matrix W into magnitude and direction components
    W = m * (W_0 + B @ A) / ||W_0 + B @ A||_c

    Where:
    - m: magnitude vector (learnable)
    - W_0: frozen pretrained weights
    - A, B: low-rank adaptation matrices (learnable)
    - ||.||_c: column-wise L2 norm

    Args:
        units: Positive integer, dimensionality of the output space.
        rank: Rank of the adaptation. Positive integer.
        alpha: LoRA scaling parameter. Float.
        use_bias: Boolean, whether the layer uses a bias vector.
        dropout: Float between 0 and 1. Fraction of input units to drop.
        activation: Activation function to use.
        kernel_initializer: Initializer for the kernel weights matrix.
        bias_initializer: Initializer for the bias vector.

        lora_a_initializer: Initializer for the A matrix.
        Defaults to 'he_uniform'.
        lora_b_initializer: Initializer for the B matrix.
        Defaults to 'zeros'.
        magnitude_initializer: Initializer for magnitude vector.
        Defaults to 'ones'.

        kernel_regularizer: Regularizer function applied to kernel weights.
        bias_regularizer: Regularizer function applied to bias.
        activity_regularizer: Regularizer function applied to output.
        kernel_constraint: Constraint function applied to kernel weights.
        bias_constraint: Constraint function applied to bias.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        units,
        rank=4,
        alpha=1.0,
        use_bias=True,
        dropout=0.0,
        activation=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        lora_a_initializer="he_uniform",
        lora_b_initializer="zeros",
        magnitude_initializer="ones",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Validate parameters
        if units <= 0:
            raise ValueError(f"units must be positive, got {units}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.units = units
        self.rank = rank
        self.alpha = alpha
        self.use_bias = use_bias
        self.dropout_rate = dropout
        self.activation = keras.activations.get(activation)

        # Initializers
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.lora_a_initializer = keras.initializers.get(lora_a_initializer)
        self.lora_b_initializer = keras.initializers.get(lora_b_initializer)
        self.magnitude_initializer = keras.initializers.get(
            magnitude_initializer
        )

        # Regularizers
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)

        # Constraints
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        # Dropout layer
        self.dropout_layer = (
            layers.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
        )

        # Scaling factor
        self.scaling = self.alpha / self.rank
        self.input_spec = None

        # Weight matrices (will be initialized in build())
        self.kernel = None  # Frozen pretrained weights W_0
        self.lora_a = None  # Low-rank matrix A (input_dim, rank)
        self.lora_b = None  # Low-rank matrix B (rank, units)
        self.magnitude = None  # Magnitude vector m (units,)
        self.bias = None

    def build(self, input_shape):
        """Build the layer weights."""
        if len(input_shape) < 2:
            raise ValueError(
                f"Input shape must have at least 2 dimensions,"
                f" got {input_shape}"
            )

        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError(
                "The last dimension of input shape must be defined"
            )

        # Build frozen kernel weights (pretrained weights W_0)
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=False,  # Frozen pretrained weights
        )

        # Build LoRA matrices
        self.lora_a = self.add_weight(
            name="lora_a",
            shape=(input_dim, self.rank),
            initializer=self.lora_a_initializer,
            trainable=True,
        )

        self.lora_b = self.add_weight(
            name="lora_b",
            shape=(self.rank, self.units),
            initializer=self.lora_b_initializer,
            trainable=True,
        )

        # Build magnitude vector
        self.magnitude = self.add_weight(
            name="magnitude",
            shape=(self.units,),
            initializer=self.magnitude_initializer,
            trainable=True,
        )

        # Build bias
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
            )

        super().build(input_shape)

    def call(self, inputs, training=None):
        if self.dropout_layer is not None:
            inputs = self.dropout_layer(inputs, training=training)

        # Compute LoRA adaptation: A @ B
        lora_adaptation = ops.matmul(self.lora_a, self.lora_b) * self.scaling

        # Combine with frozen weights: W_0 + B @ A
        combined_weight = self.kernel + lora_adaptation

        # Compute column-wise L2 norms
        column_norms = ops.sqrt(
            ops.sum(ops.square(combined_weight), axis=0, keepdims=True)
        )
        column_norms = ops.maximum(column_norms, 1e-8)

        # Normalize by column norms
        normalized_weight = combined_weight / column_norms

        # Apply magnitude scaling
        dora_weight = normalized_weight * ops.expand_dims(
            self.magnitude, axis=0
        )

        # Apply linear transformation
        outputs = ops.matmul(inputs, dora_weight)

        if self.use_bias:
            outputs = outputs + self.bias
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def get_dora_parameters(self):
        """Get DoRA-specific parameters.

        Returns:
            Dictionary containing DoRA parameters.
        """
        params = {
            "lora_a": self.lora_a,
            "lora_b": self.lora_b,
            "magnitude": self.magnitude,
        }
        if self.use_bias:
            params["bias"] = self.bias
        return params

    def get_effective_weight(self):
        """Compute the effective weight matrix after DoRA adaptation.

        Returns:
            The effective weight matrix: m * (W_0 + B @ A) / ||W_0 + B @ A||_c
        """
        # Compute adaptation
        lora_adaptation = ops.matmul(self.lora_a, self.lora_b) * self.scaling
        combined_weight = self.kernel + lora_adaptation

        # Normalize
        column_norms = ops.sqrt(
            ops.sum(ops.square(combined_weight), axis=0, keepdims=True)
        )
        column_norms = ops.maximum(column_norms, 1e-8)
        normalized_weight = combined_weight / column_norms

        # Apply magnitude
        return normalized_weight * ops.expand_dims(self.magnitude, axis=0)

    def merge_weights(self):
        """Merge DoRA weights back to a single weight matrix.

        This is useful for inference optimization
        or converting back to standard Dense layer.

        Returns:
            Dictionary with 'kernel' and optionally 'bias'.
        """
        merged_weights = {"kernel": self.get_effective_weight()}
        if self.use_bias:
            merged_weights["bias"] = self.bias
        return merged_weights

    def count_params(self):
        """Count the number of trainable parameters in DoRA layer.

        Returns:
            Number of trainable parameters.
        """
        if not self.built:
            return 0

        input_dim = self.kernel.shape[0]
        param_count = (
            input_dim * self.rank  # lora_a
            + self.rank * self.units  # lora_b
            + self.units  # magnitude
        )
        if self.use_bias:
            param_count += self.units
        return param_count

    def load_pretrained_weights(self, pretrained_kernel, pretrained_bias=None):
        """Load pretrained weights into the frozen kernel.

        Args:
            pretrained_kernel: Pretrained weight matrix.
            pretrained_bias: Optional pretrained bias vector.
        """
        if pretrained_kernel.shape != self.kernel.shape:
            raise ValueError(
                f"Pretrained kernel shape {pretrained_kernel.shape} "
                f"doesn't match expected shape {self.kernel.shape}"
            )

        self.kernel.assign(pretrained_kernel)

        # Initialize magnitude vector to column-wise
        # norms of pretrained weights
        # This ensures DoRA starts with behavior identical to original weights
        column_norms = ops.sqrt(ops.sum(ops.square(pretrained_kernel), axis=0))
        column_norms = ops.maximum(column_norms, 1e-8)
        self.magnitude.assign(column_norms)

        if pretrained_bias is not None and self.use_bias:
            if pretrained_bias.shape != self.bias.shape:
                raise ValueError(
                    f"Pretrained bias shape {pretrained_bias.shape} "
                    f"doesn't match expected shape {self.bias.shape}"
                )
            self.bias.assign(pretrained_bias)

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "rank": self.rank,
                "alpha": self.alpha,
                "use_bias": self.use_bias,
                "dropout": self.dropout_rate,
                "activation": keras.activations.serialize(self.activation),
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "lora_a_initializer": keras.initializers.serialize(
                    self.lora_a_initializer
                ),
                "lora_b_initializer": keras.initializers.serialize(
                    self.lora_b_initializer
                ),
                "magnitude_initializer": keras.initializers.serialize(
                    self.magnitude_initializer
                ),
                "kernel_regularizer": keras.regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": keras.regularizers.serialize(
                    self.bias_regularizer
                ),
                "activity_regularizer": keras.regularizers.serialize(
                    self.activity_regularizer
                ),
                "kernel_constraint": keras.constraints.serialize(
                    self.kernel_constraint
                ),
                "bias_constraint": keras.constraints.serialize(
                    self.bias_constraint
                ),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Create layer from configuration."""
        return cls(**config)

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        return input_shape[:-1] + (self.units,)


# Utility function to convert Dense layer to DoRADense
@keras_hub_export("keras_hub.layers.convert_dense_to_dora")
def convert_dense_to_dora(
    dense_layer,
    rank=4,
    alpha=1.0,
    dropout=0.0,
) -> DoRADense:
    """Convert a standard Dense layer to DoRADense layer.

    Args:
        dense_layer: The Dense layer to convert.
        rank: Rank for DoRA adaptation.
        alpha: Alpha parameter for DoRA.
        dropout: Dropout rate.

    Returns:
        DoRADense layer with pretrained weights loaded.
    """
    # Create DoRA layer with same configuration
    dora_layer = DoRADense(
        units=dense_layer.units,
        rank=rank,
        alpha=alpha,
        use_bias=dense_layer.use_bias,
        dropout=dropout,
        activation=dense_layer.activation,
        kernel_initializer=dense_layer.kernel_initializer,
        bias_initializer=dense_layer.bias_initializer,
        lora_a_initializer="he_uniform",
        lora_b_initializer="zeros",
        kernel_regularizer=dense_layer.kernel_regularizer,
        bias_regularizer=dense_layer.bias_regularizer,
        activity_regularizer=dense_layer.activity_regularizer,
        kernel_constraint=dense_layer.kernel_constraint,
        bias_constraint=dense_layer.bias_constraint,
        name=dense_layer.name + "_dora" if dense_layer.name else None,
    )

    # Build the DoRA layer if Dense layer is already built
    if dense_layer.built:
        # Build with the correct input shape from the dense layer
        input_shape = (None, dense_layer.kernel.shape[0])
        dora_layer.build(input_shape)
        # Load pretrained weights
        dora_layer.load_pretrained_weights(
            dense_layer.kernel,
            dense_layer.bias if dense_layer.use_bias else None,
        )

    return dora_layer
