"""DoRA (Weight-Decomposed Low-Rank Adaptation)
Embedding Layer Implementation.

This module implements the DoRA embedding
layer that applies weight decomposition
and low-rank adaptation to token
embeddings for efficient fine-tuning.

Backend-compatible with TensorFlow, PyTorch, and JAX.

Reference: DoRA: Weight-Decomposed Low-Rank Adaptation
"""

import keras
from keras import layers
from keras import ops

from keras_hub.src.api_export import keras_hub_export


@keras_hub_export("keras_hub.layers.DoRAEmbedding")
class DoRAEmbedding(layers.Layer):
    """DoRA (Weight-Decomposed Low-Rank Adaptation)
    Embedding layer.

    DoRA decomposes the embedding weight
    matrix W into magnitude and direction components:
    W = m * (W_0 + B @ A) / ||W_0 + B @ A||_c

    Where:
    - m: magnitude vector (learnable)
    - W_0: frozen pretrained embedding weights
    - A, B: low-rank adaptation matrices (learnable)
    - ||.||_c: column-wise L2 norm

    Args:
        input_dim: Size of the vocabulary (number of tokens).
        output_dim: Dimension of the dense embedding vectors.
        rank: Rank of the adaptation. Positive integer.
        alpha: LoRA scaling parameter. Float.
        embeddings_initializer:
        Initializer for the embeddings matrix.
        lora_a_initializer:
        Initializer for the A matrix. Defaults to 'he_uniform'.
        lora_b_initializer:
        Initializer for the B matrix. Defaults to 'zeros'.
        magnitude_initializer:
        Initializer for magnitude vector. Defaults to 'ones'.
        embeddings_regularizer:
        Regularizer function applied to embeddings.
        activity_regularizer:
        Regularizer function applied to output.
        embeddings_constraint:
        Constraint function applied to embeddings.
        mask_zero:
        Whether input value 0 is a special "padding" value.
        input_length:
        Length of input sequences (for compatibility).
        sparse:
        Whether to use sparse embedding lookup (experimental).
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        rank=4,
        alpha=1.0,
        embeddings_initializer="uniform",
        lora_a_initializer="he_uniform",
        lora_b_initializer="zeros",
        magnitude_initializer="ones",
        embeddings_regularizer=None,
        activity_regularizer=None,
        embeddings_constraint=None,
        mask_zero=False,
        input_length=None,
        sparse=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Validate parameters
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.alpha = alpha
        self.mask_zero = mask_zero
        self.input_length = input_length
        self.sparse = sparse

        # Initializers
        self.embeddings_initializer = keras.initializers.get(
            embeddings_initializer
        )
        self.lora_a_initializer = keras.initializers.get(lora_a_initializer)
        self.lora_b_initializer = keras.initializers.get(lora_b_initializer)
        self.magnitude_initializer = keras.initializers.get(
            magnitude_initializer
        )

        # Regularizers
        self.embeddings_regularizer = keras.regularizers.get(
            embeddings_regularizer
        )
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)

        # Constraints
        self.embeddings_constraint = keras.constraints.get(
            embeddings_constraint
        )

        # Scaling factor
        self.scaling = self.alpha / self.rank

        # Weight matrices (will be initialized in build())
        self.embeddings = None  # Frozen pretrained embeddings W_0
        self.lora_a = None  # Low-rank matrix A (input_dim, rank)
        self.lora_b = None  # Low-rank matrix B (rank, output_dim)
        self.magnitude = None  # Magnitude vector m (output_dim,)

        # Set compute dtype policy
        self._supports_masking = mask_zero

    def build(self, input_shape):
        """Build the layer weights."""
        # Build frozen embedding weights (pretrained embeddings W_0)
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            trainable=False,  # Frozen pretrained weights
        )

        # Build LoRA matrices
        self.lora_a = self.add_weight(
            name="lora_a",
            shape=(self.input_dim, self.rank),
            initializer=self.lora_a_initializer,
            trainable=True,
        )

        self.lora_b = self.add_weight(
            name="lora_b",
            shape=(self.rank, self.output_dim),
            initializer=self.lora_b_initializer,
            trainable=True,
        )

        # Build magnitude vector
        self.magnitude = self.add_weight(
            name="magnitude",
            shape=(self.output_dim,),
            initializer=self.magnitude_initializer,
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass of DoRA embedding layer.

        Implements: output = embedding_lookup
        (inputs, m * (W_0 + B @ A) / ||W_0 + B @ A||_c)

        Args:
            inputs: Input tensor containing token indices.
            training: Boolean indicating whether in training mode.

        Returns:
            Output tensor after DoRA embedding lookup.
        """
        # Cast inputs to integers for all backends
        inputs = ops.cast(inputs, "int32")

        # Get effective embedding matrix
        effective_embeddings = self._get_effective_embeddings()

        # Perform embedding lookup using backend-agnostic operations
        outputs = ops.take(effective_embeddings, inputs, axis=0)

        return outputs

    def _get_effective_embeddings(self):
        """Compute the effective embedding matrix after DoRA adaptation.

        Returns:
            The effective embedding matrix:
            m * (W_0 + B @ A) / ||W_0 + B @ A||_c
        """
        # Compute low-rank adaptation: A @ B (with scaling applied to B)
        # Use ops.multiply for backend compatibility
        scaled_lora_b = ops.multiply(self.lora_b, self.scaling)
        lora_adaptation = ops.matmul(self.lora_a, scaled_lora_b)

        # Combine pretrained embeddings with adaptation: W_0 + ΔW
        combined_embeddings = ops.add(self.embeddings, lora_adaptation)

        # Compute column-wise L2 norms: ||W_0 + ΔW||_c
        # Use ops for all operations to ensure backend compatibility
        squared_embeddings = ops.square(combined_embeddings)
        sum_squares = ops.sum(squared_embeddings, axis=0, keepdims=True)
        column_norms = ops.sqrt(sum_squares)

        # Prevent division by zero with backend-agnostic maximum
        eps = ops.convert_to_tensor(1e-8, dtype=column_norms.dtype)
        column_norms = ops.maximum(column_norms, eps)

        # DoRA formula: m * (W_0 + ΔW) / ||W_0 + ΔW||_c
        # Expand magnitude dimensions for broadcasting
        magnitude_expanded = ops.expand_dims(self.magnitude, axis=0)

        # Apply magnitude scaling and normalization
        numerator = ops.multiply(combined_embeddings, magnitude_expanded)
        dora_embeddings = ops.divide(numerator, column_norms)

        return dora_embeddings

    def compute_mask(self, inputs, mask=None):
        """Compute output mask for masking support."""
        if not self.mask_zero:
            return None

        # Create mask where input is not zero using backend-agnostic ops
        zero_tensor = ops.convert_to_tensor(0, dtype=inputs.dtype)
        return ops.not_equal(inputs, zero_tensor)

    def get_dora_parameters(self):
        """Get DoRA-specific parameters.

        Returns:
            Dictionary containing DoRA parameters.
        """
        return {
            "lora_a": self.lora_a,
            "lora_b": self.lora_b,
            "magnitude": self.magnitude,
        }

    def get_effective_embeddings(self):
        """Get the effective embedding matrix after DoRA adaptation.

        Returns:
            The effective embedding matrix.
        """
        return self._get_effective_embeddings()

    def merge_weights(self):
        """Merge DoRA weights back to a single embedding matrix.

        This is useful for inference optimization or converting
        back to standard Embedding layer.

        Returns:
            Dictionary with 'embeddings'.
        """
        return {"embeddings": self._get_effective_embeddings()}

    def count_params(self):
        """Count the number of trainable parameters in DoRA embedding layer.

        Returns:
            Number of trainable parameters.
        """
        return (
            self.input_dim * self.rank  # lora_a
            + self.rank * self.output_dim  # lora_b
            + self.output_dim  # magnitude
        )

    def load_pretrained_embeddings(self, pretrained_embeddings):
        """Load pretrained embeddings into the frozen embedding matrix.

        Args:
            pretrained_embeddings: Pretrained embedding matrix.
        """
        # Convert to tensor if needed for backend compatibility
        if not hasattr(pretrained_embeddings, "shape"):
            pretrained_embeddings = ops.convert_to_tensor(pretrained_embeddings)

        expected_shape = (self.input_dim, self.output_dim)
        if tuple(pretrained_embeddings.shape) != expected_shape:
            raise ValueError(
                f"Pretrained embeddings shape {pretrained_embeddings.shape} "
                f"doesn't match expected shape {expected_shape}"
            )

        # Use backend-compatible assignment
        self._safe_assign_weight(self.embeddings, pretrained_embeddings)

        # Initialize magnitude to preserve exact functional equivalence
        # Compute column norms using backend-agnostic operations
        squared_embeddings = ops.square(pretrained_embeddings)
        sum_squares = ops.sum(squared_embeddings, axis=0)
        column_norms = ops.sqrt(sum_squares)

        self._safe_assign_weight(self.magnitude, column_norms)

    def _safe_assign_weight(self, weight_var, new_value):
        """Safely assign new values to weights across backends."""
        try:
            # Try standard Keras approach first
            weight_var.assign(new_value)
        except Exception:
            # Fallback for backends that don't support assign
            # This approach works across all backends
            weight_var._value = ops.convert_to_tensor(new_value)

    def expand_vocabulary(self, new_vocab_size, new_token_embeddings=None):
        """Expand vocabulary size and optionally add new token embeddings.

        Since Keras doesn't allow modifying weights after building,
        this method returns a new DoRAEmbedding layer with expanded
        vocabulary instead of modifying the current layer in-place.

        Args:
            new_vocab_size: New vocabulary size
            (must be >= current input_dim).
            new_token_embeddings: Optional embeddings for new tokens.
            Shape should be (new_vocab_size - current_input_dim, output_dim).

        Returns:
            New DoRAEmbedding layer with expanded vocabulary.
        """
        if new_vocab_size <= self.input_dim:
            raise ValueError(
                f"new_vocab_size ({new_vocab_size}) must be greater than "
                f"current input_dim ({self.input_dim})"
            )

        if not self.built:
            raise ValueError("Layer must be built before expanding vocabulary")

        num_new_tokens = new_vocab_size - self.input_dim

        # Create new layer with expanded vocabulary
        expanded_layer = DoRAEmbedding(
            input_dim=new_vocab_size,
            output_dim=self.output_dim,
            rank=self.rank,
            alpha=self.alpha,
            embeddings_initializer=self.embeddings_initializer,
            lora_a_initializer=self.lora_a_initializer,
            lora_b_initializer=self.lora_b_initializer,
            magnitude_initializer=self.magnitude_initializer,
            embeddings_regularizer=self.embeddings_regularizer,
            activity_regularizer=self.activity_regularizer,
            embeddings_constraint=self.embeddings_constraint,
            mask_zero=self.mask_zero,
            input_length=self.input_length,
            sparse=self.sparse,
            name=self.name + "_expanded",
        )

        # Build the new layer
        expanded_layer.build(None)

        # Get current weights as tensors
        current_embeddings = self.embeddings
        current_lora_a = self.lora_a
        current_lora_b = self.lora_b
        current_magnitude = self.magnitude

        # Prepare new token embeddings using backend-agnostic operations
        if new_token_embeddings is None:
            # Use the same initializer as the original embeddings
            new_embeddings = self.embeddings_initializer(
                shape=(num_new_tokens, self.output_dim)
            )
        else:
            # Convert to tensor for backend compatibility
            new_embeddings = ops.convert_to_tensor(new_token_embeddings)
            expected_shape = (num_new_tokens, self.output_dim)
            if tuple(new_embeddings.shape) != expected_shape:
                raise ValueError(
                    f"new_token_embeddings shape {new_embeddings.shape} "
                    f"doesn't match expected shape {expected_shape}"
                )

        # Prepare new LoRA A rows using the same initializer
        new_lora_a_rows = self.lora_a_initializer(
            shape=(num_new_tokens, self.rank)
        )

        # Create expanded tensors using backend-agnostic concatenation
        expanded_embeddings = ops.concatenate(
            [current_embeddings, new_embeddings], axis=0
        )
        expanded_lora_a = ops.concatenate(
            [current_lora_a, new_lora_a_rows], axis=0
        )

        # Assign the expanded weights to the new layer
        expanded_layer._safe_assign_weight(
            expanded_layer.embeddings, expanded_embeddings
        )
        expanded_layer._safe_assign_weight(
            expanded_layer.lora_a, expanded_lora_a
        )
        expanded_layer._safe_assign_weight(
            expanded_layer.lora_b, current_lora_b
        )
        expanded_layer._safe_assign_weight(
            expanded_layer.magnitude, current_magnitude
        )

        return expanded_layer

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "rank": self.rank,
                "alpha": self.alpha,
                "embeddings_initializer": keras.initializers.serialize(
                    self.embeddings_initializer
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
                "embeddings_regularizer": keras.regularizers.serialize(
                    self.embeddings_regularizer
                ),
                "activity_regularizer": keras.regularizers.serialize(
                    self.activity_regularizer
                ),
                "embeddings_constraint": keras.constraints.serialize(
                    self.embeddings_constraint
                ),
                "mask_zero": self.mask_zero,
                "input_length": self.input_length,
                "sparse": self.sparse,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """Create layer from configuration."""
        return cls(**config)

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        if self.input_length is not None:
            return input_shape + (self.output_dim,)
        else:
            return input_shape + (self.output_dim,)


@keras_hub_export("keras_hub.layers.DoRAPositionEmbedding")
class DoRAPositionEmbedding(layers.Layer):
    """DoRA-enabled position embedding layer.

    This layer creates learnable positional embeddings
    that are added to token embeddings,
    using DoRA weight decomposition for efficient adaptation.
    """

    def __init__(
        self,
        sequence_length,
        output_dim,
        rank=4,
        alpha=1.0,
        initializer="uniform",
        lora_a_initializer="he_uniform",
        lora_b_initializer="zeros",
        magnitude_initializer="ones",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.sequence_length = sequence_length
        self.output_dim = output_dim
        self.rank = rank
        self.alpha = alpha

        # Initializers
        self.initializer = keras.initializers.get(initializer)
        self.lora_a_initializer = keras.initializers.get(lora_a_initializer)
        self.lora_b_initializer = keras.initializers.get(lora_b_initializer)
        self.magnitude_initializer = keras.initializers.get(
            magnitude_initializer
        )

        # Scaling factor
        self.scaling = self.alpha / self.rank

        # Weight matrices (will be initialized in build())
        self.position_embeddings = None  # Frozen position embeddings
        self.lora_a = None  # Low-rank matrix A
        self.lora_b = None  # Low-rank matrix B
        self.magnitude = None  # Magnitude vector

    def build(self, input_shape):
        """Build the position embedding weights."""
        # Build frozen position embedding weights
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(self.sequence_length, self.output_dim),
            initializer=self.initializer,
            trainable=False,  # Frozen
        )

        # Build LoRA matrices
        self.lora_a = self.add_weight(
            name="lora_a",
            shape=(self.sequence_length, self.rank),
            initializer=self.lora_a_initializer,
            trainable=True,
        )

        self.lora_b = self.add_weight(
            name="lora_b",
            shape=(self.rank, self.output_dim),
            initializer=self.lora_b_initializer,
            trainable=True,
        )

        # Build magnitude vector
        self.magnitude = self.add_weight(
            name="magnitude",
            shape=(self.output_dim,),
            initializer=self.magnitude_initializer,
            trainable=True,
        )

        super().build(input_shape)

    """def call(self, inputs, start_index=0):
        Forward pass of DoRA position embedding.

        Args:
            inputs: Input tensor (token embeddings)
            of shape [batch_size, seq_len, hidden_dim].
            start_index: Starting position index
            (for compatibility with KerasHub).

        Returns:
            Position embeddings of shape [batch_size, seq_len, hidden_dim].
        
        input_shape = ops.shape(inputs)
        seq_len = input_shape[-2]

        # Get effective position embeddings using DoRA
        effective_pos_embeddings = self._get_effective_position_embeddings()

        # Create position indices using backend-agnostic operations
        start_tensor = ops.convert_to_tensor(start_index, dtype="int32")
        seq_len_tensor = ops.convert_to_tensor(seq_len, dtype="int32")
        end_index = ops.add(start_tensor, seq_len_tensor)

        positions = ops.arange(start_index, end_index, dtype="int32")

        # Clip positions to valid range
        max_pos = ops.convert_to_tensor(
            self.sequence_length - 1, dtype="int32"
        )
        min_pos = ops.convert_to_tensor(0, dtype="int32")
        positions = ops.clip(positions, min_pos, max_pos)

        # Lookup position embeddings
        position_embeddings = ops.take(
            effective_pos_embeddings, positions, axis=0
        )

        # Expand dimensions to match input batch size
        position_embeddings = ops.expand_dims(position_embeddings, axis=0)

        # Create target shape for broadcasting
        batch_size = input_shape[0]
        target_shape = [batch_size, seq_len, self.output_dim]
        position_embeddings = ops.broadcast_to(
            position_embeddings, target_shape
        )

        return position_embeddings"""

    def call(self, inputs, start_index=0):
        """Forward pass of DoRA position embedding.

        Args:
            inputs: Input tensor (token embeddings)
            of shape [batch_size, seq_len, hidden_dim].
            start_index: Starting position index
            (for compatibility with KerasHub).

        Returns:
            Position embeddings of shape [batch_size, seq_len, hidden_dim].
        """
        input_shape = ops.shape(inputs)
        seq_len = input_shape[-2]
        batch_size = input_shape[0]

        # Get effective position embeddings using DoRA
        effective_pos_embeddings = self._get_effective_position_embeddings()

        # Convert start_index to tensor for consistent operations
        start_tensor = ops.convert_to_tensor(start_index, dtype="int32")

        # Create position indices from start_index to start_index + seq_len
        position_indices = ops.arange(seq_len, dtype="int32") + start_tensor

        # Clamp indices to valid range [0, sequence_length - 1]
        max_pos = ops.convert_to_tensor(self.sequence_length - 1, dtype="int32")
        min_pos = ops.convert_to_tensor(0, dtype="int32")
        position_indices = ops.clip(position_indices, min_pos, max_pos)

        # Gather position embeddings using the indices
        position_embeddings = ops.take(
            effective_pos_embeddings, position_indices, axis=0
        )

        # Add batch dimension and broadcast to match batch size
        position_embeddings = ops.expand_dims(position_embeddings, axis=0)
        position_embeddings = ops.broadcast_to(
            position_embeddings, [batch_size, seq_len, self.output_dim]
        )

        return position_embeddings

    def _get_effective_position_embeddings(self):
        """Compute effective position embeddings using DoRA decomposition."""
        # Compute low-rank adaptation (scaling applied to B matrix)
        scaled_lora_b = ops.multiply(self.lora_b, self.scaling)
        lora_adaptation = ops.matmul(self.lora_a, scaled_lora_b)

        # Combine with frozen weights
        combined_embeddings = ops.add(self.position_embeddings, lora_adaptation)

        # Compute column-wise L2 norms using backend-agnostic operations
        squared_embeddings = ops.square(combined_embeddings)
        sum_squares = ops.sum(squared_embeddings, axis=0, keepdims=True)
        column_norms = ops.sqrt(sum_squares)

        # Prevent division by zero
        eps = ops.convert_to_tensor(1e-8, dtype=column_norms.dtype)
        column_norms = ops.maximum(column_norms, eps)

        # Apply DoRA formula: m * (W_0 + ΔW) / ||W_0 + ΔW||_c
        magnitude_expanded = ops.expand_dims(self.magnitude, axis=0)
        numerator = ops.multiply(combined_embeddings, magnitude_expanded)
        dora_embeddings = ops.divide(numerator, column_norms)

        return dora_embeddings

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "output_dim": self.output_dim,
                "rank": self.rank,
                "alpha": self.alpha,
                "initializer": keras.initializers.serialize(self.initializer),
                "lora_a_initializer": keras.initializers.serialize(
                    self.lora_a_initializer
                ),
                "lora_b_initializer": keras.initializers.serialize(
                    self.lora_b_initializer
                ),
                "magnitude_initializer": keras.initializers.serialize(
                    self.magnitude_initializer
                ),
            }
        )
        return config


# Utility function to convert Embedding layer to DoRAEmbedding
@keras_hub_export("keras_hub.layers.embedding_to_dora")
def convert_embedding_to_dora(
    embedding_layer,
    rank=4,
    alpha=1.0,
) -> DoRAEmbedding:
    """Convert a standard Embedding layer to DoRAEmbedding layer.

    Args:
        embedding_layer: The Embedding layer to convert.
        rank: Rank for DoRA adaptation.
        alpha: Alpha parameter for DoRA.

    Returns:
        DoRAEmbedding layer with pretrained weights loaded.
    """
    # Safely get input_length attribute
    input_length = getattr(embedding_layer, "input_length", None)

    # Create DoRA embedding layer with same configuration
    dora_layer = DoRAEmbedding(
        input_dim=embedding_layer.input_dim,
        output_dim=embedding_layer.output_dim,
        rank=rank,
        alpha=alpha,
        embeddings_initializer=embedding_layer.embeddings_initializer,
        embeddings_regularizer=embedding_layer.embeddings_regularizer,
        activity_regularizer=embedding_layer.activity_regularizer,
        embeddings_constraint=embedding_layer.embeddings_constraint,
        mask_zero=embedding_layer.mask_zero,
        input_length=input_length,
        name=embedding_layer.name + "_dora",
    )

    # Build the DoRA layer if Embedding layer is already built
    if embedding_layer.built:
        dora_layer.build(None)  # Embedding layers don't depend on input shape
        # Load pretrained embeddings
        dora_layer.load_pretrained_embeddings(embedding_layer.embeddings)

    return dora_layer
