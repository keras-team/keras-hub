import keras
from keras import layers
from keras import ops


class Encoder(layers.Layer):
    """A simple feed-forward encoder with ReLU activations.

    This layer consists of a sequence of Dense layers with ReLU activation,
    followed by a final Dense layer with no activation.

    Args:
        layer_dims: A list of integers specifying the size of each hidden Dense
            layer.
        output_dim: Integer. The size of the output Dense layer.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.

    Example:
    >>> encoder = Encoder(layer_dims=[64, 32], output_dim=16)
    >>> x = keras.random.uniform(shape=(1, 10))
    >>> output = encoder(x)
    >>> tuple(output.shape)
    (1, 16)
    """

    def __init__(self, layer_dims, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.layer_dims = layer_dims
        self.output_dim = output_dim
        self.dense_layers = []
        for dim in layer_dims:
            self.dense_layers.append(layers.Dense(dim, activation="relu"))
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return self.output_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layer_dims": self.layer_dims,
                "output_dim": self.output_dim,
            }
        )
        return config


class Decoder(layers.Layer):
    """A simple feed-forward decoder with ReLU activations.

    This layer consists of a sequence of Dense layers with ReLU activation,
    followed by a final Dense layer with no activation.

    Args:
        layer_dims: A list of integers specifying the size of each hidden Dense
            layer.
        output_dim: Integer. The size of the output Dense layer.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.

    Example:
    >>> decoder = Decoder(layer_dims=[32, 64], output_dim=10)
    >>> x = keras.random.uniform(shape=(1, 16))
    >>> output = decoder(x)
    >>> tuple(output.shape)
    (1, 10)
    """

    def __init__(self, layer_dims, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.layer_dims = layer_dims
        self.output_dim = output_dim
        self.dense_layers = []
        for dim in layer_dims:
            self.dense_layers.append(layers.Dense(dim, activation="relu"))
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return self.output_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layer_dims": self.layer_dims,
                "output_dim": self.output_dim,
            }
        )
        return config


class VectorQuantizerEMA(layers.Layer):
    """Vector Quantizer with Exponential Moving Average (EMA) updates.

    This layer implements a vector quantization module using EMA to update
    states, which stabilizes the training process compared to codebook collapse.
    It takes an input tensor, flattens it, and maps each vector to the nearest
    element in a codebook (embeddings).

    Args:
        num_embeddings: Integer. The number of embeddings in the codebook.
        embedding_dim: Integer. The dimensionality of each embedding vector.
        decay: Float. The decay rate for the EMA updates. Defaults to `0.99`.
        eps: Float. A small epsilon value for numerical stability to avoid
            division by zero. Defaults to `1e-5`.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.

    References:
        - [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)

    Example:
    >>> vq = VectorQuantizerEMA(num_embeddings=10, embedding_dim=16)
    >>> x = keras.random.uniform(shape=(1, 5, 16))
    >>> quantized, encodings, usage_ratio, loss = vq(x)
    >>> tuple(quantized.shape)
    (1, 5, 16)
    """

    def __init__(
        self, num_embeddings, embedding_dim, decay=0.99, eps=1e-5, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.num_embeddings, self.embedding_dim),
            initializer="random_normal",
            trainable=False,
            name="embeddings",
        )
        self.ema_cluster_size = self.add_weight(
            shape=(self.num_embeddings,),
            initializer="zeros",
            trainable=False,
            name="ema_cluster_size",
        )
        self.ema_w = self.add_weight(
            shape=(self.num_embeddings, self.embedding_dim),
            initializer="random_normal",
            trainable=False,
            name="ema_w",
        )

    def _codebook_usage(self, encodings):
        usage_counts = ops.sum(encodings, axis=0)  # (num_embeddings,)
        num_used = ops.sum(ops.cast(usage_counts > 0, "float32"))
        return num_used / self.num_embeddings

    def call(self, inputs, training=False):
        input_shape = ops.shape(inputs)
        # Flatten inputs to (N, D)
        flattened_inputs = ops.reshape(inputs, (-1, self.embedding_dim))

        # Distances: x^2 + c^2 - 2xc
        # inputs: (N, D), codebook: (E, D)
        input_sq = ops.sum(flattened_inputs**2, axis=1, keepdims=True)
        codebook_sq = ops.sum(self.embeddings**2, axis=1)  # (E,)
        dot_product = ops.matmul(
            flattened_inputs, ops.transpose(self.embeddings)
        )  # (N, E)

        distances = input_sq + codebook_sq - 2 * dot_product

        # Encoding
        encoding_indices = ops.argmin(distances, axis=-1)
        encodings = ops.one_hot(encoding_indices, self.num_embeddings)  # (N, E)

        # Quantize
        quantized_flat = ops.take(self.embeddings, encoding_indices, axis=0)

        if training:
            # EMA Update
            current_counts = ops.sum(encodings, axis=0)
            updated_ema_cluster_size = (
                self.ema_cluster_size * self.decay
                + (1.0 - self.decay) * current_counts
            )

            # Laplace smoothing
            n = ops.sum(updated_ema_cluster_size)
            updated_ema_cluster_size = (
                (updated_ema_cluster_size + self.eps)
                / (n + self.num_embeddings * self.eps)
                * n
            )

            self.ema_cluster_size.assign(updated_ema_cluster_size)

            # total_assignment_sums = encoding.T @ inputs -> (E, D)
            total_assignment_sums = ops.matmul(
                ops.transpose(encodings), flattened_inputs
            )

            updated_ema_w = (
                self.ema_w * self.decay
                + (1.0 - self.decay) * total_assignment_sums
            )
            self.ema_w.assign(updated_ema_w)

            updated_embeddings = self.ema_w / (
                ops.expand_dims(self.ema_cluster_size, axis=1) + self.eps
            )
            self.embeddings.assign(updated_embeddings)

            # Quantization loss
            quantization_loss = ops.mean(
                (flattened_inputs - quantized_flat) ** 2
            )
            quantization_loss = ops.reshape(quantization_loss, (1,))

            # STE
            quantized = ops.reshape(quantized_flat, input_shape)
            quantized_flow = inputs + ops.stop_gradient(quantized - inputs)

            usage_ratio = self._codebook_usage(encodings)

            return quantized_flow, encodings, usage_ratio, quantization_loss
        else:
            quantized = ops.reshape(quantized_flat, input_shape)
            usage_ratio = self._codebook_usage(encodings)
            quantization_loss = ops.convert_to_tensor(0.0, dtype=inputs.dtype)
            quantization_loss = ops.reshape(quantization_loss, (1,))
            return quantized, encodings, usage_ratio, quantization_loss

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_embeddings": self.num_embeddings,
                "embedding_dim": self.embedding_dim,
                "decay": self.decay,
                "eps": self.eps,
            }
        )
        return config


class ResidualVectorQuantizer(layers.Layer):
    """A Residual Vector Quantizer.

    This layer applies a sequence of vector quantizers to the residual of the
    input. The first quantizer quantizes the input, the second quantizer
    quantizes the error (residual) from the first, and so on.

    Args:
        quantizers: A list of `VectorQuantizerEMA` instances (or compatible
            layers) to be applied sequentially.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.

    References:
        - [SoundStream: An End-to-End Neural Audio Codec](https://arxiv.org/abs/2107.03312)

    Example:
    >>> vq1 = VectorQuantizerEMA(num_embeddings=10, embedding_dim=16)
    >>> vq2 = VectorQuantizerEMA(num_embeddings=10, embedding_dim=16)
    >>> rvq = ResidualVectorQuantizer(quantizers=[vq1, vq2])
    >>> x = keras.random.uniform(shape=(1, 5, 16))
    >>> quantized_sum, encodings, usage_ratios, loss = rvq(x)
    >>> tuple(quantized_sum.shape)
    (1, 5, 16)
    """

    def __init__(self, quantizers, **kwargs):
        super().__init__(**kwargs)
        # quantizers should be a list of layer instances or configs?
        # Typically in Keras we pass instances.
        self.quantizers = quantizers

    def call(self, inputs, training=False):
        quantized_list = []
        encodings_list = []
        usage_ratios_list = []
        residual = inputs
        total_quantization_loss = ops.convert_to_tensor(0.0, dtype=inputs.dtype)

        for quantizer in self.quantizers:
            # Always returns 4 values now
            (
                current_quantized,
                current_encoding,
                usage_ratio,
                quantization_loss,
            ) = quantizer(residual, training=training)
            total_quantization_loss = (
                total_quantization_loss + quantization_loss
            )

            quantized_list.append(current_quantized)
            residual = residual - current_quantized
            encodings_list.append(current_encoding)
            usage_ratios_list.append(usage_ratio)

        # Stack results
        # quantized: sum of all quantized
        # encodings: stack
        # usage_ratios: stack

        quantized_sum = sum(quantized_list)  # Element-wise sum
        # ops.stack needs a list of tensors
        encodings = ops.stack(encodings_list, axis=0)
        # usage_ratios is list of scalars (tensors of rank 0) or simple tensors.
        # ops.stack works.
        usage_ratios = ops.stack(usage_ratios_list, axis=0)

        return quantized_sum, encodings, usage_ratios, total_quantization_loss

    def get_config(self):
        config = super().get_config()
        quantizers_config = []
        for q in self.quantizers:
            quantizers_config.append(keras.utils.serialize_keras_object(q))
        config.update(
            {
                "quantizers": quantizers_config,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        quantizers_config = config.pop("quantizers")
        quantizers = [
            keras.utils.deserialize_keras_object(q) for q in quantizers_config
        ]
        return cls(quantizers=quantizers, **config)
