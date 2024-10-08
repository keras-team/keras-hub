import keras

from keras_hub.src.models.flux.flux_layers import DoubleStreamBlock
from keras_hub.src.models.flux.flux_layers import EmbedND
from keras_hub.src.models.flux.flux_layers import LastLayer
from keras_hub.src.models.flux.flux_layers import MLPEmbedder
from keras_hub.src.models.flux.flux_layers import SingleStreamBlock
from keras_hub.src.models.flux.flux_maths import TimestepEmbedding


class Flux(keras.Model):
    """
    Transformer model for flow matching on sequences,
    utilizing a double-stream and single-stream block structure.

    The model processes image and text data with associated positional and timestep
    embeddings, and optionally applies guidance embedding. Double-stream blocks
    handle separate image and text streams, while single-stream blocks combine
    these streams. Ported from: https://github.com/black-forest-labs/flux

    Args:
        input_channels: int. The number of input channels.
        hidden_size: int. The hidden size of the transformer, must be divisible by `num_heads`.
        mlp_ratio: float. The ratio of the MLP dimension to the hidden size.
        num_heads: int. The number of attention heads.
        depth: int. The number of double-stream blocks.
        depth_single_blocks: int. The number of single-stream blocks.
        axes_dim: list[int]. A list of dimensions for the positional embedding axes.
        theta: int. The base frequency for positional embeddings.
        use_bias: bool. Whether to apply bias to the query, key, and value projections.
        guidance_embed: bool. If True, applies guidance embedding in the model.

    Raises:
        ValueError: If `hidden_size` is not divisible by `num_heads`, or if `sum(axes_dim)` is not equal to the
                    positional embedding dimension.
    """

    def __init__(
        self,
        input_channels,
        hidden_size,
        mlp_ratio,
        num_heads,
        depth,
        depth_single_blocks,
        axes_dim,
        theta,
        use_bias,
        guidance_embed,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = self.input_channels
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(
                f"Got {axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.positional_embedder = EmbedND(
            dim=pe_dim, theta=theta, axes_dim=axes_dim
        )
        self.image_input_embedder = keras.layers.Dense(
            self.hidden_size, use_bias=True
        )
        self.time_input_embedder = MLPEmbedder(hidden_dim=self.hidden_size)
        self.vector_embedder = MLPEmbedder(hidden_dim=self.hidden_size)
        self.guidance_input_embedder = (
            MLPEmbedder(hidden_dim=self.hidden_size)
            if guidance_embed
            else keras.layers.Identity()
        )
        self.text_input_embedder = keras.layers.Dense(self.hidden_size)

        self.double_blocks = [
            DoubleStreamBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=mlp_ratio,
                use_bias=use_bias,
            )
            for _ in range(depth)
        ]

        self.single_blocks = [
            SingleStreamBlock(
                self.hidden_size, self.num_heads, mlp_ratio=mlp_ratio
            )
            for _ in range(depth_single_blocks)
        ]

        self.final_layer = LastLayer(self.hidden_size, 1, self.output_channels)
        self.timestep_embedding = TimestepEmbedding()
        self.guidance_embed = guidance_embed

    def build(self, input_shape):
        (
            image_shape,
            image_ids_shape,
            text_shape,
            text_ids_shape,
            timestep_shape,
            y_shape,
            guidance_shape,
        ) = input_shape

        # Build input layers
        self.image_input_embedder.build(image_shape)
        self.text_input_embedder.build(text_shape)

        # Build timestep embedding and vector inputs
        self.timestep_embedding.build(timestep_shape)
        self.time_input_embedder.build(
            (None, 256)
        )  # timestep embedding size is 256
        self.vector_embedder.build(y_shape)

        if self.guidance_embed:
            if guidance_shape is None:
                raise ValueError(
                    "Guidance shape must be provided for guidance-distilled model."
                )
            self.guidance_input_embedder.build(
                (None, 256)
            )  # guidance embedding size is 256

        # Build positional embedder
        ids_shape = (
            None,
            image_ids_shape[1] + text_ids_shape[1],
            image_ids_shape[2],
        )
        self.positional_embedder.build(ids_shape)

        # Build double stream blocks
        for block in self.double_blocks:
            block.build((image_shape, text_shape, (None, 256), ids_shape))

        # Build single stream blocks
        concat_image_shape = (
            None,
            image_shape[1] + text_shape[1],
            self.hidden_size,
        )  # Concatenated shape
        for block in self.single_blocks:
            block.build((concat_image_shape, (None, 256), ids_shape))

        # Build final layer
        # Adjusted to match expected input shape for the final layer
        self.final_layer.build(
            (None, image_shape[1] + text_shape[1], self.hidden_size)
        )  # Concatenated shape

        self.built = True  # Mark as built

    def call(
        self,
        image,
        image_ids,
        text,
        text_ids,
        timesteps,
        y,
        guidance=None,
    ):
        """
        Forward pass through the Flux model.

        Args:
            image: KerasTensor. Image input tensor of shape (N, L, D) where N is the batch size,
                 L is the sequence length, and D is the feature dimension.
            image_ids: KerasTensor. Image ID input tensor of shape (N, L, D) corresponding
                 to the image sequences.
            text: KerasTensor. Text input tensor of shape (N, L, D).
            text_ids: KerasTensor. Text ID input tensor of shape (N, L, D) corresponding
                to the text sequences.
            timesteps: KerasTensor. Timestep tensor used to compute positional embeddings.
            y: KerasTensor. Additional vector input, such as target values.
            guidance: KerasTensor, optional. Guidance input tensor used
                in guidance-embedded models.

        Returns:
            KerasTensor: The output tensor of the model, processed through
            double and single stream blocks and the final layer.
        """
        if image.ndim != 3 or text.ndim != 3:
            raise ValueError(
                "Input image and text tensors must have 3 dimensions."
            )

        # running on sequences image
        image = self.image_input_embedder(image)
        vec = self.time_input_embedder(
            self.timestep_embedding(timesteps, dim=256)
        )
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_input_embedder(
                self.timestep_embedding(guidance, dim=256)
            )
        vec = vec + self.vector_embedder(y)
        text = self.text_input_embedder(text)

        ids = keras.ops.concatenate((text_ids, image_ids), axis=1)
        pe = self.positional_embedder(ids)

        for block in self.double_blocks:
            image, text = block(image=image, text=text, vec=vec, pe=pe)

        image = keras.ops.concatenate((text, image), axis=1)
        for block in self.single_blocks:
            image = block(image, vec=vec, pe=pe)
        image = image[:, text.shape[1] :, ...]

        image = self.final_layer(
            image, vec
        )  # (N, T, patch_size ** 2 * output_channels)
        return image
